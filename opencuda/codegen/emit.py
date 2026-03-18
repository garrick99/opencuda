"""
OpenCUDA codegen — lower IR to PTX text, then compile via OpenPTXas.

Strategy: IR → PTX text → OpenPTXas pipeline → cubin.
This reuses OpenPTXas's full backend (parser, regalloc, isel, scoreboard, emitter).
"""

from __future__ import annotations
from ..ir.nodes import (Module, Kernel, BasicBlock, Value, Const, Operand,
                         BinInst, CmpInst, LoadInst, StoreInst, CvtInst,
                         CallInst, ParamInst,
                         BinOp, CmpOp,
                         RetTerm, BrTerm, CondBrTerm)
from ..ir.types import (Type, ScalarTy, PtrTy, ScalarType, AddrSpace,
                         INT32, UINT32, FLOAT, VOID)


def _ptx_type(ty: Type) -> str:
    """Convert IR type to PTX type string."""
    if isinstance(ty, PtrTy):
        return 'u64'
    if isinstance(ty, ScalarTy):
        mapping = {
            ScalarType.VOID: 'u32',
            ScalarType.INT32: 's32', ScalarType.UINT32: 'u32',
            ScalarType.INT64: 's64', ScalarType.UINT64: 'u64',
            ScalarType.FLOAT: 'f32', ScalarType.DOUBLE: 'f64',
            ScalarType.HALF: 'f16',
        }
        return mapping.get(ty.scalar, 'u32')
    return 'u32'


def _ptx_reg_prefix(ty: Type) -> str:
    """Get PTX register prefix for a type."""
    if isinstance(ty, PtrTy):
        return 'rd'
    if isinstance(ty, ScalarTy):
        if ty.size == 8:
            return 'rd'
        if ty.is_float:
            return 'f'
        return 'r'
    return 'r'


def _is_ptr(ty: Type) -> bool:
    return isinstance(ty, PtrTy)


def _is_float(ty: Type) -> bool:
    return isinstance(ty, ScalarTy) and ty.is_float


def _is_64bit(ty: Type) -> bool:
    if isinstance(ty, PtrTy):
        return True
    if isinstance(ty, ScalarTy):
        return ty.size == 8
    return False


class PTXEmitter:
    """Emits PTX text from IR."""

    def __init__(self):
        self._lines: list[str] = []
        self._reg_counts: dict[str, int] = {}  # prefix → max count
        self._pred_map: dict[int, int] = {}  # Value.id → predicate register index
        self._next_pred: int = 0

    def _reg(self, v: Value) -> str:
        prefix = _ptx_reg_prefix(v.ty)
        name = f'%{prefix}{v.id}'
        self._reg_counts[prefix] = max(self._reg_counts.get(prefix, 0), v.id + 1)
        return name

    def _operand(self, op: Operand, force_type: str = None) -> str:
        if isinstance(op, Value):
            return self._reg(op)
        if isinstance(op, Const):
            # Determine if we should emit as float or int
            is_fp = force_type in ('f32', 'f64') if force_type else (
                isinstance(op.ty, ScalarTy) and op.ty.is_float)
            if is_fp:
                return f'0f{self._float_hex(float(op.value))}'
            return str(int(op.value))
        return str(op)

    def _float_hex(self, f: float) -> str:
        import struct
        return struct.pack('>f', f).hex().upper()

    def emit_kernel(self, kernel: Kernel) -> str:
        self._lines = []
        self._reg_counts = {}
        self._pred_map = {}
        self._next_pred = 0
        self._shared_val_ids: dict[str, list] = {}  # smem_name → [Value, ...]

        # Pre-scan: find Values that are shared memory variables
        if hasattr(kernel, '_shared_decls'):
            smem_names = {s[0] for s in kernel._shared_decls}
            for bb in kernel.blocks:
                for inst in bb.instructions:
                    # Find BinInst where lhs or rhs is a shared-memory Value
                    if hasattr(inst, 'lhs') and isinstance(inst.lhs, Value):
                        if inst.lhs.name in smem_names:
                            self._shared_val_ids.setdefault(inst.lhs.name, []).append(inst.lhs)
                    if hasattr(inst, 'dest') and isinstance(inst.dest, Value):
                        if inst.dest.name in smem_names:
                            self._shared_val_ids.setdefault(inst.dest.name, []).append(inst.dest)

        # First pass: collect all register usage
        body_lines = []
        self._lines = body_lines
        for bb in kernel.blocks:
            self._emit_block(bb, kernel)

        # Build the full PTX
        ptx = []
        ptx.append('.version 9.0')
        ptx.append('.target sm_120')
        ptx.append('.address_size 64')
        ptx.append('')

        # Kernel signature
        params = ', '.join(
            f'.param .{_ptx_type(p.ty)} {p.name}' for p in kernel.params
        )
        ptx.append(f'.visible .entry {kernel.name}(')
        ptx.append(f'    {params})')
        ptx.append('{')

        # Shared memory declarations
        if hasattr(kernel, '_shared_decls'):
            for sname, sty, scount in kernel._shared_decls:
                ptx_sty = _ptx_type(sty)
                ptx.append(f'    .shared .{ptx_sty} {sname}[{scount}];')

        # Register declarations
        for prefix, count in sorted(self._reg_counts.items()):
            if prefix == 'rd':
                ptx.append(f'    .reg .b64 %{prefix}<{count}>;')
            elif prefix == 'f':
                ptx.append(f'    .reg .f32 %{prefix}<{count}>;')
            else:
                ptx.append(f'    .reg .b32 %{prefix}<{count}>;')
        pred_count = max(self._next_pred, 1)
        ptx.append(f'    .reg .pred %p<{pred_count}>;')
        ptx.append('')

        # Initialize shared memory base addresses
        # Insert mov.u64 %rd, smem_name for each shared variable
        if hasattr(kernel, '_shared_decls'):
            smem_inits = []
            for sname, sty, scount in kernel._shared_decls:
                # Find all Values that reference this shared variable
                for val in self._shared_val_ids.get(sname, []):
                    reg = self._reg(val)
                    smem_inits.append(f'    mov.u64 {reg}, {sname};')
            body_lines = smem_inits + body_lines

        # Body
        ptx.extend(body_lines)

        ptx.append('}')
        return '\n'.join(ptx)

    def _emit_block(self, bb: BasicBlock, kernel: Kernel):
        if bb.label != 'entry':
            self._lines.append(f'{bb.label}:')

        for inst in bb.instructions:
            self._emit_inst(inst, kernel)

        if bb.terminator:
            self._emit_term(bb.terminator)

    def _emit_inst(self, inst, kernel: Kernel):
        if isinstance(inst, ParamInst):
            ty = kernel.params[inst.param_index].ty
            ptx_ty = _ptx_type(ty)
            self._lines.append(
                f'    ld.param.{ptx_ty} {self._reg(inst.dest)}, [{inst.param_name}];')

        elif isinstance(inst, BinInst):
            ty = inst.dest.ty
            ptx_ty = _ptx_type(ty)
            op_map = {
                BinOp.ADD: 'add', BinOp.SUB: 'sub', BinOp.MUL: 'mul.lo',
                BinOp.DIV: 'div', BinOp.MOD: 'rem',
                BinOp.AND: 'and', BinOp.OR: 'or', BinOp.XOR: 'xor',
                BinOp.SHL: 'shl', BinOp.SHR: 'shr',
            }
            ptx_op = op_map.get(inst.op, 'add')
            # Float mul doesn't need .lo qualifier
            if inst.op == BinOp.MUL and _is_float(ty):
                ptx_op = 'mul'
            # Bitwise ops use .b32 type, not .s32/.u32
            if inst.op in (BinOp.AND, BinOp.OR, BinOp.XOR, BinOp.SHL, BinOp.SHR):
                ptx_ty = f'b{ty.size * 8}' if isinstance(ty, ScalarTy) else 'b32'

            # Pointer arithmetic: use u64 for add/sub
            if _is_ptr(ty) and inst.op in (BinOp.ADD, BinOp.SUB):
                # Need to widen the integer operand to u64
                lhs = self._operand(inst.lhs)
                rhs = self._operand(inst.rhs)
                if isinstance(inst.rhs, (Value, Const)) and not _is_64bit(inst.rhs.ty if isinstance(inst.rhs, Value) else INT32):
                    # Widen rhs to u64
                    wide = kernel.new_value(f'wide{inst.dest.id}', ty)
                    self._reg_counts[_ptx_reg_prefix(ty)] = max(
                        self._reg_counts.get(_ptx_reg_prefix(ty), 0), wide.id + 1)
                    self._lines.append(
                        f'    cvt.u64.u32 {self._reg(wide)}, {rhs};')
                    rhs = self._reg(wide)
                self._lines.append(
                    f'    {ptx_op}.u64 {self._reg(inst.dest)}, {lhs}, {rhs};')
            elif _is_float(ty):
                self._lines.append(
                    f'    {ptx_op}.f32 {self._reg(inst.dest)}, '
                    f'{self._operand(inst.lhs, "f32")}, {self._operand(inst.rhs, "f32")};')
            elif _is_64bit(ty):
                self._lines.append(
                    f'    {ptx_op}.{ptx_ty} {self._reg(inst.dest)}, '
                    f'{self._operand(inst.lhs, ptx_ty)}, {self._operand(inst.rhs, ptx_ty)};')
            else:
                self._lines.append(
                    f'    {ptx_op}.{ptx_ty} {self._reg(inst.dest)}, '
                    f'{self._operand(inst.lhs, ptx_ty)}, {self._operand(inst.rhs, ptx_ty)};')

        elif isinstance(inst, CmpInst):
            ty = inst.lhs.ty if isinstance(inst.lhs, Value) else INT32
            ptx_ty = _ptx_type(ty)
            op_map = {
                CmpOp.LT: 'lt', CmpOp.LE: 'le', CmpOp.GT: 'gt',
                CmpOp.GE: 'ge', CmpOp.EQ: 'eq', CmpOp.NE: 'ne',
            }
            cmp_str = op_map[inst.op]
            # Allocate a predicate register (separate numbering from GPRs)
            if inst.dest.id not in self._pred_map:
                self._pred_map[inst.dest.id] = self._next_pred
                self._next_pred += 1
            pred_idx = self._pred_map[inst.dest.id]
            pred = f'%p{pred_idx}'
            self._lines.append(
                f'    setp.{cmp_str}.{ptx_ty} {pred}, '
                f'{self._operand(inst.lhs, ptx_ty)}, {self._operand(inst.rhs, ptx_ty)};')

        elif isinstance(inst, LoadInst):
            ty = inst.dest.ty
            ptx_ty = _ptx_type(ty)
            addr_space = 'global'
            if isinstance(inst.addr, Value) and isinstance(inst.addr.ty, PtrTy):
                if inst.addr.ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
            self._lines.append(
                f'    ld.{addr_space}.{ptx_ty} {self._reg(inst.dest)}, '
                f'[{self._operand(inst.addr)}];')

        elif isinstance(inst, StoreInst):
            ty = inst.value.ty if isinstance(inst.value, Value) else INT32
            ptx_ty = _ptx_type(ty)
            addr_space = 'global'
            if isinstance(inst.addr, Value) and isinstance(inst.addr.ty, PtrTy):
                if inst.addr.ty.addr_space == AddrSpace.SHARED:
                    addr_space = 'shared'
            self._lines.append(
                f'    st.{addr_space}.{ptx_ty} [{self._operand(inst.addr)}], '
                f'{self._operand(inst.value, ptx_ty)};')

        elif isinstance(inst, CallInst):
            if inst.func.startswith('atomic'):
                # Atomic operations: atomicAdd(addr, val) → atom.global.add.type
                atomic_ops = {
                    'atomicAdd': 'add', 'atomicSub': 'add',  # sub emitted as add with negated val
                    'atomicMin': 'min', 'atomicMax': 'max',
                    'atomicAnd': 'and', 'atomicOr': 'or', 'atomicXor': 'xor',
                    'atomicExch': 'exch', 'atomicCAS': 'cas',
                }
                ptx_op = atomic_ops.get(inst.func, 'add')
                addr = self._operand(inst.args[0]) if inst.args else '%rd0'
                val = self._operand(inst.args[1]) if len(inst.args) > 1 else '0'
                # Determine type from the value
                val_ty = 'u32'
                if len(inst.args) > 1 and isinstance(inst.args[1], Value):
                    val_ty = _ptx_type(inst.args[1].ty)
                elif len(inst.args) > 1 and isinstance(inst.args[1], Const):
                    if isinstance(inst.args[1].value, float):
                        val_ty = 'f32'
                dest = self._reg(inst.dest) if inst.dest else '%r0'
                self._lines.append(
                    f'    atom.global.{ptx_op}.{val_ty} {dest}, [{addr}], {self._operand(inst.args[1], val_ty)};')
            elif inst.func == '__syncthreads':
                self._lines.append('    bar.sync 0;')
            elif inst.func in ('threadIdx.x', 'threadIdx.y', 'threadIdx.z'):
                # Special register read
                sr_map = {'threadIdx.x': 'tid.x', 'threadIdx.y': 'tid.y',
                          'threadIdx.z': 'tid.z'}
                sr = sr_map[inst.func]
                self._lines.append(
                    f'    mov.u32 {self._reg(inst.dest)}, %{sr};')
            elif inst.func in ('blockIdx.x', 'blockIdx.y', 'blockIdx.z'):
                sr_map = {'blockIdx.x': 'ctaid.x', 'blockIdx.y': 'ctaid.y',
                          'blockIdx.z': 'ctaid.z'}
                sr = sr_map[inst.func]
                self._lines.append(
                    f'    mov.u32 {self._reg(inst.dest)}, %{sr};')
            elif inst.func in ('blockDim.x', 'blockDim.y', 'blockDim.z'):
                sr_map = {'blockDim.x': 'ntid.x', 'blockDim.y': 'ntid.y',
                          'blockDim.z': 'ntid.z'}
                sr = sr_map[inst.func]
                self._lines.append(
                    f'    mov.u32 {self._reg(inst.dest)}, %{sr};')

    def _emit_term(self, term):
        if isinstance(term, RetTerm):
            self._lines.append('    ret;')
        elif isinstance(term, BrTerm):
            self._lines.append(f'    bra {term.target};')
        elif isinstance(term, CondBrTerm):
            if isinstance(term.cond, Value) and term.cond.id in self._pred_map:
                pred = f'%p{self._pred_map[term.cond.id]}'
            else:
                pred = '%p0'
            self._lines.append(f'    @{pred} bra {term.true_bb};')
            self._lines.append(f'    bra {term.false_bb};')


def ir_to_ptx(module: Module) -> dict[str, str]:
    """Convert IR module to PTX text for each kernel."""
    emitter = PTXEmitter()
    result = {}
    for kernel in module.kernels:
        result[kernel.name] = emitter.emit_kernel(kernel)
    return result
