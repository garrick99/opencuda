"""
OpenCUDA IR optimization passes.

Pass 1: Constant folding — evaluate Const op Const at compile time.
Pass 2: CSE — reuse identical computations within a basic block.

SAFETY RULE: Replacements never cross basic block boundaries.
This prevents the loop writeback bug where a variable initialized
in the entry block (float sum = 0) gets replaced by Const(0) in
the loop body, causing the loop condition to never see updates.
"""

from __future__ import annotations
from ..ir.nodes import (Module, Kernel, BasicBlock, Value, Const, Operand,
                         BinInst, CmpInst, LoadInst, StoreInst, CvtInst,
                         CallInst, ParamInst, BinOp, CmpOp)
from ..ir.types import INT32, UINT32, FLOAT, ScalarTy


def _const_val(op: Operand):
    if isinstance(op, Const):
        return op.value
    return None


def _fold_bin(op: BinOp, a, b, is_float: bool):
    if a is None or b is None:
        return None
    try:
        if is_float:
            a, b = float(a), float(b)
        else:
            a, b = int(a), int(b)
        if op == BinOp.ADD: return a + b
        if op == BinOp.SUB: return a - b
        if op == BinOp.MUL: return a * b
        if op == BinOp.DIV and b != 0: return a // b if not is_float else a / b
        if op == BinOp.MOD and b != 0: return a % b
        if op == BinOp.AND: return int(a) & int(b)
        if op == BinOp.OR:  return int(a) | int(b)
        if op == BinOp.XOR: return int(a) ^ int(b)
        if op == BinOp.SHL: return int(a) << int(b)
        if op == BinOp.SHR: return int(a) >> int(b)
    except:
        pass
    return None


def constant_fold(kernel: Kernel) -> int:
    """
    Fold Const op Const into a single Const.
    Also folds safe identity ops: x * 0 → 0.

    SAFETY: Replacements are LOCAL to each basic block. A fold in the
    entry block does NOT propagate to the loop body. This prevents the
    loop writeback bug.

    SAFETY: We never put Const results into the cross-instruction
    replacement map. A folded instruction is simply removed; its
    Value ceases to exist. Only Value→Value replacements (identity
    folds where dest and replacement are both registers) propagate.
    """
    folded = 0

    for bb in kernel.blocks:
        # Per-block replacement map — does NOT leak to other blocks
        replacements: dict[int, Operand] = {}

        def _resolve(op: Operand) -> Operand:
            if isinstance(op, Value) and op.id in replacements:
                r = replacements[op.id]
                # Follow chains but only for Value→Value
                while isinstance(r, Value) and r.id in replacements:
                    r = replacements[r.id]
                return r
            return op

        new_insts = []
        for inst in bb.instructions:
            if isinstance(inst, BinInst):
                lhs = _resolve(inst.lhs)
                rhs = _resolve(inst.rhs)
                inst.lhs = lhs
                inst.rhs = rhs

                is_float = isinstance(inst.dest.ty, ScalarTy) and inst.dest.ty.is_float
                lv = _const_val(lhs)
                rv = _const_val(rhs)

                # Full constant fold (both operands are Const)
                result = _fold_bin(inst.op, lv, rv, is_float)
                if result is not None:
                    # DON'T put in replacements — the dest Value might be
                    # read in another block. Just emit a materialization.
                    # Actually: we CAN fold if we emit the constant inline
                    # wherever dest is used. But that's complex. Instead:
                    # emit a simpler instruction (mov-like: add dest, const, 0)
                    # with the folded value.
                    if is_float:
                        inst.lhs = Const(inst.dest.ty, result)
                        inst.rhs = Const(inst.dest.ty, 0.0)
                        inst.op = BinOp.ADD
                    else:
                        inst.lhs = Const(inst.dest.ty, int(result))
                        inst.rhs = Const(inst.dest.ty, 0)
                        inst.op = BinOp.ADD
                    folded += 1
                    # Keep the instruction (it's now simpler but still writes dest)

                # Strength reduction: mul by power of 2 → shift left (integers only)
                elif not is_float and inst.op == BinOp.MUL and rv is not None and isinstance(rv, int) and rv > 0 and (rv & (rv-1)) == 0:
                    shift = rv.bit_length() - 1
                    inst.op = BinOp.SHL
                    inst.rhs = Const(inst.dest.ty, shift)
                    folded += 1
                elif not is_float and inst.op == BinOp.MUL and lv is not None and isinstance(lv, int) and lv > 0 and (lv & (lv-1)) == 0:
                    shift = lv.bit_length() - 1
                    inst.op = BinOp.SHL
                    inst.lhs = inst.rhs
                    inst.rhs = Const(inst.dest.ty, shift)
                    folded += 1

                # Safe identity fold: x * 0 → replace instruction with "add dest, 0, 0"
                elif inst.op == BinOp.MUL and (rv == 0 or lv == 0):
                    inst.lhs = Const(inst.dest.ty, 0)
                    inst.rhs = Const(inst.dest.ty, 0)
                    inst.op = BinOp.ADD
                    folded += 1

            elif isinstance(inst, CmpInst):
                inst.lhs = _resolve(inst.lhs)
                inst.rhs = _resolve(inst.rhs)
            elif isinstance(inst, LoadInst):
                inst.addr = _resolve(inst.addr)
            elif isinstance(inst, StoreInst):
                inst.addr = _resolve(inst.addr)
                inst.value = _resolve(inst.value)

            new_insts.append(inst)
        bb.instructions = new_insts

    return folded


def cse(kernel: Kernel) -> int:
    """
    Common Subexpression Elimination (local, per basic block).

    If two BinInst in the same block have identical (op, lhs, rhs),
    the second reuses the first result.

    SAFETY: per-block only. Never eliminates an instruction whose
    dest was already written in this block (loop writeback pattern).
    """
    eliminated = 0

    for bb in kernel.blocks:
        seen: dict[tuple, Value] = {}
        replacements: dict[int, Value] = {}
        written_ids: set[int] = set()
        new_insts = []

        def _key(op: Operand):
            if isinstance(op, Value):
                v = op
                while v.id in replacements:
                    v = replacements[v.id]
                return ('val', v.id)
            if isinstance(op, Const):
                return ('const', op.value)
            return ('other', id(op))

        for inst in bb.instructions:
            if isinstance(inst, BinInst):
                if isinstance(inst.lhs, Value) and inst.lhs.id in replacements:
                    inst.lhs = replacements[inst.lhs.id]
                if isinstance(inst.rhs, Value) and inst.rhs.id in replacements:
                    inst.rhs = replacements[inst.rhs.id]

                # Don't CSE if dest was already written (loop writeback)
                if inst.dest.id in written_ids:
                    new_insts.append(inst)
                    continue

                # Include dest TYPE in key — prevents merging int and float
                # variables that happen to have the same init value
                dest_type_key = str(inst.dest.ty)
                key = (inst.op, _key(inst.lhs), _key(inst.rhs), dest_type_key)
                if key in seen:
                    replacements[inst.dest.id] = seen[key]
                    eliminated += 1
                    continue

                seen[key] = inst.dest
                written_ids.add(inst.dest.id)

            else:
                # Apply replacements to other instruction types
                if isinstance(inst, CmpInst):
                    if isinstance(inst.lhs, Value) and inst.lhs.id in replacements:
                        inst.lhs = replacements[inst.lhs.id]
                    if isinstance(inst.rhs, Value) and inst.rhs.id in replacements:
                        inst.rhs = replacements[inst.rhs.id]
                elif isinstance(inst, LoadInst):
                    if isinstance(inst.addr, Value) and inst.addr.id in replacements:
                        inst.addr = replacements[inst.addr.id]
                elif isinstance(inst, StoreInst):
                    if isinstance(inst.addr, Value) and inst.addr.id in replacements:
                        inst.addr = replacements[inst.addr.id]
                    if isinstance(inst.value, Value) and inst.value.id in replacements:
                        inst.value = replacements[inst.value.id]

            new_insts.append(inst)
        bb.instructions = new_insts

    return eliminated


def optimize(module: Module, verbose: bool = False) -> Module:
    """Run all optimization passes on the module."""
    from .unroll import unroll_loops

    for kernel in module.kernels:
        # Loop unrolling disabled — needs loop-carried variable chaining.
        # The unroller creates new Values per iteration but doesn't connect
        # the accumulator output of iteration N to the input of iteration N+1.
        n_unroll = 0  # unroll_loops(kernel, max_unroll=16)
        n_fold = constant_fold(kernel)
        n_cse = cse(kernel)
        if verbose:
            total = n_unroll + n_fold + n_cse
            if total > 0:
                parts = []
                if n_unroll: parts.append(f"{n_unroll} loops unrolled")
                if n_fold: parts.append(f"{n_fold} constants folded")
                if n_cse: parts.append(f"{n_cse} CSE eliminated")
                print(f"[opt] {kernel.name}: {', '.join(parts)}")
    return module
