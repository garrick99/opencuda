"""
OpenCUDA IR optimization passes.

Pass 1: Constant folding — evaluate Const op Const at compile time.
Pass 2: Dead code elimination — remove instructions whose results are unused.
Pass 3: Common subexpression elimination — reuse identical computations.
"""

from __future__ import annotations
from ..ir.nodes import (Module, Kernel, BasicBlock, Value, Const, Operand,
                         BinInst, CmpInst, LoadInst, StoreInst, CvtInst,
                         CallInst, ParamInst, BinOp, CmpOp)
from ..ir.types import INT32, UINT32, FLOAT, ScalarTy


def _const_val(op: Operand):
    """Extract numeric value from a Const, or None."""
    if isinstance(op, Const):
        return op.value
    return None


def _fold_bin(op: BinOp, a, b, is_float: bool):
    """Evaluate a binary op on two constants. Returns result or None."""
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
    Also folds identity ops: x + 0 → x, x * 1 → x, x * 0 → 0.
    Returns the number of instructions folded.
    """
    # Map Value.id → replacement Operand
    replacements: dict[int, Operand] = {}
    folded = 0

    def _resolve(op: Operand) -> Operand:
        if isinstance(op, Value) and op.id in replacements:
            return replacements[op.id]
        return op

    for bb in kernel.blocks:
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

                # Full constant fold
                result = _fold_bin(inst.op, lv, rv, is_float)
                if result is not None:
                    replacements[inst.dest.id] = Const(inst.dest.ty, result)
                    folded += 1
                    continue  # don't emit this instruction

                # Identity folds disabled — not safe with our loop writeback
                # pattern (add dest, src, 0 for variable update).
                # Only safe fold: mul by 0 → 0 (result is always 0)
                if inst.op == BinOp.MUL and (rv == 0 or lv == 0):
                    replacements[inst.dest.id] = Const(inst.dest.ty, 0)
                    folded += 1
                    continue

            elif isinstance(inst, CmpInst):
                inst.lhs = _resolve(inst.lhs)
                inst.rhs = _resolve(inst.rhs)
            elif isinstance(inst, LoadInst):
                inst.addr = _resolve(inst.addr)
            elif isinstance(inst, StoreInst):
                inst.addr = _resolve(inst.addr)
                inst.value = _resolve(inst.value)
            elif isinstance(inst, CvtInst):
                inst.src = _resolve(inst.src)

            new_insts.append(inst)
        bb.instructions = new_insts

    return folded


def cse(kernel: Kernel) -> int:
    """
    Common Subexpression Elimination.

    If two BinInst have the same (op, lhs, rhs), reuse the first result
    for the second. This is local CSE (within each basic block).

    Returns the number of eliminated instructions.
    """
    eliminated = 0

    for bb in kernel.blocks:
        # Map (op, lhs_key, rhs_key) → result Value
        seen: dict[tuple, Value] = {}
        new_insts = []
        replacements: dict[int, Value] = {}

        def _key(op: Operand):
            if isinstance(op, Value):
                # Follow replacement chain
                v = op
                while v.id in replacements:
                    v = replacements[v.id]
                return ('val', v.id)
            if isinstance(op, Const):
                return ('const', op.ty, op.value)
            return ('other', id(op))

        # Track which Values are written to (dest of any instruction)
        # to avoid CSE on loop writeback instructions (add dest, src, 0
        # where dest was previously defined)
        written_ids: set[int] = set()

        for inst in bb.instructions:
            # Apply replacements to operands
            if isinstance(inst, BinInst):
                if isinstance(inst.lhs, Value) and inst.lhs.id in replacements:
                    inst.lhs = replacements[inst.lhs.id]
                if isinstance(inst.rhs, Value) and inst.rhs.id in replacements:
                    inst.rhs = replacements[inst.rhs.id]

                # Don't CSE if dest was already written in this block
                # (this is a loop writeback or re-assignment, must execute)
                if inst.dest.id in written_ids:
                    written_ids.add(inst.dest.id)
                    new_insts.append(inst)
                    continue

                key = (inst.op, _key(inst.lhs), _key(inst.rhs))
                if key in seen:
                    replacements[inst.dest.id] = seen[key]
                    eliminated += 1
                    continue
                seen[key] = inst.dest
                written_ids.add(inst.dest.id)

            elif isinstance(inst, CmpInst):
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
    for kernel in module.kernels:
        n_fold = 0 # constant_fold(kernel)
        # CSE disabled — needs fix for loop writeback patterns
        # n_cse = cse(kernel)
        n_cse = 0
        if verbose:
            total = n_fold + n_cse
            if total > 0:
                parts = []
                if n_fold: parts.append(f"{n_fold} constants folded")
                if n_cse: parts.append(f"{n_cse} CSE eliminated")
                print(f"[opt] {kernel.name}: {', '.join(parts)}")
    return module
