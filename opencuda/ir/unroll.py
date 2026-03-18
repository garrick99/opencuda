"""
Loop unrolling pass for OpenCUDA IR.

Detects for-loops with compile-time-known trip counts and unrolls them
by duplicating the loop body N times, eliminating branch overhead.

Target: inner loops like `for (int k = 0; k < 16; k++)` in tiled matmul.
These have constant bounds and simple induction variables.
"""

from __future__ import annotations
from copy import deepcopy
from ..ir.nodes import (Kernel, BasicBlock, Value, Const, Operand,
                         BinInst, CmpInst, LoadInst, StoreInst,
                         CallInst, ParamInst, CvtInst,
                         BinOp, CmpOp,
                         RetTerm, BrTerm, CondBrTerm)
from ..ir.types import Type


def _find_unrollable_loops(kernel: Kernel) -> list[dict]:
    """
    Find for-loops that can be unrolled.

    A loop is unrollable if:
    1. It has a condition block that compares an induction var to a constant
    2. The increment block adds a constant to the induction var
    3. The trip count is known at compile time and <= 64
    """
    loops = []
    blocks_by_label = {bb.label: bb for bb in kernel.blocks}

    for bb in kernel.blocks:
        if bb.terminator is None:
            continue
        if not isinstance(bb.terminator, CondBrTerm):
            continue

        cond_bb = bb
        # Check if this is a loop condition: compares a value to a constant
        cmp_inst = None
        for inst in cond_bb.instructions:
            if isinstance(inst, CmpInst):
                cmp_inst = inst
                break

        if cmp_inst is None:
            continue

        # Check for constant bound
        bound = None
        induction_var = None
        if isinstance(cmp_inst.rhs, Const):
            bound = int(cmp_inst.rhs.value)
            induction_var = cmp_inst.lhs
        elif isinstance(cmp_inst.lhs, Const):
            bound = int(cmp_inst.lhs.value)
            induction_var = cmp_inst.rhs

        if bound is None or not isinstance(induction_var, Value):
            continue
        if bound > 64 or bound <= 0:
            continue

        # Find the body and increment blocks
        true_label = cond_bb.terminator.true_bb
        false_label = cond_bb.terminator.false_bb

        if true_label not in blocks_by_label or false_label not in blocks_by_label:
            continue

        body_bb = blocks_by_label[true_label]
        exit_bb = blocks_by_label[false_label]

        # Find increment block (body's terminator should branch to it)
        if body_bb.terminator is None or not isinstance(body_bb.terminator, BrTerm):
            continue

        inc_label = body_bb.terminator.target
        if inc_label not in blocks_by_label:
            continue
        inc_bb = blocks_by_label[inc_label]

        # Verify increment branches back to condition
        if inc_bb.terminator is None or not isinstance(inc_bb.terminator, BrTerm):
            continue
        if inc_bb.terminator.target != cond_bb.label:
            continue

        # Check that induction variable starts at 0 (or known constant)
        # and increments by 1
        # For now, just check the increment block has an add by 1
        has_inc = False
        for inst in inc_bb.instructions:
            if isinstance(inst, BinInst) and inst.op == BinOp.ADD:
                if isinstance(inst.rhs, Const) and inst.rhs.value == 1:
                    has_inc = True
                elif isinstance(inst.lhs, Const) and inst.lhs.value == 1:
                    has_inc = True

        if not has_inc:
            continue

        loops.append({
            'cond_bb': cond_bb,
            'body_bb': body_bb,
            'inc_bb': inc_bb,
            'exit_bb': exit_bb,
            'bound': bound,
            'induction_var': induction_var,
            'cmp_op': cmp_inst.op,
        })

    return loops


def unroll_loops(kernel: Kernel, max_unroll: int = 16) -> int:
    """
    Unroll eligible for-loops in the kernel.

    Returns the number of loops unrolled.
    """
    loops = _find_unrollable_loops(kernel)
    unrolled = 0

    for loop in loops:
        bound = loop['bound']
        if bound > max_unroll:
            continue

        cond_bb = loop['cond_bb']
        body_bb = loop['body_bb']
        inc_bb = loop['inc_bb']
        exit_bb = loop['exit_bb']
        induction_var = loop['induction_var']

        # Replace the condition block's branch with a direct branch to body
        # and duplicate body+inc `bound` times, then branch to exit.

        # Find the block that branches TO the condition block (the entry)
        entry_to_cond = None
        for bb in kernel.blocks:
            if bb.terminator and isinstance(bb.terminator, BrTerm):
                if bb.terminator.target == cond_bb.label:
                    if bb != inc_bb:  # not the loop-back
                        entry_to_cond = bb
                        break

        if entry_to_cond is None:
            continue

        # Build unrolled body: copy body + inc instructions `bound` times
        # Replace induction variable references with the iteration constant
        unrolled_insts = []
        for iteration in range(bound):
            # For each instruction in body, copy and resolve induction var
            for inst in body_bb.instructions:
                new_inst = _copy_inst_with_const_induction(inst, induction_var,
                                                           iteration, kernel)
                if new_inst is not None:
                    unrolled_insts.append(new_inst)

            # Copy increment instructions (except the induction var update itself)
            for inst in inc_bb.instructions:
                # Skip the induction variable increment (it's now constant)
                if isinstance(inst, BinInst) and inst.dest.id == induction_var.id:
                    continue
                # Skip writeback copies for the induction var
                if isinstance(inst, BinInst) and isinstance(inst.lhs, Value):
                    if inst.lhs.id == induction_var.id:
                        continue
                new_inst = _copy_inst_with_const_induction(inst, induction_var,
                                                           iteration, kernel)
                if new_inst is not None:
                    unrolled_insts.append(new_inst)

        # Replace the condition block contents with the unrolled body
        # and branch directly to exit
        cond_bb.instructions = unrolled_insts
        cond_bb.terminator = BrTerm(exit_bb.label)

        # Remove the old body, inc blocks (they're now dead)
        kernel.blocks = [bb for bb in kernel.blocks
                         if bb.label not in (body_bb.label, inc_bb.label)]

        # Fix the entry block to branch directly to the condition (which now has the unrolled code)
        # (it already does this, so no change needed)

        unrolled += 1

    return unrolled


def _copy_inst_with_const_induction(inst, induction_var: Value,
                                     iteration: int, kernel: Kernel):
    """
    Copy an instruction, replacing references to the induction variable
    with a Const for the current iteration.
    """
    def _replace(op: Operand) -> Operand:
        if isinstance(op, Value) and op.id == induction_var.id:
            return Const(induction_var.ty, iteration)
        return op

    if isinstance(inst, BinInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        return BinInst(new_dest, inst.op, _replace(inst.lhs), _replace(inst.rhs))
    elif isinstance(inst, LoadInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        return LoadInst(new_dest, _replace(inst.addr))
    elif isinstance(inst, StoreInst):
        return StoreInst(_replace(inst.addr), _replace(inst.value))
    elif isinstance(inst, CmpInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        return CmpInst(new_dest, inst.op, _replace(inst.lhs), _replace(inst.rhs))
    elif isinstance(inst, CallInst):
        return inst  # don't modify calls
    elif isinstance(inst, ParamInst):
        return inst

    return inst
