"""
Loop unrolling pass for OpenCUDA IR.

Detects for-loops with compile-time-known trip counts and unrolls them.
Handles loop-carried variables (accumulators) by chaining each iteration's
output to the next iteration's input.

The key insight: in `for (int k=0; k<16; k++) { sum += f(k); }`, the
variable `sum` is "loop-carried" — each iteration reads the previous
iteration's output. The unroller must connect these across copies.
"""

from __future__ import annotations
from ..ir.nodes import (Kernel, BasicBlock, Value, Const, Operand,
                         BinInst, CmpInst, LoadInst, StoreInst,
                         CallInst, ParamInst, CvtInst,
                         BinOp, CmpOp,
                         RetTerm, BrTerm, CondBrTerm)


def _find_unrollable_loops(kernel: Kernel) -> list[dict]:
    """Find for-loops with constant bounds that can be unrolled."""
    loops = []
    blocks_by_label = {bb.label: bb for bb in kernel.blocks}

    for bb in kernel.blocks:
        if not isinstance(getattr(bb, 'terminator', None), CondBrTerm):
            continue

        cond_bb = bb
        cmp_inst = None
        for inst in cond_bb.instructions:
            if isinstance(inst, CmpInst):
                cmp_inst = inst

        if cmp_inst is None:
            continue

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
        if bound <= 0 or bound > 64:
            continue

        true_label = cond_bb.terminator.true_bb
        false_label = cond_bb.terminator.false_bb
        if true_label not in blocks_by_label or false_label not in blocks_by_label:
            continue

        body_bb = blocks_by_label[true_label]
        exit_bb = blocks_by_label[false_label]

        if not isinstance(getattr(body_bb, 'terminator', None), BrTerm):
            continue
        inc_label = body_bb.terminator.target
        if inc_label not in blocks_by_label:
            continue
        inc_bb = blocks_by_label[inc_label]

        if not isinstance(getattr(inc_bb, 'terminator', None), BrTerm):
            continue
        if inc_bb.terminator.target != cond_bb.label:
            continue

        # Find loop-carried variables from the increment block.
        # Pattern: add CANONICAL, NEW_VAL, 0  (writeback copy)
        # The CANONICAL is the loop-carried var, NEW_VAL is the body's output.
        carried_vars = {}  # canonical_id → (canonical_Value, new_Value)
        for inst in inc_bb.instructions:
            if isinstance(inst, BinInst) and inst.op == BinOp.ADD:
                if isinstance(inst.rhs, Const) and inst.rhs.value == 0:
                    # add CANONICAL, NEW_VAL, 0 → writeback
                    if isinstance(inst.lhs, Value):
                        carried_vars[inst.dest.id] = (inst.dest, inst.lhs)
                elif isinstance(inst.lhs, Const) and inst.lhs.value == 0:
                    if isinstance(inst.rhs, Value):
                        carried_vars[inst.dest.id] = (inst.dest, inst.rhs)

        loops.append({
            'cond_bb': cond_bb,
            'body_bb': body_bb,
            'inc_bb': inc_bb,
            'exit_bb': exit_bb,
            'bound': bound,
            'induction_var': induction_var,
            'carried_vars': carried_vars,
        })

    return loops


def unroll_loops(kernel: Kernel, max_unroll: int = 16) -> int:
    """Unroll eligible for-loops with loop-carried variable chaining."""
    loops = _find_unrollable_loops(kernel)
    unrolled_count = 0

    for loop in loops:
        bound = loop['bound']
        if bound > max_unroll:
            continue

        cond_bb = loop['cond_bb']
        body_bb = loop['body_bb']
        inc_bb = loop['inc_bb']
        exit_bb = loop['exit_bb']
        induction_var = loop['induction_var']
        carried_vars = loop['carried_vars']

        # Build the value mapping for each iteration.
        # Start: induction_var → Const(0), carried vars → their canonical Values
        # Each iteration: create new Values, chain carried vars from prev output.

        all_unrolled_insts = []

        # Persistent remap across iterations for loop-carried variables
        carried_remap = {}  # canonical_id → current Value (chains across iterations)

        for iteration in range(bound):
            # Build replacement map: start from carried state + induction var
            remap = dict(carried_remap)  # inherit carried var chain
            remap[induction_var.id] = Const(induction_var.ty, iteration)

            # Carried variables: for iteration 0, use the canonical (entry) value.
            # For iteration N>0, use the output from iteration N-1.
            # (The previous iteration's "new_val" becomes this iteration's input.)
            # We'll update carried_var mapping after processing each iteration.

            # Copy body instructions with remapping
            iter_new_vals = {}  # Maps body dest id → new Value for this iteration

            for inst in body_bb.instructions:
                new_inst = _remap_inst(inst, remap, kernel, iteration)
                if new_inst is not None:
                    all_unrolled_insts.append(new_inst)
                    # Track the new dest for carried variable chaining
                    if hasattr(new_inst, 'dest') and hasattr(inst, 'dest'):
                        iter_new_vals[inst.dest.id] = new_inst.dest

            # After processing body: update carried variable mapping for NEXT iteration.
            for canonical_id, (canonical_val, new_val) in carried_vars.items():
                if new_val.id in iter_new_vals:
                    carried_remap[canonical_id] = iter_new_vals[new_val.id]

        # After all iterations: write back carried variables to canonical registers
        for canonical_id, (canonical_val, new_val) in carried_vars.items():
            if canonical_id in carried_remap and isinstance(carried_remap[canonical_id], Value):
                final_val = carried_remap[canonical_id]
                # Emit: canonical = final_val + 0 (writeback copy)
                zero = Const(canonical_val.ty, 0.0 if hasattr(canonical_val.ty, 'is_float') and canonical_val.ty.is_float else 0)
                all_unrolled_insts.append(
                    BinInst(canonical_val, BinOp.ADD, final_val, zero))

        # Replace cond block with unrolled instructions → branch to exit
        cond_bb.instructions = all_unrolled_insts
        cond_bb.terminator = BrTerm(exit_bb.label)

        # Remove dead body and inc blocks
        kernel.blocks = [bb for bb in kernel.blocks
                         if bb.label not in (body_bb.label, inc_bb.label)]

        unrolled_count += 1

    return unrolled_count


def _remap_inst(inst, remap: dict, kernel: Kernel, iteration: int):
    """Copy an instruction, replacing Values according to remap."""

    def _r(op: Operand) -> Operand:
        if isinstance(op, Value) and op.id in remap:
            return remap[op.id]
        return op

    if isinstance(inst, BinInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        new_inst = BinInst(new_dest, inst.op, _r(inst.lhs), _r(inst.rhs))
        # Update remap so later instructions in this iteration see the new dest
        remap[inst.dest.id] = new_dest
        return new_inst

    elif isinstance(inst, LoadInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        new_inst = LoadInst(new_dest, _r(inst.addr))
        remap[inst.dest.id] = new_dest
        return new_inst

    elif isinstance(inst, StoreInst):
        return StoreInst(_r(inst.addr), _r(inst.value))

    elif isinstance(inst, CmpInst):
        new_dest = kernel.new_value(f"{inst.dest.name}_u{iteration}", inst.dest.ty)
        new_inst = CmpInst(new_dest, inst.op, _r(inst.lhs), _r(inst.rhs))
        remap[inst.dest.id] = new_dest
        return new_inst

    elif isinstance(inst, CallInst):
        if inst.func == '__syncthreads':
            return inst  # keep barriers
        return inst

    return inst
