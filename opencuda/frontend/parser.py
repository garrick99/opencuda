"""
OpenCUDA parser — CUDA-subset C to IR.

Supported:
  - __global__ kernel functions
  - Types: int, unsigned int, float, void, pointers
  - Expressions: arithmetic, comparison, array indexing, member access
  - Statements: variable decl, assignment, if/else, for, return
  - Built-ins: threadIdx.x/y/z, blockIdx.x/y/z, blockDim.x/y/z, __syncthreads()
"""

from __future__ import annotations
from typing import Optional

from .lexer import Token, TokKind, lex
from ..ir.types import (Type, ScalarTy, PtrTy, AddrSpace, ScalarType, StructTy,
                         INT32, UINT32, FLOAT, VOID, INT64, UINT64, DOUBLE)
from ..ir.nodes import (Module, Kernel, KernelParam, BasicBlock,
                         Value, Const, Operand,
                         BinInst, CmpInst, LoadInst, StoreInst,
                         CvtInst, CallInst, ParamInst,
                         BinOp, CmpOp,
                         RetTerm, BrTerm, CondBrTerm)


class ParseError(Exception):
    pass


class Parser:
    def __init__(self, tokens: list[Token]):
        self._toks = tokens
        self._pos = 0
        self._kernel: Optional[Kernel] = None
        self._cur_block: Optional[BasicBlock] = None
        self._variables: dict[str, Value] = {}
        self._block_count = 0
        self._struct_types: dict[str, StructTy] = {}  # struct name → StructTy
        self._typedefs: dict[str, Type] = {}  # typedef name → Type
        self._break_targets: list[str] = []     # stack of break target labels
        self._continue_targets: list[str] = []  # stack of continue target labels

    # -- Token helpers -------------------------------------------------------

    def _peek(self) -> Token:
        return self._toks[self._pos]

    def _advance(self) -> Token:
        tok = self._toks[self._pos]
        self._pos += 1
        return tok

    def _expect(self, kind: TokKind) -> Token:
        tok = self._peek()
        if tok.kind != kind:
            raise ParseError(f"Line {tok.line}: expected {kind.name}, got {tok.kind.name} '{tok.value}'")
        return self._advance()

    def _match(self, kind: TokKind) -> Optional[Token]:
        if self._peek().kind == kind:
            return self._advance()
        return None

    def _at(self, kind: TokKind) -> bool:
        return self._peek().kind == kind

    # -- IR helpers ----------------------------------------------------------

    def _new_block(self, prefix: str = "BB") -> BasicBlock:
        self._block_count += 1
        label = f"{prefix}_{self._block_count}"
        bb = BasicBlock(label=label)
        self._kernel.blocks.append(bb)
        return bb

    def _emit(self, inst):
        self._cur_block.instructions.append(inst)

    def _new_val(self, name: str, ty: Type) -> Value:
        return self._kernel.new_value(name, ty)

    # -- Type parsing --------------------------------------------------------

    def _parse_type(self) -> Type:
        """Parse a C type specifier."""
        tok = self._peek()

        if tok.kind == TokKind.KW_VOID:
            self._advance()
            return VOID
        elif tok.kind == TokKind.KW_FLOAT:
            self._advance()
            return FLOAT
        elif tok.kind == TokKind.KW_DOUBLE:
            self._advance()
            return DOUBLE
        elif tok.kind == TokKind.KW_INT:
            self._advance()
            return INT32
        elif tok.kind == TokKind.KW_UNSIGNED:
            self._advance()
            if self._match(TokKind.KW_INT):
                pass
            elif self._match(TokKind.KW_LONG):
                if self._match(TokKind.KW_LONG):
                    return UINT64
            return UINT32
        elif tok.kind == TokKind.KW_LONG:
            self._advance()
            if self._match(TokKind.KW_LONG):
                return INT64
            return INT32  # treat 'long' as int32 for simplicity

        # Struct type
        if tok.kind == TokKind.KW_STRUCT:
            self._advance()
            sname = self._expect(TokKind.IDENT).value
            if sname in self._struct_types:
                return self._struct_types[sname]
            raise ParseError(f"Line {tok.line}: undefined struct '{sname}'")

        # Typedef'd type
        if tok.kind == TokKind.IDENT and tok.value in self._typedefs:
            self._advance()
            return self._typedefs[tok.value]

        raise ParseError(f"Line {tok.line}: expected type, got '{tok.value}'")

    def _parse_type_with_ptr(self) -> Type:
        """Parse type followed by optional pointer stars."""
        base = self._parse_type()
        while self._match(TokKind.STAR):
            base = PtrTy(base, AddrSpace.GLOBAL)
        return base

    # -- Expression parsing (precedence climbing) ----------------------------

    def _parse_expr(self) -> Operand:
        return self._parse_assign_expr()

    def _parse_assign_expr(self) -> Operand:
        lhs = self._parse_or_expr()
        # Ternary: cond ? true_expr : false_expr
        if self._match(TokKind.QUESTION):
            true_val = self._parse_expr()
            self._expect(TokKind.COLON)
            false_val = self._parse_assign_expr()
            # Lower to: if (cond) { dest = true } else { dest = false }
            result_ty = true_val.ty if isinstance(true_val, Value) else (
                false_val.ty if isinstance(false_val, Value) else INT32)
            dest = self._new_val("ternary", result_ty)
            true_bb = self._new_block("tern_true")
            false_bb = self._new_block("tern_false")
            merge_bb = self._new_block("tern_merge")
            self._cur_block.terminator = CondBrTerm(lhs, true_bb.label, false_bb.label)
            self._cur_block = true_bb
            # Emit: dest = true_val
            self._emit(BinInst(dest, BinOp.ADD, true_val,
                               Const(result_ty, 0.0 if (isinstance(result_ty, ScalarTy) and result_ty.is_float) else 0)))
            true_bb.terminator = BrTerm(merge_bb.label)
            self._cur_block = false_bb
            self._emit(BinInst(dest, BinOp.ADD, false_val,
                               Const(result_ty, 0.0 if (isinstance(result_ty, ScalarTy) and result_ty.is_float) else 0)))
            false_bb.terminator = BrTerm(merge_bb.label)
            self._cur_block = merge_bb
            return dest
        if self._match(TokKind.ASSIGN):
            rhs = self._parse_assign_expr()
            if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                self._emit(StoreInst(addr=lhs, value=rhs))
                return rhs
            # Variable assignment: update the variable binding
            if isinstance(lhs, Value) and lhs.name in self._variables:
                self._variables[lhs.name] = rhs
            return rhs
        # Compound assignment: +=, -=, *=
        for tok_kind, op in [(TokKind.PLUS_EQ, BinOp.ADD),
                             (TokKind.MINUS_EQ, BinOp.SUB),
                             (TokKind.STAR_EQ, BinOp.MUL)]:
            if self._match(tok_kind):
                rhs = self._parse_assign_expr()
                if isinstance(lhs, Value):
                    new_val = self._new_val(f"{lhs.name}_compound", lhs.ty)
                    self._emit(BinInst(new_val, op, lhs, rhs))
                    if lhs.name in self._variables:
                        self._variables[lhs.name] = new_val
                    return new_val
        return lhs

    def _parse_or_expr(self) -> Operand:
        lhs = self._parse_and_expr()
        while self._match(TokKind.OR):
            rhs = self._parse_and_expr()
            dest = self._new_val("or", INT32)
            self._emit(BinInst(dest, BinOp.OR, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_and_expr(self) -> Operand:
        lhs = self._parse_bitor_expr()
        while self._match(TokKind.AND):
            rhs = self._parse_bitor_expr()
            dest = self._new_val("and", INT32)
            self._emit(BinInst(dest, BinOp.AND, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_bitor_expr(self) -> Operand:
        lhs = self._parse_bitxor_expr()
        while self._match(TokKind.PIPE):
            rhs = self._parse_bitxor_expr()
            dest = self._new_val("bitor", self._result_type(lhs, rhs))
            self._emit(BinInst(dest, BinOp.OR, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_bitxor_expr(self) -> Operand:
        lhs = self._parse_bitand_expr()
        while self._match(TokKind.CARET):
            rhs = self._parse_bitand_expr()
            dest = self._new_val("bitxor", self._result_type(lhs, rhs))
            self._emit(BinInst(dest, BinOp.XOR, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_bitand_expr(self) -> Operand:
        lhs = self._parse_cmp_expr()
        while self._match(TokKind.AMP):
            rhs = self._parse_cmp_expr()
            dest = self._new_val("bitand", self._result_type(lhs, rhs))
            self._emit(BinInst(dest, BinOp.AND, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_cmp_expr(self) -> Operand:
        lhs = self._parse_shift_expr()
        cmp_ops = {
            TokKind.EQ: CmpOp.EQ, TokKind.NE: CmpOp.NE,
            TokKind.LT: CmpOp.LT, TokKind.LE: CmpOp.LE,
            TokKind.GT: CmpOp.GT, TokKind.GE: CmpOp.GE,
        }
        for tok_kind, cmp_op in cmp_ops.items():
            if self._match(tok_kind):
                rhs = self._parse_add_expr()
                dest = self._new_val("cmp", INT32)
                self._emit(CmpInst(dest, cmp_op, lhs, rhs))
                return dest
        return lhs

    def _result_type(self, a: Operand, b: Operand) -> Type:
        """Determine result type with promotion (float wins over int, wider wins)."""
        a_ty = a.ty if isinstance(a, Value) else (FLOAT if isinstance(a, Const) and isinstance(a.value, float) else INT32)
        b_ty = b.ty if isinstance(b, Value) else (FLOAT if isinstance(b, Const) and isinstance(b.value, float) else INT32)
        # Float promotion
        if (isinstance(a_ty, ScalarTy) and a_ty.is_float) or (isinstance(b_ty, ScalarTy) and b_ty.is_float):
            return FLOAT
        # Pointer arithmetic
        if isinstance(a_ty, PtrTy):
            return a_ty
        if isinstance(b_ty, PtrTy):
            return b_ty
        # 64-bit promotion
        if (isinstance(a_ty, ScalarTy) and a_ty.size == 8) or (isinstance(b_ty, ScalarTy) and b_ty.size == 8):
            return a_ty if isinstance(a_ty, ScalarTy) and a_ty.size == 8 else b_ty
        return a_ty

    def _parse_shift_expr(self) -> Operand:
        lhs = self._parse_add_expr()
        while True:
            if self._match(TokKind.LSHIFT):
                rhs = self._parse_add_expr()
                dest = self._new_val("shl", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.SHL, lhs, rhs))
                lhs = dest
            elif self._match(TokKind.RSHIFT):
                rhs = self._parse_add_expr()
                dest = self._new_val("shr", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.SHR, lhs, rhs))
                lhs = dest
            else:
                break
        return lhs

    def _parse_add_expr(self) -> Operand:
        lhs = self._parse_mul_expr()
        while True:
            if self._match(TokKind.PLUS):
                rhs = self._parse_mul_expr()
                dest = self._new_val("add", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.ADD, lhs, rhs))
                lhs = dest
            elif self._match(TokKind.MINUS):
                rhs = self._parse_mul_expr()
                dest = self._new_val("sub", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.SUB, lhs, rhs))
                lhs = dest
            else:
                break
        return lhs

    def _parse_mul_expr(self) -> Operand:
        lhs = self._parse_unary_expr()
        while True:
            if self._match(TokKind.STAR):
                rhs = self._parse_unary_expr()
                dest = self._new_val("mul", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.MUL, lhs, rhs))
                lhs = dest
            elif self._match(TokKind.SLASH):
                rhs = self._parse_unary_expr()
                dest = self._new_val("div", self._result_type(lhs, rhs))
                self._emit(BinInst(dest, BinOp.DIV, lhs, rhs))
                lhs = dest
            else:
                break
        return lhs

    def _parse_unary_expr(self) -> Operand:
        if self._match(TokKind.STAR):
            operand = self._parse_unary_expr()
            if isinstance(operand, Value) and isinstance(operand.ty, PtrTy):
                dest = self._new_val("deref", operand.ty.pointee)
                self._emit(LoadInst(dest, operand))
                return dest
        if self._match(TokKind.MINUS):
            operand = self._parse_unary_expr()
            ty = operand.ty if isinstance(operand, Value) else INT32
            dest = self._new_val("neg", ty)
            zero = Const(FLOAT, 0.0) if (isinstance(ty, ScalarTy) and ty.is_float) else Const(ty, 0)
            self._emit(BinInst(dest, BinOp.SUB, zero, operand))
            return dest
        if self._match(TokKind.TILDE):
            operand = self._parse_unary_expr()
            dest = self._new_val("bnot", operand.ty if isinstance(operand, Value) else INT32)
            self._emit(BinInst(dest, BinOp.XOR, operand, Const(INT32, -1)))
            return dest
        if self._match(TokKind.BANG):
            operand = self._parse_unary_expr()
            dest = self._new_val("lnot", INT32)
            self._emit(CmpInst(dest, CmpOp.EQ, operand, Const(INT32, 0)))
            return dest
        # Cast: (type)expr
        if self._at(TokKind.LPAREN):
            saved = self._pos
            self._advance()
            try:
                cast_ty = self._parse_type()
                # Check for pointer
                while self._match(TokKind.STAR):
                    cast_ty = PtrTy(cast_ty, AddrSpace.GLOBAL)
                self._expect(TokKind.RPAREN)
                operand = self._parse_unary_expr()
                dest = self._new_val("cast", cast_ty)
                self._emit(CvtInst(dest, operand))
                return dest
            except ParseError:
                self._pos = saved
        return self._parse_postfix_expr()

    def _parse_postfix_expr(self) -> Operand:
        lhs = self._parse_primary_expr()

        while True:
            # i++ / i--
            if self._match(TokKind.PLUSPLUS):
                if isinstance(lhs, Value):
                    old = lhs
                    new_val = self._new_val(f"{old.name}_inc", old.ty)
                    self._emit(BinInst(new_val, BinOp.ADD, old, Const(old.ty, 1)))
                    self._variables[old.name] = new_val
                    lhs = old  # post-increment returns old value
                continue
            if self._match(TokKind.MINUSMINUS):
                if isinstance(lhs, Value):
                    old = lhs
                    new_val = self._new_val(f"{old.name}_dec", old.ty)
                    self._emit(BinInst(new_val, BinOp.SUB, old, Const(old.ty, 1)))
                    self._variables[old.name] = new_val
                    lhs = old
                continue

            if self._match(TokKind.LBRACKET):
                # Array indexing: ptr[index]
                index = self._parse_expr()
                self._expect(TokKind.RBRACKET)
                if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                    elem_size = lhs.ty.pointee.size
                    # addr = base + index * elem_size
                    if elem_size != 1:
                        scaled = self._new_val("scale", INT32)
                        self._emit(BinInst(scaled, BinOp.MUL, index, Const(index.ty if isinstance(index, Value) else INT32, elem_size)))
                        index = scaled
                    addr = self._new_val("addr", lhs.ty)
                    self._emit(BinInst(addr, BinOp.ADD, lhs, index))
                    # Load the value
                    dest = self._new_val("elem", lhs.ty.pointee)
                    self._emit(LoadInst(dest, addr))
                    lhs = dest
            elif self._match(TokKind.DOT):
                member = self._expect(TokKind.IDENT).value
                # Built-in: threadIdx.x, blockIdx.y, etc.
                if isinstance(lhs, Value) and lhs.name in ('threadIdx', 'blockIdx', 'blockDim'):
                    builtin = f"{lhs.name}.{member}"
                    dest = self._new_val(builtin.replace('.', '_'), INT32)
                    self._emit(CallInst(dest, builtin))
                    lhs = dest
                # Struct member access
                elif isinstance(lhs, Value) and isinstance(lhs.ty, StructTy):
                    sty = lhs.ty
                    field_off = sty.field_offset(member)
                    field_ty = sty.field_type(member)
                    # Compute address: &lhs + field_offset
                    # For now, emit as a load from a computed offset
                    # (this assumes lhs is a pointer to the struct)
                    dest = self._new_val(f"{lhs.name}_{member}", field_ty)
                    # TODO: proper struct field access via pointer arithmetic
                    lhs = dest
                elif isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy) and isinstance(lhs.ty.pointee, StructTy):
                    sty = lhs.ty.pointee
                    field_off = sty.field_offset(member)
                    field_ty = sty.field_type(member)
                    # ptr->field: compute address and load
                    offset_val = self._new_val("foff", INT32)
                    self._emit(BinInst(offset_val, BinOp.ADD, lhs, Const(INT32, field_off)))
                    addr = self._new_val("faddr", PtrTy(field_ty, lhs.ty.addr_space))
                    self._emit(BinInst(addr, BinOp.ADD, lhs, Const(INT32, field_off)))
                    dest = self._new_val(f"{member}", field_ty)
                    self._emit(LoadInst(dest, addr))
                    lhs = dest
            else:
                break
        return lhs

    def _parse_primary_expr(self) -> Operand:
        tok = self._peek()

        if tok.kind == TokKind.INT_LIT:
            self._advance()
            val = int(tok.value.rstrip('uUlL'), 0)
            return Const(INT32, val)

        if tok.kind == TokKind.FLOAT_LIT:
            self._advance()
            val = float(tok.value.rstrip('fF'))
            return Const(FLOAT, val)

        if tok.kind == TokKind.IDENT:
            name = tok.value
            self._advance()

            # Check for function call
            if self._match(TokKind.LPAREN):
                args = []
                if not self._at(TokKind.RPAREN):
                    args.append(self._parse_expr())
                    while self._match(TokKind.COMMA):
                        args.append(self._parse_expr())
                self._expect(TokKind.RPAREN)

                if name == '__syncthreads':
                    self._emit(CallInst(None, '__syncthreads', args))
                    return Const(VOID, 0)
                elif name in ('__shfl_sync', '__shfl_up_sync', '__shfl_down_sync',
                              '__shfl_xor_sync', '__ballot_sync'):
                    dest = self._new_val(name.replace('__',''), INT32)
                    self._emit(CallInst(dest, name, args))
                    return dest
                elif name in ('atomicAdd', 'atomicSub', 'atomicMin', 'atomicMax',
                              'atomicAnd', 'atomicOr', 'atomicXor', 'atomicExch',
                              'atomicCAS'):
                    # Atomic operations: atomicAdd(addr, val) → atom.global.add
                    if len(args) >= 2:
                        addr = args[0]
                        val = args[1]
                        result_ty = val.ty if isinstance(val, Value) else INT32
                        dest = self._new_val(f"atomic_{name}", result_ty)
                        self._emit(CallInst(dest, name, args))
                        return dest
                    return Const(INT32, 0)
                elif hasattr(self, '_device_funcs') and name in self._device_funcs:
                    # Inline __device__ function: bind args, replay body
                    dfunc = self._device_funcs[name]
                    saved_vars = dict(self._variables)
                    saved_pos = self._pos

                    # Bind arguments to parameters
                    for (pname, pty), arg in zip(dfunc['params'], args):
                        self._variables[pname] = arg

                    # Parse body — stop at 'return' and capture the expr
                    self._pos = dfunc['body_start']
                    self._expect(TokKind.LBRACE)
                    result = Const(dfunc['ret_ty'], 0)
                    while not self._at(TokKind.RBRACE):
                        if self._match(TokKind.KW_RETURN):
                            if not self._at(TokKind.SEMI):
                                result = self._parse_expr()
                            self._match(TokKind.SEMI)
                            # Skip rest of function
                            depth = 1
                            while depth > 0:
                                if self._peek().kind == TokKind.LBRACE: depth += 1
                                if self._peek().kind == TokKind.RBRACE: depth -= 1
                                if depth > 0: self._advance()
                            break
                        else:
                            self._parse_stmt()

                    self._pos = saved_pos
                    # Restore original variables
                    self._variables = saved_vars
                    return result
                else:
                    dest = self._new_val(name, INT32)
                    self._emit(CallInst(dest, name, args))
                    return dest

            # Variable reference
            if name in self._variables:
                return self._variables[name]

            # Built-in names (threadIdx, blockIdx, blockDim)
            if name in ('threadIdx', 'blockIdx', 'blockDim'):
                return Value(name, INT32)  # placeholder, resolved by .member access

            raise ParseError(f"Line {tok.line}: undefined variable '{name}'")

        if tok.kind == TokKind.LPAREN:
            self._advance()
            expr = self._parse_expr()
            self._expect(TokKind.RPAREN)
            return expr

        raise ParseError(f"Line {tok.line}: unexpected token '{tok.value}'")

    # -- Statement parsing ---------------------------------------------------

    def _parse_stmt(self):
        tok = self._peek()

        # const declaration: skip the 'const' keyword and parse as normal
        if tok.kind == TokKind.KW_CONST:
            self._advance()
            tok = self._peek()  # re-read after consuming 'const'

        # __shared__ declaration: __shared__ type name[size];
        if tok.kind == TokKind.KW_SHARED:
            self._advance()
            ty = self._parse_type()
            name = self._expect(TokKind.IDENT).value
            self._expect(TokKind.LBRACKET)
            size_tok = self._expect(TokKind.INT_LIT)
            size = int(size_tok.value)
            self._expect(TokKind.RBRACKET)
            self._expect(TokKind.SEMI)
            # Create a shared-memory pointer variable
            smem_ty = PtrTy(ScalarTy(ScalarType.FLOAT) if ty == FLOAT else ty, AddrSpace.SHARED)
            val = self._new_val(name, smem_ty)
            self._variables[name] = val
            # Store smem info for codegen (size in bytes)
            if not hasattr(self._kernel, '_shared_decls'):
                self._kernel._shared_decls = []
            self._kernel._shared_decls.append((name, ty, size))
            return

        # Variable declaration: type name [= expr];
        if tok.kind in (TokKind.KW_INT, TokKind.KW_UNSIGNED, TokKind.KW_FLOAT,
                        TokKind.KW_DOUBLE, TokKind.KW_VOID, TokKind.KW_LONG):
            ty = self._parse_type_with_ptr()
            name = self._expect(TokKind.IDENT).value
            val = self._new_val(name, ty)
            self._variables[name] = val

            if self._match(TokKind.ASSIGN):
                rhs = self._parse_expr()
                # For simplicity: treat the variable as the RHS value directly
                self._variables[name] = rhs if isinstance(rhs, Value) else val
                if isinstance(rhs, Const):
                    # Need to materialize the constant
                    self._emit(BinInst(val, BinOp.ADD, rhs, Const(ty, 0)))
                    self._variables[name] = val
                elif isinstance(rhs, Value) and rhs != val:
                    self._variables[name] = rhs
            self._expect(TokKind.SEMI)
            return

        # If statement
        if tok.kind == TokKind.KW_IF:
            self._advance()
            self._expect(TokKind.LPAREN)
            cond = self._parse_expr()
            self._expect(TokKind.RPAREN)

            true_bb = self._new_block("if_true")
            false_bb = self._new_block("if_false")
            merge_bb = self._new_block("if_merge")

            self._cur_block.terminator = CondBrTerm(cond, true_bb.label, false_bb.label)

            self._cur_block = true_bb
            self._parse_stmt_or_block()
            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(merge_bb.label)

            self._cur_block = false_bb
            if self._match(TokKind.KW_ELSE):
                self._parse_stmt_or_block()
            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(merge_bb.label)

            self._cur_block = merge_bb
            return

        # For loop — uses mutable variable model
        # Variables modified in the loop body/increment are written back to their
        # canonical Value so the condition block always reads the current value.
        if tok.kind == TokKind.KW_FOR:
            self._advance()
            self._expect(TokKind.LPAREN)

            # Snapshot variables before init to track which get modified
            vars_before = dict(self._variables)

            # Parse init statement
            self._parse_stmt()

            # Save token positions for condition and increment
            cond_start = self._pos
            depth = 0
            while not (self._peek().kind == TokKind.SEMI and depth == 0):
                if self._peek().kind == TokKind.LPAREN: depth += 1
                if self._peek().kind == TokKind.RPAREN: depth -= 1
                self._advance()
            self._advance()  # skip ;

            inc_start = self._pos
            depth = 0
            while not (self._peek().kind == TokKind.RPAREN and depth == 0):
                if self._peek().kind == TokKind.LPAREN: depth += 1
                if self._peek().kind == TokKind.RPAREN: depth -= 1
                self._advance()
            self._expect(TokKind.RPAREN)
            body_resume = self._pos

            # Snapshot variables after init (these are the "loop variables")
            loop_vars = dict(self._variables)

            # Build CFG
            cond_bb = self._new_block("for_cond")
            body_bb = self._new_block("for_body")
            inc_bb = self._new_block("for_inc")
            exit_bb = self._new_block("for_exit")

            self._cur_block.terminator = BrTerm(cond_bb.label)

            # Emit condition — must read current loop variable values
            self._cur_block = cond_bb
            self._pos = cond_start
            cond = self._parse_expr()
            cond_bb.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

            # Emit body (with break → exit_bb, continue → inc_bb)
            self._pos = body_resume
            self._cur_block = body_bb
            self._break_targets.append(exit_bb.label)
            self._continue_targets.append(inc_bb.label)
            self._parse_stmt_or_block()
            self._break_targets.pop()
            self._continue_targets.pop()
            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(inc_bb.label)

            # Emit increment
            self._cur_block = inc_bb
            saved_pos = self._pos
            self._pos = inc_start
            self._parse_expr()
            self._pos = saved_pos

            # Write back modified variables to their canonical loop-entry Values
            # so the condition block reads the updated values on the next iteration.
            for var_name, init_val in loop_vars.items():
                cur_val = self._variables.get(var_name)
                if cur_val is not None and cur_val is not init_val and isinstance(cur_val, Value):
                    # Variable was modified — emit a copy back to the init register
                    # PTX: mov dest, src (but we emit add dest, src, 0 for simplicity)
                    self._emit(BinInst(init_val, BinOp.ADD, cur_val, Const(init_val.ty, 0)))
                    self._variables[var_name] = init_val

            inc_bb.terminator = BrTerm(cond_bb.label)
            self._cur_block = exit_bb
            return

        # While loop
        if tok.kind == TokKind.KW_WHILE:
            self._advance()
            self._expect(TokKind.LPAREN)

            cond_bb = self._new_block("while_cond")
            body_bb = self._new_block("while_body")
            exit_bb = self._new_block("while_exit")

            self._cur_block.terminator = BrTerm(cond_bb.label)
            self._cur_block = cond_bb
            cond = self._parse_expr()
            self._expect(TokKind.RPAREN)
            cond_bb.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

            self._cur_block = body_bb
            self._break_targets.append(exit_bb.label)
            self._continue_targets.append(cond_bb.label)
            self._parse_stmt_or_block()
            self._break_targets.pop()
            self._continue_targets.pop()
            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(cond_bb.label)

            self._cur_block = exit_bb
            return

        # do/while loop
        if tok.kind == TokKind.KW_DO:
            self._advance()
            body_bb = self._new_block("do_body")
            cond_bb = self._new_block("do_cond")
            exit_bb = self._new_block("do_exit")

            self._cur_block.terminator = BrTerm(body_bb.label)
            self._cur_block = body_bb
            self._break_targets.append(exit_bb.label)
            self._continue_targets.append(cond_bb.label)
            self._parse_stmt_or_block()
            self._break_targets.pop()
            self._continue_targets.pop()
            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(cond_bb.label)

            self._cur_block = cond_bb
            self._expect(TokKind.KW_WHILE)
            self._expect(TokKind.LPAREN)
            cond = self._parse_expr()
            self._expect(TokKind.RPAREN)
            self._expect(TokKind.SEMI)
            cond_bb.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

            self._cur_block = exit_bb
            return

        # break
        if tok.kind == TokKind.KW_BREAK:
            self._advance()
            self._expect(TokKind.SEMI)
            if not self._break_targets:
                raise ParseError("break outside of loop")
            self._cur_block.terminator = BrTerm(self._break_targets[-1])
            # Dead code after break — create a new unreachable block
            self._cur_block = self._new_block("after_break")
            return

        # continue
        if tok.kind == TokKind.KW_CONTINUE:
            self._advance()
            self._expect(TokKind.SEMI)
            if not self._continue_targets:
                raise ParseError("continue outside of loop")
            self._cur_block.terminator = BrTerm(self._continue_targets[-1])
            self._cur_block = self._new_block("after_continue")
            return

        # switch/case
        if tok.kind == TokKind.KW_SWITCH:
            self._advance()
            self._expect(TokKind.LPAREN)
            switch_val = self._parse_expr()
            self._expect(TokKind.RPAREN)
            self._expect(TokKind.LBRACE)

            exit_bb = self._new_block("switch_exit")
            self._break_targets.append(exit_bb.label)

            # Collect cases
            cases = []  # (value, block_label)
            default_bb = None
            while not self._match(TokKind.RBRACE):
                if self._peek().kind == TokKind.KW_CASE:
                    self._advance()
                    case_val = self._parse_expr()
                    self._expect(TokKind.COLON)
                    case_bb = self._new_block("case")
                    cases.append((case_val, case_bb))
                    self._cur_block = case_bb
                elif self._peek().kind == TokKind.KW_DEFAULT:
                    self._advance()
                    self._expect(TokKind.COLON)
                    default_bb = self._new_block("default")
                    self._cur_block = default_bb
                else:
                    self._parse_stmt()

            self._break_targets.pop()

            # Build comparison chain: if switch_val == case_val → branch to case_bb
            chain_bb = self._kernel.blocks[-len(cases) - (1 if default_bb else 0) - 2]  # entry block before cases
            # Actually, rebuild chain from the entry point
            # Simple approach: emit a chain of if-else comparisons
            entry_bb = self._new_block("switch_dispatch")
            chain_bb.terminator = BrTerm(entry_bb.label) if chain_bb.terminator is None else chain_bb.terminator

            cur = entry_bb
            for case_val, case_bb in cases:
                cmp = self._new_val("cmp", ScalarTy(ScalarType.BOOL))
                self._cur_block = cur
                self._emit(CmpInst(cmp, CmpOp.EQ, switch_val, case_val))
                next_bb = self._new_block("switch_next")
                cur.terminator = CondBrTerm(cmp, case_bb.label, next_bb.label)
                cur = next_bb

            # Default or exit
            self._cur_block = cur
            if default_bb:
                cur.terminator = BrTerm(default_bb.label)
            else:
                cur.terminator = BrTerm(exit_bb.label)

            # Ensure all case blocks fall through to exit if no terminator
            for _, case_bb in cases:
                if case_bb.terminator is None:
                    case_bb.terminator = BrTerm(exit_bb.label)
            if default_bb and default_bb.terminator is None:
                default_bb.terminator = BrTerm(exit_bb.label)

            self._cur_block = exit_bb
            return

        # Return
        if tok.kind == TokKind.KW_RETURN:
            self._advance()
            self._expect(TokKind.SEMI)
            self._cur_block.terminator = RetTerm()
            return

        # Expression statement — check for array assignment: ptr[idx] = expr;
        saved_pos = self._pos
        lhs = self._parse_lvalue_or_expr()
        if self._match(TokKind.ASSIGN):
            rhs = self._parse_expr()
            if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                self._emit(StoreInst(addr=lhs, value=rhs))
            elif isinstance(lhs, Value) and lhs.name in self._variables:
                self._variables[lhs.name] = rhs
            self._expect(TokKind.SEMI)
            return
        # Compound assignment: +=, -=, *=
        for tok_kind, op in [(TokKind.PLUS_EQ, BinOp.ADD),
                             (TokKind.MINUS_EQ, BinOp.SUB),
                             (TokKind.STAR_EQ, BinOp.MUL)]:
            if self._match(tok_kind):
                rhs = self._parse_expr()
                if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                    # Array compound: load current, compute, store back
                    cur = self._new_val("cur", lhs.ty.pointee)
                    self._emit(LoadInst(cur, lhs))
                    result = self._new_val("compound", cur.ty)
                    self._emit(BinInst(result, op, cur, rhs))
                    self._emit(StoreInst(addr=lhs, value=result))
                else:
                    result = self._new_val("compound", lhs.ty if isinstance(lhs, Value) else INT32)
                    self._emit(BinInst(result, op, lhs, rhs))
                    if isinstance(lhs, Value) and lhs.name in self._variables:
                        self._variables[lhs.name] = result
                self._expect(TokKind.SEMI)
                return
        self._expect(TokKind.SEMI)

    def _parse_lvalue_or_expr(self) -> Operand:
        """Parse an expression that might be an lvalue (address for assignment).

        For ptr[index], returns the ADDRESS (PtrTy) without loading.
        For other expressions, returns the value normally.
        """
        tok = self._peek()
        if tok.kind == TokKind.IDENT:
            name = tok.value
            if name in self._variables:
                var = self._variables[name]
                if isinstance(var.ty, PtrTy):
                    self._advance()
                    if self._match(TokKind.LBRACKET):
                        index = self._parse_expr()
                        self._expect(TokKind.RBRACKET)
                        elem_size = var.ty.pointee.size
                        if elem_size != 1:
                            scaled = self._new_val("scale", INT32)
                            self._emit(BinInst(scaled, BinOp.MUL, index, Const(index.ty if isinstance(index, Value) else INT32, elem_size)))
                            index = scaled
                        addr = self._new_val("addr", var.ty)
                        self._emit(BinInst(addr, BinOp.ADD, var, index))
                        return addr  # Return ADDRESS, not loaded value
        # Fall back to normal expression parsing
        return self._parse_expr()

    def _parse_stmt_or_block(self):
        if self._match(TokKind.LBRACE):
            while not self._match(TokKind.RBRACE):
                self._parse_stmt()
        else:
            self._parse_stmt()

    # -- Top-level parsing ---------------------------------------------------

    def _parse_kernel(self):
        self._expect(TokKind.KW_GLOBAL)
        # Skip optional __launch_bounds__(maxThreads, minBlocks)
        if self._at(TokKind.IDENT) and self._peek().value == '__launch_bounds__':
            self._advance()
            self._expect(TokKind.LPAREN)
            depth = 1
            while depth > 0:
                if self._peek().kind == TokKind.LPAREN: depth += 1
                if self._peek().kind == TokKind.RPAREN: depth -= 1
                self._advance()
            # Consumed the closing )
        ret_ty = self._parse_type()  # should be void
        name = self._expect(TokKind.IDENT).value

        # Parameters
        self._expect(TokKind.LPAREN)
        params = []
        if not self._at(TokKind.RPAREN):
            while True:
                pty = self._parse_type_with_ptr()
                pname = self._expect(TokKind.IDENT).value
                params.append(KernelParam(pname, pty))
                if not self._match(TokKind.COMMA):
                    break
        self._expect(TokKind.RPAREN)

        self._kernel = Kernel(name=name, params=params)
        self._variables = {}
        self._block_count = 0

        # Load kernel parameters into variables
        entry = self._new_block("entry")
        self._cur_block = entry
        for i, p in enumerate(params):
            val = self._new_val(p.name, p.ty)
            self._emit(ParamInst(val, i, p.name))
            self._variables[p.name] = val

        # Parse body
        self._expect(TokKind.LBRACE)
        while not self._match(TokKind.RBRACE):
            self._parse_stmt()

        # Ensure terminator
        if self._cur_block.terminator is None:
            self._cur_block.terminator = RetTerm()

        return self._kernel

    def _parse_struct_def(self):
        """Parse: struct Name { type field; ... };"""
        self._expect(TokKind.KW_STRUCT)
        name = self._expect(TokKind.IDENT).value
        self._expect(TokKind.LBRACE)
        fields = []
        while not self._at(TokKind.RBRACE):
            fty = self._parse_type_with_ptr()
            fname = self._expect(TokKind.IDENT).value
            self._expect(TokKind.SEMI)
            fields.append((fname, fty))
        self._expect(TokKind.RBRACE)
        self._expect(TokKind.SEMI)
        sty = StructTy(name, tuple(fields))
        self._struct_types[name] = sty
        return sty

    def _parse_typedef(self):
        """Parse: typedef struct Name Name;  or  typedef type name;"""
        self._expect(TokKind.KW_TYPEDEF)
        if self._at(TokKind.KW_STRUCT):
            sty = self._parse_struct_def()
            # typedef struct Foo Foo; — the name after } is the typedef alias
            # But we already consumed ;. Check if there's another ident.
            self._typedefs[sty.name] = sty
        else:
            ty = self._parse_type_with_ptr()
            alias = self._expect(TokKind.IDENT).value
            self._expect(TokKind.SEMI)
            self._typedefs[alias] = ty

    def _parse_device_func(self):
        """Parse __device__ function and store for inlining."""
        self._expect(TokKind.KW_DEVICE)
        ret_ty = self._parse_type_with_ptr()
        name = self._expect(TokKind.IDENT).value

        self._expect(TokKind.LPAREN)
        params = []
        if not self._at(TokKind.RPAREN):
            while True:
                pty = self._parse_type_with_ptr()
                pname = self._expect(TokKind.IDENT).value
                params.append((pname, pty))
                if not self._match(TokKind.COMMA):
                    break
        self._expect(TokKind.RPAREN)

        # Save token range for the body to replay during inlining
        body_start = self._pos
        # Skip the body
        self._expect(TokKind.LBRACE)
        depth = 1
        while depth > 0:
            if self._peek().kind == TokKind.LBRACE: depth += 1
            if self._peek().kind == TokKind.RBRACE: depth -= 1
            self._advance()
        body_end = self._pos

        self._device_funcs[name] = {
            'name': name,
            'ret_ty': ret_ty,
            'params': params,
            'body_start': body_start,
            'body_end': body_end,
        }

    def parse_module(self) -> Module:
        mod = Module()
        self._device_funcs = {}

        while not self._at(TokKind.EOF):
            if self._at(TokKind.KW_GLOBAL):
                mod.kernels.append(self._parse_kernel())
            elif self._at(TokKind.KW_DEVICE):
                self._parse_device_func()
            elif self._at(TokKind.KW_STRUCT):
                self._parse_struct_def()
            elif self._at(TokKind.KW_TYPEDEF):
                self._parse_typedef()
            else:
                self._advance()
        return mod


def parse(source: str) -> Module:
    """Parse CUDA-subset C source into an IR Module."""
    tokens = lex(source)
    return Parser(tokens).parse_module()
