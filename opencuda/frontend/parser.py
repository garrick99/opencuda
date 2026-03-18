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
from ..ir.types import (Type, ScalarTy, PtrTy, AddrSpace, ScalarType,
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
        self._variables: dict[str, Value] = {}  # name → SSA value (pointer to stack slot or register)
        self._block_count = 0

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

    def _new_block(self, label: str = None) -> BasicBlock:
        if label is None:
            self._block_count += 1
            label = f"BB{self._block_count}"
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
        if self._match(TokKind.ASSIGN):
            rhs = self._parse_assign_expr()
            # lhs must be an addressable value — emit store
            if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                self._emit(StoreInst(addr=lhs, value=rhs))
                return rhs
            # Direct variable assignment
            return rhs
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
        lhs = self._parse_cmp_expr()
        while self._match(TokKind.AND):
            rhs = self._parse_cmp_expr()
            dest = self._new_val("and", INT32)
            self._emit(BinInst(dest, BinOp.AND, lhs, rhs))
            lhs = dest
        return lhs

    def _parse_cmp_expr(self) -> Operand:
        lhs = self._parse_add_expr()
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

    def _parse_add_expr(self) -> Operand:
        lhs = self._parse_mul_expr()
        while True:
            if self._match(TokKind.PLUS):
                rhs = self._parse_mul_expr()
                dest = self._new_val("add", lhs.ty if isinstance(lhs, Value) else INT32)
                self._emit(BinInst(dest, BinOp.ADD, lhs, rhs))
                lhs = dest
            elif self._match(TokKind.MINUS):
                rhs = self._parse_mul_expr()
                dest = self._new_val("sub", lhs.ty if isinstance(lhs, Value) else INT32)
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
                dest = self._new_val("mul", lhs.ty if isinstance(lhs, Value) else INT32)
                self._emit(BinInst(dest, BinOp.MUL, lhs, rhs))
                lhs = dest
            elif self._match(TokKind.SLASH):
                rhs = self._parse_unary_expr()
                dest = self._new_val("div", lhs.ty if isinstance(lhs, Value) else INT32)
                self._emit(BinInst(dest, BinOp.DIV, lhs, rhs))
                lhs = dest
            else:
                break
        return lhs

    def _parse_unary_expr(self) -> Operand:
        if self._match(TokKind.STAR):
            # Pointer dereference
            operand = self._parse_unary_expr()
            if isinstance(operand, Value) and isinstance(operand.ty, PtrTy):
                dest = self._new_val("deref", operand.ty.pointee)
                self._emit(LoadInst(dest, operand))
                return dest
        if self._match(TokKind.MINUS):
            operand = self._parse_unary_expr()
            dest = self._new_val("neg", operand.ty if isinstance(operand, Value) else INT32)
            self._emit(BinInst(dest, BinOp.SUB, Const(INT32, 0), operand))
            return dest
        return self._parse_postfix_expr()

    def _parse_postfix_expr(self) -> Operand:
        lhs = self._parse_primary_expr()

        while True:
            if self._match(TokKind.LBRACKET):
                # Array indexing: ptr[index]
                index = self._parse_expr()
                self._expect(TokKind.RBRACKET)
                if isinstance(lhs, Value) and isinstance(lhs.ty, PtrTy):
                    elem_size = lhs.ty.pointee.size
                    # addr = base + index * elem_size
                    if elem_size != 1:
                        scaled = self._new_val("scale", INT32)
                        self._emit(BinInst(scaled, BinOp.MUL, index, Const(INT32, elem_size)))
                        index = scaled
                    addr = self._new_val("addr", lhs.ty)
                    self._emit(BinInst(addr, BinOp.ADD, lhs, index))
                    # Load the value
                    dest = self._new_val("elem", lhs.ty.pointee)
                    self._emit(LoadInst(dest, addr))
                    lhs = dest
            elif self._match(TokKind.DOT):
                # Member access: threadIdx.x, blockIdx.y, etc.
                member = self._expect(TokKind.IDENT).value
                if isinstance(lhs, Value) and lhs.name in ('threadIdx', 'blockIdx', 'blockDim'):
                    builtin = f"{lhs.name}.{member}"
                    dest = self._new_val(builtin.replace('.', '_'), INT32)
                    self._emit(CallInst(dest, builtin))
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

        # For loop
        if tok.kind == TokKind.KW_FOR:
            self._advance()
            self._expect(TokKind.LPAREN)
            # Init
            self._parse_stmt()
            # Condition
            cond_bb = self._new_block("for_cond")
            body_bb = self._new_block("for_body")
            exit_bb = self._new_block("for_exit")

            self._cur_block.terminator = BrTerm(cond_bb.label)
            self._cur_block = cond_bb
            cond = self._parse_expr()
            self._expect(TokKind.SEMI)
            # Save increment expression position for later
            inc_start = self._pos
            # Skip increment for now
            depth = 0
            while not (self._peek().kind == TokKind.RPAREN and depth == 0):
                if self._peek().kind == TokKind.LPAREN: depth += 1
                if self._peek().kind == TokKind.RPAREN: depth -= 1
                self._advance()
            self._expect(TokKind.RPAREN)

            cond_bb.terminator = CondBrTerm(cond, body_bb.label, exit_bb.label)

            self._cur_block = body_bb
            self._parse_stmt_or_block()

            # Increment
            inc_bb = self._new_block("for_inc")
            if self._cur_block.terminator is None:
                self._cur_block.terminator = BrTerm(inc_bb.label)
            self._cur_block = inc_bb
            # Re-parse the increment expression
            saved_pos = self._pos
            self._pos = inc_start
            self._parse_expr()
            self._pos = saved_pos
            inc_bb.terminator = BrTerm(cond_bb.label)

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
                # lhs is an address — store to it
                self._emit(StoreInst(addr=lhs, value=rhs))
            self._expect(TokKind.SEMI)
            return
        # Compound assignment: +=, -=, *=
        for tok_kind, op in [(TokKind.PLUS_EQ, BinOp.ADD),
                             (TokKind.MINUS_EQ, BinOp.SUB),
                             (TokKind.STAR_EQ, BinOp.MUL)]:
            if self._match(tok_kind):
                rhs = self._parse_expr()
                result = self._new_val("compound", lhs.ty if isinstance(lhs, Value) else INT32)
                self._emit(BinInst(result, op, lhs, rhs))
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
                            self._emit(BinInst(scaled, BinOp.MUL, index, Const(INT32, elem_size)))
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

    def parse_module(self) -> Module:
        mod = Module()
        while not self._at(TokKind.EOF):
            if self._at(TokKind.KW_GLOBAL):
                mod.kernels.append(self._parse_kernel())
            else:
                self._advance()  # skip non-kernel top-level
        return mod


def parse(source: str) -> Module:
    """Parse CUDA-subset C source into an IR Module."""
    tokens = lex(source)
    return Parser(tokens).parse_module()
