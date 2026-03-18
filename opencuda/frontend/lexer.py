"""
OpenCUDA lexer — tokenizes CUDA-subset C source code.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from enum import Enum, auto


class TokKind(Enum):
    # Literals
    INT_LIT   = auto()
    FLOAT_LIT = auto()
    IDENT     = auto()

    # Keywords
    KW_GLOBAL   = auto()  # __global__
    KW_SHARED   = auto()  # __shared__
    KW_DEVICE   = auto()  # __device__
    KW_VOID     = auto()
    KW_INT      = auto()
    KW_UNSIGNED = auto()
    KW_FLOAT    = auto()
    KW_DOUBLE   = auto()
    KW_HALF     = auto()
    KW_CHAR     = auto()
    KW_SHORT    = auto()
    KW_LONG     = auto()
    KW_IF       = auto()
    KW_ELSE     = auto()
    KW_FOR      = auto()
    KW_WHILE    = auto()
    KW_RETURN   = auto()
    KW_STRUCT   = auto()
    KW_TYPEDEF  = auto()
    KW_CONST    = auto()
    KW_BREAK    = auto()
    KW_CONTINUE = auto()

    # Operators
    PLUS     = auto()
    MINUS    = auto()
    STAR     = auto()
    SLASH    = auto()
    PERCENT  = auto()
    AMP      = auto()
    PIPE     = auto()
    CARET    = auto()
    TILDE    = auto()
    BANG     = auto()
    LSHIFT   = auto()
    RSHIFT   = auto()
    EQ       = auto()
    NE       = auto()
    LT       = auto()
    LE       = auto()
    GT       = auto()
    GE       = auto()
    AND      = auto()  # &&
    OR       = auto()  # ||
    ASSIGN   = auto()  # =
    PLUS_EQ  = auto()
    MINUS_EQ = auto()
    STAR_EQ  = auto()

    # Punctuation
    LPAREN   = auto()
    RPAREN   = auto()
    LBRACE   = auto()
    RBRACE   = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    SEMI     = auto()
    COMMA    = auto()
    DOT      = auto()
    ARROW    = auto()  # ->

    # Special
    PLUSPLUS  = auto()
    MINUSMINUS = auto()

    EOF = auto()


@dataclass
class Token:
    kind: TokKind
    value: str
    line: int
    col: int


_KEYWORDS = {
    '__global__': TokKind.KW_GLOBAL,
    '__shared__': TokKind.KW_SHARED,
    '__device__': TokKind.KW_DEVICE,
    'void': TokKind.KW_VOID,
    'int': TokKind.KW_INT,
    'unsigned': TokKind.KW_UNSIGNED,
    'float': TokKind.KW_FLOAT,
    'double': TokKind.KW_DOUBLE,
    'half': TokKind.KW_HALF,
    'char': TokKind.KW_CHAR,
    'short': TokKind.KW_SHORT,
    'long': TokKind.KW_LONG,
    'if': TokKind.KW_IF,
    'else': TokKind.KW_ELSE,
    'for': TokKind.KW_FOR,
    'while': TokKind.KW_WHILE,
    'return': TokKind.KW_RETURN,
    'struct': TokKind.KW_STRUCT,
    'typedef': TokKind.KW_TYPEDEF,
    'const': TokKind.KW_CONST,
    'break': TokKind.KW_BREAK,
    'continue': TokKind.KW_CONTINUE,
}

_TOKEN_RE = re.compile(r"""
    (?P<COMMENT_LINE>   //[^\n]*            ) |
    (?P<COMMENT_BLOCK>  /\*.*?\*/           ) |
    (?P<FLOAT_LIT>      [0-9]+\.[0-9]*[fF]?
                      | [0-9]*\.[0-9]+[fF]?
                      | [0-9]+[fF]          ) |
    (?P<HEX_LIT>        0[xX][0-9a-fA-F]+[uUlL]* ) |
    (?P<INT_LIT>        [0-9]+[uUlL]*       ) |
    (?P<IDENT>          [a-zA-Z_][a-zA-Z0-9_]* ) |
    (?P<LSHIFT>         <<                  ) |
    (?P<RSHIFT>         >>                  ) |
    (?P<PLUSPLUS>        \+\+               ) |
    (?P<MINUSMINUS>     --                  ) |
    (?P<PLUS_EQ>        \+=                 ) |
    (?P<MINUS_EQ>       -=                  ) |
    (?P<STAR_EQ>        \*=                 ) |
    (?P<EQ>             ==                  ) |
    (?P<NE>             !=                  ) |
    (?P<LE>             <=                  ) |
    (?P<GE>             >=                  ) |
    (?P<AND>            &&                  ) |
    (?P<OR>             \|\|               ) |
    (?P<ARROW>          ->                  ) |
    (?P<PLUS>           \+                  ) |
    (?P<MINUS>          -                   ) |
    (?P<STAR>           \*                  ) |
    (?P<SLASH>          /                   ) |
    (?P<PERCENT>        %                   ) |
    (?P<AMP>            &                   ) |
    (?P<PIPE>           \|                  ) |
    (?P<CARET>          \^                  ) |
    (?P<TILDE>          ~                   ) |
    (?P<BANG>           !                   ) |
    (?P<LT>             <                   ) |
    (?P<GT>             >                   ) |
    (?P<ASSIGN>         =                   ) |
    (?P<LPAREN>         \(                  ) |
    (?P<RPAREN>         \)                  ) |
    (?P<LBRACE>         \{                  ) |
    (?P<RBRACE>         \}                  ) |
    (?P<LBRACKET>       \[                  ) |
    (?P<RBRACKET>       \]                  ) |
    (?P<SEMI>           ;                   ) |
    (?P<COMMA>          ,                   ) |
    (?P<DOT>            \.                  ) |
    (?P<WS>             [ \t\r]+            ) |
    (?P<NEWLINE>        \n                  )
""", re.VERBOSE | re.DOTALL)

_GROUP_TO_KIND = {
    'PLUS': TokKind.PLUS, 'MINUS': TokKind.MINUS, 'STAR': TokKind.STAR,
    'SLASH': TokKind.SLASH, 'PERCENT': TokKind.PERCENT,
    'AMP': TokKind.AMP, 'PIPE': TokKind.PIPE, 'CARET': TokKind.CARET,
    'TILDE': TokKind.TILDE, 'BANG': TokKind.BANG,
    'LSHIFT': TokKind.LSHIFT, 'RSHIFT': TokKind.RSHIFT,
    'EQ': TokKind.EQ, 'NE': TokKind.NE,
    'LT': TokKind.LT, 'LE': TokKind.LE, 'GT': TokKind.GT, 'GE': TokKind.GE,
    'AND': TokKind.AND, 'OR': TokKind.OR,
    'ASSIGN': TokKind.ASSIGN,
    'PLUS_EQ': TokKind.PLUS_EQ, 'MINUS_EQ': TokKind.MINUS_EQ,
    'STAR_EQ': TokKind.STAR_EQ,
    'LPAREN': TokKind.LPAREN, 'RPAREN': TokKind.RPAREN,
    'LBRACE': TokKind.LBRACE, 'RBRACE': TokKind.RBRACE,
    'LBRACKET': TokKind.LBRACKET, 'RBRACKET': TokKind.RBRACKET,
    'SEMI': TokKind.SEMI, 'COMMA': TokKind.COMMA, 'DOT': TokKind.DOT,
    'ARROW': TokKind.ARROW,
    'PLUSPLUS': TokKind.PLUSPLUS, 'MINUSMINUS': TokKind.MINUSMINUS,
}


def lex(source: str) -> list[Token]:
    """Tokenize CUDA-subset C source code."""
    tokens = []
    line = 1
    col = 1

    for m in _TOKEN_RE.finditer(source):
        kind_name = m.lastgroup
        val = m.group()

        if kind_name in ('WS', 'COMMENT_LINE', 'COMMENT_BLOCK'):
            line += val.count('\n')
            continue
        if kind_name == 'NEWLINE':
            line += 1
            col = 1
            continue

        if kind_name == 'IDENT':
            tok_kind = _KEYWORDS.get(val, TokKind.IDENT)
        elif kind_name == 'INT_LIT' or kind_name == 'HEX_LIT':
            tok_kind = TokKind.INT_LIT
        elif kind_name == 'FLOAT_LIT':
            tok_kind = TokKind.FLOAT_LIT
        else:
            tok_kind = _GROUP_TO_KIND.get(kind_name)
            if tok_kind is None:
                continue

        tokens.append(Token(tok_kind, val, line, col))

    tokens.append(Token(TokKind.EOF, '', line, col))
    return tokens
