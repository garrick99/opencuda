"""
OpenCUDA type system.

Supports the CUDA-subset types needed for compute kernels:
  - Scalar: int, unsigned int, float, double, half
  - Pointer: T* with address space (global, shared, local)
  - Void (for kernel return type)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class ScalarType(Enum):
    VOID   = "void"
    BOOL   = "bool"
    INT8   = "char"
    UINT8  = "unsigned char"
    INT16  = "short"
    UINT16 = "unsigned short"
    INT32  = "int"
    UINT32 = "unsigned int"
    INT64  = "long long"
    UINT64 = "unsigned long long"
    FLOAT  = "float"
    DOUBLE = "double"
    HALF   = "half"


@dataclass(frozen=True)
class Type:
    """Base type."""
    pass


@dataclass(frozen=True)
class ScalarTy(Type):
    scalar: ScalarType

    @property
    def size(self) -> int:
        """Size in bytes."""
        sizes = {
            ScalarType.VOID: 0, ScalarType.BOOL: 1,
            ScalarType.INT8: 1, ScalarType.UINT8: 1,
            ScalarType.INT16: 2, ScalarType.UINT16: 2,
            ScalarType.INT32: 4, ScalarType.UINT32: 4,
            ScalarType.INT64: 8, ScalarType.UINT64: 8,
            ScalarType.FLOAT: 4, ScalarType.DOUBLE: 8,
            ScalarType.HALF: 2,
        }
        return sizes[self.scalar]

    @property
    def is_float(self) -> bool:
        return self.scalar in (ScalarType.FLOAT, ScalarType.DOUBLE, ScalarType.HALF)

    @property
    def is_signed(self) -> bool:
        return self.scalar in (ScalarType.INT8, ScalarType.INT16, ScalarType.INT32,
                               ScalarType.INT64)

    def __str__(self) -> str:
        return self.scalar.value


class AddrSpace(Enum):
    GENERIC = "generic"
    GLOBAL  = "global"
    SHARED  = "shared"
    LOCAL   = "local"
    CONST   = "const"


@dataclass(frozen=True)
class PtrTy(Type):
    pointee: Type
    addr_space: AddrSpace = AddrSpace.GLOBAL

    @property
    def size(self) -> int:
        return 8  # 64-bit pointers on SM_120

    def __str__(self) -> str:
        return f"{self.pointee}*"


# Common type shorthands
VOID   = ScalarTy(ScalarType.VOID)
BOOL   = ScalarTy(ScalarType.BOOL)
INT32  = ScalarTy(ScalarType.INT32)
UINT32 = ScalarTy(ScalarType.UINT32)
INT64  = ScalarTy(ScalarType.INT64)
UINT64 = ScalarTy(ScalarType.UINT64)
FLOAT  = ScalarTy(ScalarType.FLOAT)
DOUBLE = ScalarTy(ScalarType.DOUBLE)
HALF   = ScalarTy(ScalarType.HALF)
