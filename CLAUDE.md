# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

OpenCUDA is a pure-Python CUDA compiler that translates CUDA-subset C to PTX assembly (and optionally to SM_120 cubin via OpenPTXas). No NVIDIA toolchain required for PTX emission; `ptxas` is only needed for the validation test suite.

## Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run all tests
pytest opencuda/tests/test_compiler.py -v

# Run a single test by name (e.g. vector_add)
pytest opencuda/tests/test_compiler.py -v -k vector_add

# Compile a kernel to PTX
python -m opencuda tests/vector_add.cu --emit-ptx

# Compile to cubin (requires openptxas)
python -m opencuda tests/vector_add.cu --out kernel.cubin
```

The test suite has two parametrized test functions, each running over all 33 `.cu` files in `tests/`:
- `test_parse_and_emit` — verifies PTX output contains `.version`, `.entry`, and `ret;`
- `test_ptxas_validates` — feeds PTX to NVIDIA's `ptxas` for SM_120 validation

## Architecture

The compilation pipeline is a straight sequence of passes:

```
CUDA C (.cu)
  → [Lexer]        lexer.py         Token stream
  → [Preprocessor] preprocess.py    #define substitution
  → [Parser]       parser.py        SSA IR (Module/Kernel/BasicBlock/Value)
  → [Optimizer]    optimize.py      Constant folding, CSE, loop unrolling
  → [Codegen]      emit.py          PTX 9.0 text (sm_120)
  → [OpenPTXas]    (external)       cubin binary
```

### IR (`opencuda/ir/`)

- **nodes.py** — SSA IR: `Value` (virtual register, one assignment), `Const`, instruction variants (`BinInst`, `CmpInst`, `LoadInst`, `StoreInst`, `CvtInst`, `CallInst`, `PhiInst`, `ParamInst`), terminators (`RetTerm`, `BrTerm`, `CondBrTerm`), `BasicBlock`, `Kernel`, `Module`. All nodes are frozen dataclasses.
- **types.py** — Type system: `ScalarTy`, `PtrTy` (with address space), `StructTy`. Frozen for hashing and CSE keying.
- **optimize.py** — Three passes run in sequence: loop unrolling (via `unroll.py`), constant folding (with strength reduction), CSE. **Critical constraint: all optimizations are per-basic-block only** — never cross block boundaries to avoid loop writeback bugs.
- **unroll.py** — Detects for-loops with compile-time trip counts ≤16. Chains loop-carried variables explicitly (output of iteration N feeds input of iteration N+1).

### Codegen (`opencuda/codegen/emit.py`)

- `PTXEmitter` walks basic blocks and emits PTX instructions.
- Register naming by value ID: integer (`r`), float (`f`), double (`fd`/`rd`), predicates (`p0–p127`).
- Widen cache tracks `cvt.u64.u32` to avoid re-widening the same register twice (CVT CSE).
- Naive register allocation: no coloring or spilling — each `Value` gets its own PTX register.

### Frontend (`opencuda/frontend/`)

- **lexer.py** — Regex-based tokenizer for CUDA-subset C.
- **preprocess.py** — `#define NAME VALUE` text substitution; all other directives are ignored.
- **parser.py** — Recursive descent with operator precedence climbing. Emits SSA IR directly (no AST stage). Handles CUDA intrinsics (`threadIdx`, `blockIdx`, `blockDim`, `__syncthreads`, atomics, warp shuffles, `__shared__`, `__device__`).

## Key Design Constraints

- **SSA form**: every `Value` is assigned exactly once; `PhiInst` merges values at control-flow join points.
- **Per-block optimization safety**: constant folding and CSE must never propagate values across basic block boundaries — doing so can incorrectly eliminate loop-carried updates.
- **Type-aware CSE**: CSE keys include the result type to prevent merging int and float computations that happen to share the same operand IDs.
- **PTX target**: SM_120 (Blackwell / RTX 5090), PTX ISA version 9.0, 64-bit addressing throughout.

## Known Limitations

- `float16` is parsed but emitted as `f32`
- Device function inlining does not support multiple return points inside if-blocks
- Integer division/remainder emits PTX `div`/`rem` and relies on OpenPTXas for SASS expansion
- Register allocation is naive (by value ID); no coloring, no spilling
- No texture/surface memory, cooperative groups, or tensor operations
