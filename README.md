# OpenCUDA

Open-source CUDA compiler. Compiles CUDA C directly to PTX assembly and executable GPU binaries — no NVIDIA software required.

Targets NVIDIA Blackwell (SM_120 / RTX 5090). Paired with [OpenPTXas](https://github.com/garrick99/openptxas) for the full pipeline: **CUDA C → PTX → cubin**.

## What You Can Do

```bash
# Compile a CUDA kernel to PTX
python -m opencuda kernel.cu --emit-ptx

# Compile to executable cubin (via OpenPTXas)
python -m opencuda kernel.cu --out kernel.cubin

# Run tests (66 tests, all passing)
pytest opencuda/tests/test_compiler.py -v
```

The generated PTX passes NVIDIA's own ptxas validation. The generated cubins execute on RTX 5090 hardware.

## What's Supported

**33 test kernels compile through the full pipeline with zero errors:**

| Feature | Status | Examples |
|---|---|---|
| **Arithmetic** | `+ - * / %` all operators | vector_add, saxpy, matmul |
| **Bitwise** | `& \| ^ ~ << >>` | bitwise_test, histogram |
| **Compound assignment** | `+= -= *= /= %= &= \|= ^= <<= >>=` | compound_assign |
| **Control flow** | if/else, for, while, do/while | for_loop, while_loop, do_while |
| **Switch/case** | switch/case/default with break | switch_test |
| **Break/continue** | Loop exit and skip | break_continue |
| **Ternary** | `cond ? a : b` | ternary_test |
| **Types** | int, unsigned, float, double, pointers, structs | cast_test, type_promo, struct_test |
| **Shared memory** | `__shared__` arrays with `__syncthreads()` | shared_mem, matmul_tiled, reduce |
| **Atomics** | 9 atomic ops (add, sub, min, max, and, or, xor, exch, cas) | atomic_test |
| **Warp shuffles** | `__shfl_sync`, `__shfl_down_sync`, `__ballot_sync`, etc. | warp_test |
| **Device functions** | `__device__` with inlining and return values | device_func, device_return |
| **Preprocessor** | `#define` text substitution | define_test |
| **Multi-kernel** | Multiple `__global__` functions per file | multi_kernel |
| **Optimization** | Constant folding, CSE, loop unrolling (≤16x) | for_loop |

## Architecture

```
CUDA C source (.cu)
    ↓
[Lexer]         Tokenize (regex-based, all C operators + CUDA keywords)
    ↓
[Preprocessor]  #define substitution
    ↓
[Parser]        Recursive descent → SSA IR (basic blocks + CFG)
    ↓
[Optimizer]     Constant folding, CSE, loop unrolling
    ↓
[Codegen]       PTX 9.0 text emission (sm_120, 64-bit addressing)
    ↓
[OpenPTXas]     PTX → cubin binary (optional)
    ↓
GPU execution   RTX 5090 verified ✓
```

Pure Python 3.11+. No dependencies beyond pytest for testing.

## Requirements

- Python 3.11+
- For cubin generation: [OpenPTXas](https://github.com/garrick99/openptxas)
- For PTX validation: NVIDIA ptxas (optional, only for `test_ptxas_validates`)
- For GPU execution: NVIDIA CUDA toolkit + RTX 5090/4090

## Known Limitations

- No `float16` SASS instructions (parsed, emits as f32)
- Device function inlining doesn't support multiple return points inside if-blocks
- Integer division/remainder emits PTX `div`/`rem` (requires ptxas or OpenPTXas for SASS expansion)
- No texture/surface memory operations
- No cooperative groups or tensor operations
- Register allocation is naive (by value ID, no coloring or spilling)

## License

See LICENSE file.
