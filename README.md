# OpenCUDA

Open-source CUDA compiler targeting NVIDIA Blackwell (SM_120 / RTX 5090).

Compiles CUDA-subset C directly to PTX, with optional cubin generation via [OpenPTXas](https://github.com/garrick99/openptxas).

## Status

**Working.** Compiles `vector_add` kernel from C source → PTX → cubin → correct results on RTX 5090 (256 elements verified).

## Usage

```bash
# C → PTX
python -m opencuda kernel.cu --emit-ptx

# C → cubin (via OpenPTXas backend)
python -m opencuda kernel.cu --out kernel.cubin

# Verbose output
python -m opencuda kernel.cu --emit-ptx -v
```

## Example

```c
__global__ void vector_add(float *out, float *a, float *b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
```

```
$ python -m opencuda vector_add.cu --emit-ptx -v
[opencuda] Parsing vector_add.cu...
[opencuda] 1 kernel(s) found
[opencuda] Wrote PTX: vector_add.ptx
```

## Supported C Features

- `__global__` kernel functions
- Types: `int`, `unsigned int`, `float`, `double`, `void`, pointers (`T*`)
- Built-ins: `threadIdx.x/y/z`, `blockIdx.x/y/z`, `blockDim.x/y/z`, `__syncthreads()`
- Control flow: `if`/`else`, `for` loops, `return`
- Arithmetic: `+ - * / % & | ^ << >>`
- Comparisons: `== != < <= > >=`
- Array indexing: `ptr[i]` (read and write)
- Pointer arithmetic

## Architecture

```
  CUDA-subset C source
        |
  [Lexer]  — tokenizes C
        |
  [Parser] — recursive descent, generates SSA IR
        |
  [IR]     — typed SSA basic blocks with control flow
        |
  [Codegen] — lowers IR to PTX text
        |
  [OpenPTXas] — assembles PTX to SM_120 cubin (optional)
        |
  [RTX 5090]
```

## Stack

| Layer | Project | Role |
|-------|---------|------|
| Frontend | **OpenCUDA** | C → PTX |
| Backend | [OpenPTXas](https://github.com/garrick99/openptxas) | PTX → cubin |
| Hardware | RTX 5090 | SM_120 / Blackwell |

## Requirements

- Python 3.11+
- OpenPTXas (for cubin generation)
- CUDA toolkit (only for GPU test harness compilation)

## License

Private. All rights reserved.
