// Minimal CUDA driver API loader for OpenCUDA/OpenPTXas cubins.
// Compiles with: nvcc -o gpu_loader gpu_loader.cu -lcuda
// Usage: gpu_loader kernel.cubin kernel_name [N]

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char* str; cuGetErrorString(err, &str); \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", __FILE__, __LINE__, str, err); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <cubin_path> <kernel_name> [N]\n", argv[0]);
        return 1;
    }
    const char* cubin_path = argv[1];
    const char* kernel_name = argv[2];
    int N = (argc > 3) ? atoi(argv[3]) : 256;

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CHECK_CU(cuDevicePrimaryCtxRetain(&ctx, dev));
    CHECK_CU(cuCtxSetCurrent(ctx));

    // Load cubin
    CUmodule mod;
    CUresult load_err = cuModuleLoad(&mod, cubin_path);
    if (load_err != CUDA_SUCCESS) {
        const char* str; cuGetErrorString(load_err, &str);
        fprintf(stderr, "Failed to load cubin '%s': %s\n", cubin_path, str);
        return 1;
    }
    CUfunction func;
    CUresult func_err = cuModuleGetFunction(&func, mod, kernel_name);
    if (func_err != CUDA_SUCCESS) {
        const char* str; cuGetErrorString(func_err, &str);
        fprintf(stderr, "Failed to find kernel '%s': %s\n", kernel_name, str);
        return 1;
    }

    printf("Loaded %s::%s (N=%d)\n", cubin_path, kernel_name, N);

    // Allocate device memory
    CUdeviceptr d_a, d_b, d_out;
    size_t bytes = N * sizeof(float);
    CHECK_CU(cuMemAlloc(&d_a, bytes));
    CHECK_CU(cuMemAlloc(&d_b, bytes));
    CHECK_CU(cuMemAlloc(&d_out, bytes));

    // Initialize host data
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }
    CHECK_CU(cuMemcpyHtoD(d_a, h_a, bytes));
    CHECK_CU(cuMemcpyHtoD(d_b, h_b, bytes));

    // Launch kernel: assumes (out, a, b, n) parameter signature
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    void* args[] = { &d_out, &d_a, &d_b, &N };
    CHECK_CU(cuLaunchKernel(func, blocks, 1, 1, threads, 1, 1, 0, 0, args, NULL));
    CHECK_CU(cuCtxSynchronize());

    // Read back results
    CHECK_CU(cuMemcpyDtoH(h_out, d_out, bytes));

    // Verify (assumes vector_add: out[i] = a[i] + b[i])
    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_out[i] - expected) > 0.001f) {
            if (errors < 5)
                printf("  MISMATCH [%d]: got %.3f, expected %.3f\n", i, h_out[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("PASS: %d elements verified correct\n", N);
    } else {
        printf("FAIL: %d/%d mismatches\n", errors, N);
    }

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_out);
    free(h_a); free(h_b); free(h_out);
    cuModuleUnload(mod);
    cuDevicePrimaryCtxRelease(dev);
    return errors > 0 ? 1 : 0;
}
