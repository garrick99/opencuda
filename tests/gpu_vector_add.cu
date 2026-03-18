/**
 * Test harness for vector_add compiled by OpenCUDA.
 */
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *name = nullptr, *str = nullptr; \
        cuGetErrorName(err, &name); \
        cuGetErrorString(err, &str); \
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", \
                __FILE__, __LINE__, name ? name : "?", str ? str : "?"); \
        exit(1); \
    } \
} while(0)

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <cubin>\n", argv[0]);
        return 1;
    }

    CHECK_CU(cuInit(0));
    CUdevice dev;
    CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx;
    CUctxCreateParams p = {};
    CHECK_CU(cuCtxCreate(&ctx, &p, 0, dev));

    CUmodule mod;
    CHECK_CU(cuModuleLoad(&mod, argv[1]));
    CUfunction func;
    CHECK_CU(cuModuleGetFunction(&func, mod, "vector_add"));

    const int N = 256;
    float h_a[N], h_b[N], h_out[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
        h_out[i] = 0.0f;
    }

    CUdeviceptr d_a, d_b, d_out;
    CHECK_CU(cuMemAlloc(&d_a, N * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_b, N * sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_out, N * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_a, h_a, N * sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_b, h_b, N * sizeof(float)));

    int n = N;
    void *args[] = { &d_out, &d_a, &d_b, &n };
    CHECK_CU(cuLaunchKernel(func, 1,1,1, N,1,1, 0,0, args, nullptr));
    CHECK_CU(cuCtxSynchronize());

    CHECK_CU(cuMemcpyDtoH(h_out, d_out, N * sizeof(float)));

    int errors = 0;
    for (int i = 0; i < N; i++) {
        float expected = h_a[i] + h_b[i];
        if (fabsf(h_out[i] - expected) > 0.001f) {
            if (errors < 5)
                printf("  MISMATCH at [%d]: got %.1f expected %.1f\n", i, h_out[i], expected);
            errors++;
        }
    }

    printf("Vector Add Test (N=%d)\n", N);
    printf("  Cubin: %s\n", argv[1]);
    if (errors == 0) {
        printf("  *** PASS — all %d elements correct! ***\n", N);
        // Print a few samples
        for (int i = 0; i < 4; i++)
            printf("  [%d]: %.1f + %.1f = %.1f\n", i, h_a[i], h_b[i], h_out[i]);
    } else {
        printf("  *** FAIL — %d/%d mismatches ***\n", errors, N);
    }

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_out);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
    return errors ? 1 : 0;
}
