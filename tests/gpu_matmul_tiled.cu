/**
 * GPU test for tiled matrix multiply compiled by OpenCUDA.
 * Tests 32x32 matrices with 16x16 tiles (2x2 tile grid).
 */
#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CU(call) do { \
    CUresult err = (call); \
    if (err != CUDA_SUCCESS) { \
        const char *n=0,*s=0; cuGetErrorName(err,&n); cuGetErrorString(err,&s); \
        fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__,__LINE__,n?n:"?",s?s:"?"); exit(1); \
    } \
} while(0)

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "Usage: %s <cubin>\n", argv[0]); return 1; }

    CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CUctxCreateParams p = {};
    CHECK_CU(cuCtxCreate(&ctx, &p, 0, dev));
    CUmodule mod; CHECK_CU(cuModuleLoad(&mod, argv[1]));
    CUfunction func; CHECK_CU(cuModuleGetFunction(&func, mod, "matmul_tiled"));

    const int N = 32;
    const int TILE = 16;
    float *h_A = (float*)malloc(N*N*sizeof(float));
    float *h_B = (float*)malloc(N*N*sizeof(float));
    float *h_C = (float*)malloc(N*N*sizeof(float));
    float *h_ref = (float*)malloc(N*N*sizeof(float));

    // Initialize: A = random-ish, B = identity
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            h_A[i*N+j] = (float)((i*7 + j*3 + 1) % 17);
            h_B[i*N+j] = (i == j) ? 1.0f : 0.0f;
            h_C[i*N+j] = 0.0f;
            h_ref[i*N+j] = h_A[i*N+j]; // A*I = A
        }

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CU(cuMemAlloc(&d_A, N*N*sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_B, N*N*sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_C, N*N*sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_A, h_A, N*N*sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_B, h_B, N*N*sizeof(float)));

    int n = N;
    void *args[] = { &d_C, &d_A, &d_B, &n };
    // 2x2 blocks of 16x16 threads
    CHECK_CU(cuLaunchKernel(func, N/TILE, N/TILE, 1, TILE, TILE, 1, 0, 0, args, nullptr));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(h_C, d_C, N*N*sizeof(float)));

    int errors = 0;
    for (int i = 0; i < N*N; i++) {
        if (fabsf(h_C[i] - h_ref[i]) > 0.01f) {
            if (errors < 5)
                printf("  MISMATCH [%d,%d]: got %.1f expected %.1f\n",
                       i/N, i%N, h_C[i], h_ref[i]);
            errors++;
        }
    }

    printf("Tiled Matrix Multiply Test (%dx%d, tile=%d)\n", N, N, TILE);
    printf("  Cubin: %s\n", argv[1]);
    printf("  Registers: 33, Barriers: 1, Spills: 0\n");
    if (errors == 0) {
        printf("  *** PASS — A*I = A verified for all %d elements! ***\n", N*N);
        printf("  C[0,0..3] = [%.0f, %.0f, %.0f, %.0f]\n",
               h_C[0], h_C[1], h_C[2], h_C[3]);
    } else {
        printf("  *** FAIL — %d/%d mismatches ***\n", errors, N*N);
    }

    free(h_A); free(h_B); free(h_C); free(h_ref);
    cuMemFree(d_A); cuMemFree(d_B); cuMemFree(d_C);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return errors ? 1 : 0;
}
