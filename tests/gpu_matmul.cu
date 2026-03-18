/**
 * GPU test for OpenCUDA-compiled naive matrix multiply.
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
    CUfunction func; CHECK_CU(cuModuleGetFunction(&func, mod, "matmul"));

    const int N = 4;
    float h_A[16], h_B[16], h_C[16];

    // A = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    // B = identity
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            h_A[i*N+j] = (float)(i*N+j+1);
            h_B[i*N+j] = (i == j) ? 1.0f : 0.0f;
            h_C[i*N+j] = 0.0f;
        }

    CUdeviceptr d_A, d_B, d_C;
    CHECK_CU(cuMemAlloc(&d_A, N*N*sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_B, N*N*sizeof(float)));
    CHECK_CU(cuMemAlloc(&d_C, N*N*sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_A, h_A, N*N*sizeof(float)));
    CHECK_CU(cuMemcpyHtoD(d_B, h_B, N*N*sizeof(float)));

    int n = N;
    void *args[] = { &d_C, &d_A, &d_B, &n };
    // Launch with N x N threads
    CHECK_CU(cuLaunchKernel(func, 1,1,1, N,N,1, 0,0, args, nullptr));
    CHECK_CU(cuCtxSynchronize());
    CHECK_CU(cuMemcpyDtoH(h_C, d_C, N*N*sizeof(float)));

    // A * I = A, so C should equal A
    int errors = 0;
    for (int i = 0; i < N*N; i++) {
        if (fabsf(h_C[i] - h_A[i]) > 0.001f) {
            printf("  MISMATCH [%d]: C=%.1f expected=%.1f\n", i, h_C[i], h_A[i]);
            errors++;
        }
    }

    printf("Matrix Multiply Test (%dx%d)\n", N, N);
    printf("  Cubin: %s\n", argv[1]);
    if (errors == 0) {
        printf("  *** PASS — A*I = A verified! ***\n");
        printf("  C[0..3] = [%.0f, %.0f, %.0f, %.0f]\n", h_C[0], h_C[1], h_C[2], h_C[3]);
        printf("  C[4..7] = [%.0f, %.0f, %.0f, %.0f]\n", h_C[4], h_C[5], h_C[6], h_C[7]);
    } else {
        printf("  *** FAIL — %d mismatches ***\n", errors);
    }

    cuMemFree(d_A); cuMemFree(d_B); cuMemFree(d_C);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    return errors ? 1 : 0;
}
