// SAXPY: y = a*x + y
__global__ void saxpy(float *y, float *x, float a, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}
