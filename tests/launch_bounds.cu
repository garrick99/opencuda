__global__ __launch_bounds__(256, 4)
void bounded_kernel(float *out, float *in, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = in[i] + 1.0f;
    }
}
