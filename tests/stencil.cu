// 1D stencil: out[i] = 0.25 * (in[i-1] + 2*in[i] + in[i+1])
__global__ void stencil1d(float *out, float *in, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i > 0) {
        if (i < n - 1) {
            out[i] = 0.25f * (in[i - 1] + 2.0f * in[i] + in[i + 1]);
        }
    }
}
