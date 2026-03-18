__device__ float square(float x) {
    return x * x;
}

__global__ void apply_square(float *out, float *in, int n) {
    int i = threadIdx.x;
    if (i < n) {
        out[i] = square(in[i]);
    }
}
