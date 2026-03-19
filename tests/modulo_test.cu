__global__ void mod_test(int* out, int* in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    int val = in[tid];
    out[tid] = val % 7;
}
