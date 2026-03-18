__global__ void smem_test(float *out, float *in) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    smem[tid] = in[tid];
    __syncthreads();
    out[tid] = smem[tid];
}
