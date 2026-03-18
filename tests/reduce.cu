// Block-level parallel sum reduction using shared memory
__global__ void block_reduce(float *out, float *in, int n) {
    __shared__ float smem[256];
    int tid = threadIdx.x;
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    // Load to shared memory
    if (i < n) {
        smem[tid] = in[i];
    }
    __syncthreads();

    // Reduction in shared memory
    int stride = 128;
    while (stride > 0) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
        stride = stride / 2;
    }

    // Thread 0 writes result
    if (tid < 1) {
        out[blockIdx.x] = smem[0];
    }
}
