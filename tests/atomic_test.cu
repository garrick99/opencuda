__global__ void atomic_sum(int *out, int *in, int n) {
    int i = threadIdx.x;
    if (i < n) {
        atomicAdd(out, in[i]);
    }
}
