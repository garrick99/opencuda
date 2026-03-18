// Warp-level reduction using shuffle
__global__ void warp_reduce(int *out, int *in, int n) {
    int i = threadIdx.x;
    int val = 0;
    if (i < n) {
        val = in[i];
    }

    // Warp-level reduction via shuffle
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);

    if (i < 1) {
        out[0] = val;
    }
}
