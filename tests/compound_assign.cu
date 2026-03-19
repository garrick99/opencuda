__global__ void compound_ops(int* data, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int val = data[tid];
    val += 10;
    val -= 3;
    val *= 2;
    val &= 0xFF;
    val |= 0x100;
    val ^= 0x55;
    val <<= 2;
    val >>= 1;
    data[tid] = val;
}
