__global__ void bitops(unsigned int *out, unsigned int *a, unsigned int *b, int n) {
    int i = threadIdx.x;
    if (i < n) {
        unsigned int x = a[i];
        unsigned int y = b[i];
        unsigned int and_r = x & y;
        unsigned int or_r = x | y;
        unsigned int xor_r = x ^ y;
        unsigned int not_r = ~x;
        unsigned int shl_r = x << 4;
        unsigned int shr_r = x >> 8;
        out[i] = and_r + or_r + xor_r + not_r + shl_r + shr_r;
    }
}
