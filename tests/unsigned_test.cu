// Test unsigned arithmetic — critical for field math
__global__ void uint_math(unsigned int *out, unsigned int *a, unsigned int *b, unsigned int n) {
    unsigned int i = threadIdx.x;
    if (i < n) {
        unsigned int x = a[i];
        unsigned int y = b[i];
        // These must use u32 operations, not s32
        unsigned int sum = x + y;         // add.u32
        unsigned int diff = x - y;        // sub.u32 (wrapping)
        unsigned int prod = x * y;        // mul.lo.u32
        unsigned int shifted = x >> 16;   // shr.b32 (logical, not arithmetic)
        unsigned int masked = x & 0xFF;   // and.b32
        out[i] = sum + diff + prod + shifted + masked;
    }
}
