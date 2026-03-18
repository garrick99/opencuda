__global__ void mixed_math(float *out, float *a, int *b) {
    int i = threadIdx.x;
    float val = a[i] * 2 + 1.0f;
    out[i] = val;
}
