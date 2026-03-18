__global__ void sum_reduce(float *out, float *in, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += in[i];
    }
    out[0] = sum;
}
