__global__ void negation(float *out, float *in, int n) {
    int i = threadIdx.x;
    if (i < n) {
        float x = in[i];
        out[i] = -x;
    }
}
