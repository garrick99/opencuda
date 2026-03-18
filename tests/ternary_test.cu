__global__ void clamp_kernel(float *out, float *in, int n) {
    int i = threadIdx.x;
    if (i < n) {
        float val = in[i];
        // Ternary: clamp to [0, 1]
        float clamped = (val < 0.0f) ? 0.0f : val;
        out[i] = (clamped > 1.0f) ? 1.0f : clamped;
    }
}
