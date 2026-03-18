// Local array in registers
__global__ void local_arr(float *out, float *in, int n) {
    int i = threadIdx.x;
    if (i < n) {
        float val = in[i];
        // Use local variable (scalar, not array — arrays need .local memory)
        float doubled = val * 2.0f;
        float halved = val * 0.5f;
        out[i] = doubled + halved;
    }
}
