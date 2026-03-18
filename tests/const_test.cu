__global__ void const_math(float *out, float *in, int n) {
    const float pi = 3.14159f;
    const float scale = 2.0f;
    int i = threadIdx.x;
    if (i < n) {
        out[i] = in[i] * pi * scale;
    }
}
