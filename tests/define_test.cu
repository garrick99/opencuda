#define BLOCK_SIZE 256
#define SCALE 2.0f

__global__ void scaled_copy(float *out, float *in, int n) {
    int i = threadIdx.x + blockIdx.x * BLOCK_SIZE;
    if (i < n) {
        out[i] = in[i] * SCALE;
    }
}
