__global__ void kernel_a(float *out, float *in, int n) {
    int i = threadIdx.x;
    if (i < n) {
        out[i] = in[i] * 2.0f;
    }
}

__global__ void kernel_b(float *out, float *a, float *b, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

__global__ void kernel_c(int *out, int n) {
    int sum = 0;
    for (int i = 0; i < n; i++) {
        sum += i;
    }
    out[0] = sum;
}
