__device__ float square(float x) {
    return x * x;
}

__device__ int add_three(int a, int b, int c) {
    int sum = a + b + c;
    return sum;
}

__global__ void use_device_funcs(float* fout, int* iout, float* fin, int* iin, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    fout[tid] = square(fin[tid]);
    iout[tid] = add_three(iin[tid], 10, 20);
}
