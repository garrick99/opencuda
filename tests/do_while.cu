__global__ void count_down(int* result, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int count = 0;
    int i = n;
    do {
        count = count + 1;
        i = i - 1;
    } while (i > 0);
    result[tid] = count;
}
