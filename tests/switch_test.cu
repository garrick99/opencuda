__global__ void classify(int* input, int* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int val = input[tid];
    int result = 0;
    switch (val) {
        case 0:
            result = 10;
            break;
        case 1:
            result = 20;
            break;
        case 2:
            result = 30;
            break;
        default:
            result = -1;
            break;
    }
    output[tid] = result;
}
