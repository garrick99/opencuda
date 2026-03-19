__global__ void find_first(int* data, int* result, int n, int target) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    int found = -1;
    for (int i = 0; i < n; i++) {
        if (data[i] == target) {
            found = i;
            break;
        }
    }
    result[tid] = found;
}

__global__ void skip_negatives(float* out, float* in, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        if (in[i] < 0.0f) continue;
        sum = sum + in[i];
    }
    out[tid] = sum;
}
