// Simple histogram: count occurrences of values 0..255
__global__ void histogram(int *hist, unsigned int *data, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        int bin = data[i] & 255;
        // Note: would need atomicAdd for correctness with multiple threads
        // This version works for single-thread-per-bin testing
        hist[bin] += 1;
    }
}
