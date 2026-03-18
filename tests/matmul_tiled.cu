// Tiled matrix multiply using shared memory
// C = A * B, all NxN matrices
// TILE_SIZE = 16, each block computes a 16x16 tile of C

__global__ void matmul_tiled(float *C, float *A, float *B, int N) {
    __shared__ float As[256];
    __shared__ float Bs[256];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = ty + blockIdx.y * 16;
    int col = tx + blockIdx.x * 16;

    float sum = 0.0f;

    int numTiles = N / 16;
    for (int t = 0; t < numTiles; t++) {
        // Load tile of A into shared memory
        As[ty * 16 + tx] = A[row * N + t * 16 + tx];

        // Load tile of B into shared memory
        Bs[ty * 16 + tx] = B[(t * 16 + ty) * N + col];

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < 16; k++) {
            sum += As[ty * 16 + k] * Bs[k * 16 + tx];
        }

        __syncthreads();
    }

    C[row * N + col] = sum;
}
