// Multi-dimensional shared memory access pattern
// Uses 1D shared array with manual 2D indexing (tile[ty*16+tx])
__global__ void transpose(float *out, float *in, int N) {
    __shared__ float tile[256];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = tx + blockIdx.x * 16;
    int y = ty + blockIdx.y * 16;

    // Read from global, write to shared (coalesced read)
    tile[ty * 16 + tx] = in[y * N + x];
    __syncthreads();

    // Read from shared (transposed), write to global
    int out_x = ty + blockIdx.y * 16;
    int out_y = tx + blockIdx.x * 16;
    out[out_y * N + out_x] = tile[tx * 16 + ty];
}
