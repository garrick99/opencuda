struct Vec2 {
    float x;
    float y;
};

__global__ void dot_product(float *out, struct Vec2 *a, struct Vec2 *b, int n) {
    int i = threadIdx.x;
    if (i < n) {
        out[i] = a[i].x * b[i].x + a[i].y * b[i].y;
    }
}
