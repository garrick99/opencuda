__global__ void countdown(int *out, int n) {
    int count = 0;
    while (n > 0) {
        count += n;
        n = n - 1;
    }
    out[0] = count;
}
