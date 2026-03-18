__global__ void cast_ops(float *fout, int *iout, float *fin, int *iin, int n) {
    int i = threadIdx.x;
    if (i < n) {
        float f = fin[i];
        int x = iin[i];
        // Cast operations
        iout[i] = (int)f;
        fout[i] = (float)x;
    }
}
