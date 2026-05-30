// Minimal HIP vector-add kernel — fixture used by ml_scenarios.rs
// (only compiled when a HIP toolchain is present on the host).
//
//   hipcc tests/fixtures/ml/tiny_hip.hip -o tiny_hip && ./tiny_hip
//
// Prints `hip_kernel_ok` on success.

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 16;
    float h_a[n], h_b[n], h_c[n];
    for (int i = 0; i < n; ++i) { h_a[i] = (float)i; h_b[i] = 2.0f * i; }
    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, n * sizeof(float));
    hipMalloc(&d_b, n * sizeof(float));
    hipMalloc(&d_c, n * sizeof(float));
    hipMemcpy(d_a, h_a, n * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, n * sizeof(float), hipMemcpyHostToDevice);
    hipLaunchKernelGGL(add, dim3(1), dim3(n), 0, 0, d_a, d_b, d_c, n);
    hipDeviceSynchronize();
    hipMemcpy(h_c, d_c, n * sizeof(float), hipMemcpyDeviceToHost);
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != 3.0f * i) { fprintf(stderr, "mismatch at %d: %f vs %f\n", i, h_c[i], 3.0f * i); ++errors; }
    }
    hipFree(d_a); hipFree(d_b); hipFree(d_c);
    hipDeviceReset();
    if (errors) {
        // _exit avoids HIP/HSA atexit handlers that hang under the simulator
        // (see rocjitsu tests/hip_vector_add_test.cpp main()).
        _exit(1);
    }
    printf("hip_kernel_ok\n");
    fflush(stdout);
    _exit(0);
}
