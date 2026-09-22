#include "vector_lib.h"
#include <hip/hip_runtime.h>
#include <stdlib.h>

// HIP kernel for vector addition
__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" int vector_add_and_sum(int n, float* result_sum) {
    if (n <= 0 || result_sum == nullptr) {
        return -1;
    }

    const size_t size = n * sizeof(float);

    // Allocate host memory
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);
    float* h_c = (float*)malloc(size);

    if (!h_a || !h_b || !h_c) {
        free(h_a);
        free(h_b);
        free(h_c);
        return -1;
    }

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    if (hipMalloc(&d_a, size) != hipSuccess ||
        hipMalloc(&d_b, size) != hipSuccess ||
        hipMalloc(&d_c, size) != hipSuccess) {
        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_c);
        free(h_a);
        free(h_b);
        free(h_c);
        return -1;
    }

    // Copy input data to device
    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    hipLaunchKernelGGL(vectorAdd, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0,
                       d_a, d_b, d_c, n);

    // Copy result back to host
    hipMemcpy(h_c, d_c, size, hipMemcpyDeviceToHost);

    // Verify and sum results
    float sum = 0.0f;
    bool success = true;
    for (int i = 0; i < n; i++) {
        float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected) {
            success = false;
            break;
        }
        sum += h_c[i];
    }

    // Cleanup
    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    if (success) {
        *result_sum = sum;
        return 0;
    } else {
        return -1;
    }
}
