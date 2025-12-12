/**
 * Multi-wrapper test kernel 1.
 *
 * This file is compiled with -fgpu-rdc (relocatable device code) and linked
 * with multi_wrapper_kernel2.hip to create a shared library with MULTIPLE
 * __CudaFatBinaryWrapper structures in .hipFatBinSegment.
 *
 * This tests the kpack splitter's ability to handle binaries built with
 * relocatable device code, which is common in large projects like RCCL.
 */

#include <hip/hip_runtime.h>

__global__ void kernel1(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

extern "C" void launch_kernel1(float* data, int n) {
    kernel1<<<(n + 255) / 256, 256>>>(data, n);
}
