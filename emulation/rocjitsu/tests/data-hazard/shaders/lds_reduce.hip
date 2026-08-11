// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Parallel reduction using LDS (shared memory).
// Exercises ds_read/ds_write instructions and s_wait_dscnt waits.
__global__ void lds_reduce(const int* input, int* output, unsigned int n) {
    __shared__ int sdata[256];
    unsigned int   tid = threadIdx.x;
    unsigned int   i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    constexpr unsigned int arraySize       = 256 * 256;
    constexpr int          threadsPerBlock = 256;
    constexpr int          numBlocks       = arraySize / threadsPerBlock;

    std::vector<int> input(arraySize, 1);
    std::vector<int> output(numBlocks, 0);
    int *            d_in, *d_out;

    HIP_CHECK(hipMalloc(&d_in, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, numBlocks * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_in, input.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));

    lds_reduce<<<numBlocks, threadsPerBlock>>>(d_in, d_out, arraySize);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output.data(), d_out, numBlocks * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    std::cout << "[";
    for (int i = 0; i < numBlocks; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << output[i];
    }
    std::cout << "]" << std::endl;

    bool pass = true;
    for (int i = 0; i < numBlocks; ++i) {
        if (output[i] != threadsPerBlock) {
            std::cerr << "Mismatch at block " << i << ": expected " << threadsPerBlock << " got " << output[i]
                      << std::endl;
            pass = false;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
