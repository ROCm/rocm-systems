// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Atomic histogram kernel.
// Exercises global_atomic_add instructions and associated wait counters.
__global__ void atomic_histogram(const int* input, int* histogram, unsigned int n, int num_bins) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int bin = input[i] % num_bins;
        if (bin < 0)
            bin += num_bins;
        atomicAdd(&histogram[bin], 1);
    }
}

int main() {
    constexpr unsigned int arraySize       = 256 * 256;
    constexpr int          numBins         = 16;
    constexpr int          threadsPerBlock = 256;
    constexpr int          numBlocks       = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<int> input(arraySize);
    std::vector<int> histogram(numBins, 0);
    int *            d_in, *d_hist;

    for (unsigned int i = 0; i < arraySize; ++i) {
        input[i] = static_cast<int>(i);
    }

    HIP_CHECK(hipMalloc(&d_in, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_hist, numBins * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_in, input.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_hist, 0, numBins * sizeof(int)));

    atomic_histogram<<<numBlocks, threadsPerBlock>>>(d_in, d_hist, arraySize, numBins);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(histogram.data(), d_hist, numBins * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_hist));

    std::cout << "[";
    for (int i = 0; i < numBins; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << histogram[i];
    }
    std::cout << "]" << std::endl;

    int  expected = arraySize / numBins;
    bool pass     = true;
    for (int i = 0; i < numBins; ++i) {
        if (histogram[i] != expected) {
            std::cerr << "Bin " << i << ": expected " << expected << " got " << histogram[i] << std::endl;
            pass = false;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
