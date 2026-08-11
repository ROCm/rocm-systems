// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include "utils.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

// Addition of two values.
__global__ void double_unsafe(const int* x, int* y, const std::size_t size) {
    const std::size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    y[index]                = x[index] * 2;
}

int main() {
    constexpr int         numOfBlocks     = 4;
    constexpr int         threadsPerBlock = 64;
    constexpr std::size_t arraySize       = numOfBlocks * threadsPerBlock;

    std::vector<int> x(arraySize), y(arraySize);
    int *            d_y, *d_x;

    // Setup input values.
    std::iota(x.begin(), x.end(), 0);

    // Allocate device copies of x, y and c.
    HIP_CHECK(hipMalloc(&d_y, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_x, arraySize * sizeof(int)));

    // Copy input values to device.
    HIP_CHECK(hipMemcpy(d_x, x.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));

    // Launch double_unsafe() kernel on GPU.
    double_unsafe<<<numOfBlocks, threadsPerBlock>>>(d_x, d_y, arraySize);
    // Check the kernel launch
    HIP_CHECK(hipGetLastError());
    // Check for kernel execution error
    HIP_CHECK(hipDeviceSynchronize());

    // Copy the result back to the host.
    HIP_CHECK(hipMemcpy(y.data(), d_y, arraySize * sizeof(int), hipMemcpyDeviceToHost));

    // Cleanup allocated memory.
    HIP_CHECK(hipFree(d_y));
    HIP_CHECK(hipFree(d_x));

    std::cout << "[";
    for (std::size_t i = 0; i < arraySize; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << y[i];
    }
    std::cout << "]" << std::endl;

    // Validate: each element should be x[i] * 2 = 1 * 2 = 2
    bool pass = true;
    for (std::size_t i = 0; i < arraySize; ++i) {
        if (y[i] != 2 * x[i]) {
            std::cerr << "Mismatch at " << i << ": expected " << 2 * x[i] << " got " << y[i] << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
