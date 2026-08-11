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
__global__ void add(const int* a, const int* b, int* c, std::size_t size) {
    const std::size_t index = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    constexpr int         numOfBlocks     = 4;
    constexpr int         threadsPerBlock = 64;
    constexpr std::size_t arraySize       = numOfBlocks * threadsPerBlock;

    std::vector<int> a(arraySize), b(arraySize), c(arraySize);
    int *            d_a, *d_b, *d_c;

    // Setup input values.
    std::iota(a.begin(), a.end(), 0);
    std::iota(b.begin(), b.end(), arraySize);

    // Allocate device copies of a, b and c.
    HIP_CHECK(hipMalloc(&d_a, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_c, arraySize * sizeof(int)));

    // Copy input values to device.
    HIP_CHECK(hipMemcpy(d_a, a.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, b.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));

    // Launch add() kernel on GPU.
    add<<<numOfBlocks, threadsPerBlock>>>(d_a, d_b, d_c, arraySize);
    // Check the kernel launch
    HIP_CHECK(hipGetLastError());
    // Check for kernel execution error
    HIP_CHECK(hipDeviceSynchronize());

    // Copy the result back to the host.
    HIP_CHECK(hipMemcpy(c.data(), d_c, arraySize * sizeof(int), hipMemcpyDeviceToHost));

    // Cleanup allocated memory.
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    // Print full result buffer as JSON array.
    std::cout << "[";
    for (std::size_t i = 0; i < arraySize; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << c[i];
    }
    std::cout << "]" << std::endl;

    // Validate: each element should be a[i] + b[i]
    bool pass = true;
    for (std::size_t i = 0; i < arraySize; ++i) {
        if (c[i] != a[i] + b[i]) {
            std::cerr << "Mismatch at " << i << ": expected " << (a[i] + b[i]) << " got " << c[i] << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
