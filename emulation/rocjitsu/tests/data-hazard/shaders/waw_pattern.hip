// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// WAW (Write-After-Write) pattern: two separate stores to the same output
// address. The second store depends on a loaded value, so removing
// s_wait_loadcnt could cause the second store to use an uninitialized/stale
// value (data hazard).
__global__ void waw_pattern(const int* a, const int* b, int* output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // First write: store a[i]
        output[i] = a[i];

        // Second write: overwrite with a[i] + b[i]
        // Both writes target the same address; this exercises WAW ordering.
        int va    = a[i];
        int vb    = b[i];
        output[i] = va + vb;
    }
}

int main() {
    constexpr unsigned int arraySize       = 256 * 256;
    constexpr int          threadsPerBlock = 256;
    constexpr int          numBlocks       = arraySize / threadsPerBlock;

    std::vector<int> a(arraySize), b(arraySize), output(arraySize, 0);
    int *            d_a, *d_b, *d_out;

    for (unsigned int i = 0; i < arraySize; ++i) {
        a[i] = static_cast<int>(i);
        b[i] = static_cast<int>(i * 2);
    }

    HIP_CHECK(hipMalloc(&d_a, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_b, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, arraySize * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_a, a.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, b.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));

    waw_pattern<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_out, arraySize);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output.data(), d_out, arraySize * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_out));

    std::cout << "[";
    for (unsigned int i = 0; i < arraySize; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << output[i];
    }
    std::cout << "]" << std::endl;

    bool pass = true;
    for (unsigned int i = 0; i < arraySize; ++i) {
        int expected = static_cast<int>(i) + static_cast<int>(i * 2);
        if (output[i] != expected) {
            std::cerr << "Mismatch at " << i << ": expected " << expected << " got " << output[i] << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
