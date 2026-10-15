// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Address dependency: the second load takes its address from the result of the
// first, so the wait between them is what makes that address defined. The
// dependency lives only in the address operand of a load, which no destination
// register and no stored value covers.
__global__ void pointer_chase(int** ptrs, int* result, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int* target = ptrs[i];  // load a pointer (global_load -> VGPR pair)
        result[i]   = *target;  // address the next load with it
    }
}

int main() {
    constexpr unsigned int arraySize       = 256 * 256;
    constexpr int          threadsPerBlock = 256;
    constexpr int          numBlocks       = arraySize / threadsPerBlock;

    std::vector<int>  values(arraySize), result(arraySize);
    std::vector<int*> ptrs(arraySize);
    int *             d_values, *d_result;
    int**             d_ptrs;

    for (unsigned int i = 0; i < arraySize; ++i) {
        values[i] = static_cast<int>(i) * 7 + 3;
    }

    HIP_CHECK(hipMalloc(&d_values, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_result, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_ptrs, arraySize * sizeof(int*)));
    HIP_CHECK(hipMemcpy(d_values, values.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));

    // Each thread chases a pointer to its own element, in an order the compiler
    // cannot fold back into an index.
    for (unsigned int i = 0; i < arraySize; ++i) {
        ptrs[i] = d_values + ((i * 37) % arraySize);
    }
    HIP_CHECK(hipMemcpy(d_ptrs, ptrs.data(), arraySize * sizeof(int*), hipMemcpyHostToDevice));

    pointer_chase<<<numBlocks, threadsPerBlock>>>(d_ptrs, d_result, arraySize);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(result.data(), d_result, arraySize * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_values));
    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(d_ptrs));

    std::cout << "[";
    for (unsigned int i = 0; i < arraySize; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << result[i];
    }
    std::cout << "]" << std::endl;

    bool pass = true;
    for (unsigned int i = 0; i < arraySize; ++i) {
        int expected = values[(i * 37) % arraySize];
        if (result[i] != expected) {
            std::cerr << "Mismatch at " << i << ": expected " << expected << " got " << result[i] << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
