// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Scratch (private) memory test: uses a local array large enough to force
// spilling to scratch memory. Exercises scratch_load/scratch_store instructions
// and associated wait counters.
__global__ void scratch_access(const int* input, int* output, unsigned int n) {
    int          local_buf[64];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n) {
        // Fill local array from global memory (forces scratch writes)
        for (int j = 0; j < 64; ++j) {
            local_buf[j] = input[tid] + j;
        }

        // Read back from scratch and accumulate
        int sum = 0;
        for (int j = 0; j < 64; ++j) {
            sum += local_buf[j];
        }

        output[tid] = sum;
    }
}

int main() {
    constexpr unsigned int arraySize       = 256 * 256;
    constexpr int          threadsPerBlock = 256;
    constexpr int          numBlocks       = arraySize / threadsPerBlock;

    std::vector<int> input(arraySize), output(arraySize, 0);
    int *            d_in, *d_out;

    for (unsigned int i = 0; i < arraySize; ++i) {
        input[i] = static_cast<int>(i % 100);
    }

    HIP_CHECK(hipMalloc(&d_in, arraySize * sizeof(int)));
    HIP_CHECK(hipMalloc(&d_out, arraySize * sizeof(int)));
    HIP_CHECK(hipMemcpy(d_in, input.data(), arraySize * sizeof(int), hipMemcpyHostToDevice));

    scratch_access<<<numBlocks, threadsPerBlock>>>(d_in, d_out, arraySize);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output.data(), d_out, arraySize * sizeof(int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    std::cout << "[";
    for (unsigned int i = 0; i < arraySize; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << output[i];
    }
    std::cout << "]" << std::endl;

    // Expected: sum of (input[i] + j) for j in 0..63 = 64*input[i] + 63*64/2 =
    // 64*input[i] + 2016
    bool pass = true;
    for (unsigned int i = 0; i < arraySize; ++i) {
        int expected = 64 * input[i] + 2016;
        if (output[i] != expected) {
            std::cerr << "Mismatch at " << i << ": expected " << expected << " got " << output[i] << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
