// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Same-thread LDS read/write ordering pattern.
//
// Each thread reads its slot from shared memory (ds_read, DScnt increments),
// then writes a new value back to the same slot. Ordinary DS operations from
// one wave complete in issue order, and no other thread touches that slot, so
// this is not an address race.
//
// The shader is retained to verify that removing waits which merely complete a
// store before a barrier does not create a false positive.
__global__ void lds_war_pattern(const int* input, int* output, unsigned int n) {
    __shared__ int sdata[256];
    unsigned int   tid = threadIdx.x;
    unsigned int   i   = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Same-wave DS issue order makes the read complete before this write.
    int val    = sdata[tid];       // ds_read  — DScnt increments
    sdata[tid] = val + 1;          // ds_write — same private-per-thread slot
    __syncthreads();

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

    lds_war_pattern<<<numBlocks, threadsPerBlock>>>(d_in, d_out, arraySize);
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

    // Each thread increments its own value from 1 to 2; thread 0 publishes its slot.
    bool pass = true;
    for (int i = 0; i < numBlocks; ++i) {
        if (output[i] != 2) {
            std::cerr << "Mismatch at block " << i << ": expected 2 got " << output[i] << std::endl;
            pass = false;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
