// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// LDS → transcendental hazard: threads collaboratively load data into
// shared memory (LDS), then each thread reads its LDS slot and
// immediately feeds the value into v_exp_f32.
//
// Hazard surface:
//   - ds_write (store to LDS) → barrier → ds_read (load from LDS)
//   - ds_read writes VGPR → v_exp_f32 reads it  (ds→trans hazard,
//     requires s_wait_dscnt before transcendental issue)
//   - v_exp_f32 writes VGPR → v_add_f32 reads it  (trans→VALU)
__global__ void trans_lds(const float* __restrict__ input, float* __restrict__ output, unsigned int n) {
    __shared__ float sdata[256];
    unsigned int     tid = threadIdx.x;
    unsigned int     i   = blockIdx.x * blockDim.x + tid;

    // Load global → LDS
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Read from LDS and apply transcendental
    float val = sdata[tid];  // ds_read → VGPR
    float e   = __expf(val); // v_exp_f32 (ds→trans hazard)

    // Also read the neighbor (with wrap) to create a second ds→trans dep
    float neighbor = sdata[(tid + 1) % blockDim.x];
    float e2       = __logf(neighbor + 1.0f); // v_log_f32 (ds→VALU→trans)

    if (i < n) {
        output[i] = e + e2;
    }
}

int main() {
    constexpr unsigned int arraySize       = 256 * 256;
    constexpr int          threadsPerBlock = 256;
    constexpr int          numBlocks       = arraySize / threadsPerBlock;

    std::vector<float> input(arraySize), output(arraySize, 0.0f);
    float *            d_in, *d_out;

    // Small values so exp() stays in range
    for (unsigned int i = 0; i < arraySize; ++i) {
        input[i] = static_cast<float>(i % 10) * 0.1f;
    }

    HIP_CHECK(hipMalloc(&d_in, arraySize * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, arraySize * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, input.data(), arraySize * sizeof(float), hipMemcpyHostToDevice));

    trans_lds<<<numBlocks, threadsPerBlock>>>(d_in, d_out, arraySize);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output.data(), d_out, arraySize * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));

    std::cout << "[";
    for (unsigned int i = 0; i < arraySize; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << output[i];
    }
    std::cout << "]" << std::endl;

    // Validate: output[i] = exp(input[i]) + log(input[(i+1)%block] + 1)
    bool pass = true;
    for (unsigned int i = 0; i < arraySize; ++i) {
        unsigned int tid          = i % threadsPerBlock;
        unsigned int block_start  = (i / threadsPerBlock) * threadsPerBlock;
        unsigned int neighbor_idx = block_start + ((tid + 1) % threadsPerBlock);

        float expected_e = std::exp(input[i]);
        float expected_l = std::log(input[neighbor_idx] + 1.0f);
        float expected   = expected_e + expected_l;
        float diff       = std::fabs(output[i] - expected);
        // A non-finite result fails every tolerance comparison, so it is
        // rejected outright rather than passing by default.
        if (!std::isfinite(output[i]) || (diff > 0.05f && diff > 0.01f * std::fabs(expected))) {
            std::cerr << "Mismatch at " << i << ": expected " << expected << " got " << output[i] << " (diff " << diff
                      << ")" << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
