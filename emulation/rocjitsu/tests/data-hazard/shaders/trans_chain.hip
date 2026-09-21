// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Transcendental → transcendental chain hazard: loads values, applies
// v_exp_f32, then immediately feeds the result into v_log_f32.
// Since log(exp(x)) ≈ x for small x, the output should approximate the
// input, providing a strong correctness oracle.
//
// Hazard surface:
//   - global_load → v_exp_f32  (load→trans)
//   - v_exp_f32 writes VGPR → v_log_f32 reads it  (trans→trans dependency)
//   - v_log_f32 writes VGPR → global_store reads it  (trans→store)
__global__ void trans_chain(const float* __restrict__ input, float* __restrict__ output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x   = input[i];  // global_load → VGPR
        float e   = __expf(x); // v_exp_f32 (load→trans hazard)
        float l   = __logf(e); // v_log_f32 (trans→trans hazard)
        output[i] = l;         // global_store
    }
}

int main() {
    constexpr unsigned int arraySize       = 256 * 256;
    constexpr int          threadsPerBlock = 256;
    constexpr int          numBlocks       = arraySize / threadsPerBlock;

    std::vector<float> input(arraySize), output(arraySize, 0.0f);
    float *            d_in, *d_out;

    // Use small positive values so exp() stays in range and log(exp(x)) ≈ x
    for (unsigned int i = 0; i < arraySize; ++i) {
        input[i] = static_cast<float>(i % 64) * 0.1f;
    }

    HIP_CHECK(hipMalloc(&d_in, arraySize * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, arraySize * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, input.data(), arraySize * sizeof(float), hipMemcpyHostToDevice));

    trans_chain<<<numBlocks, threadsPerBlock>>>(d_in, d_out, arraySize);
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

    // Validate: log(exp(x)) ≈ x within device precision
    bool pass = true;
    for (unsigned int i = 0; i < arraySize; ++i) {
        float expected = input[i];
        float diff     = std::fabs(output[i] - expected);
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
