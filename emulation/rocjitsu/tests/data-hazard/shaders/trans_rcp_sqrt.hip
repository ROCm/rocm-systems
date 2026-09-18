// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Load → rcp / sqrt transcendental hazard: loads float values, then
// computes reciprocal (v_rcp_f32), square root (v_sqrt_f32), and
// reciprocal square root (v_rsq_f32) in sequence.
//
// Hazard surface:
//   - global_load → v_rcp_f32   (load→trans)
//   - global_load → v_sqrt_f32  (load→trans)
//   - v_rcp_f32 writes VGPR → v_mul_f32 reads it  (trans→VALU)
//   - v_sqrt_f32 writes VGPR → v_add_f32 reads it (trans→VALU)
//   - v_mul_f32 → v_rsq_f32  (VALU→trans)
//   - v_rsq_f32 → global_store (trans→store)
__global__ void trans_rcp_sqrt(const float* __restrict__ input, float* __restrict__ output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = input[i]; // global_load → VGPR

        float r = __frcp_rn(x);  // v_rcp_f32  (load→trans)
        float s = __fsqrt_rn(x); // v_sqrt_f32 (load→trans)

        float prod = r * s; // v_mul_f32  (trans→VALU)
        float sum  = s + r; // v_add_f32  (trans→VALU)

        // Feed VALU result back into transcendental
        float rs = __frsqrt_rn(prod + sum); // v_rsq_f32 (VALU→trans)

        output[i] = rs;
    }
}

int main() {
    constexpr unsigned int arraySize       = 256 * 256;
    constexpr int          threadsPerBlock = 256;
    constexpr int          numBlocks       = arraySize / threadsPerBlock;

    std::vector<float> input(arraySize), output(arraySize, 0.0f);
    float *            d_in, *d_out;

    // Positive values > 0 to make rcp/sqrt well-defined
    for (unsigned int i = 0; i < arraySize; ++i) {
        input[i] = static_cast<float>(i + 1) * 0.01f;
    }

    HIP_CHECK(hipMalloc(&d_in, arraySize * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, arraySize * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, input.data(), arraySize * sizeof(float), hipMemcpyHostToDevice));

    trans_rcp_sqrt<<<numBlocks, threadsPerBlock>>>(d_in, d_out, arraySize);
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

    // Validate: rsqrt(rcp(x)*sqrt(x) + sqrt(x)+rcp(x))
    //         = rsqrt(1/sqrt(x) + sqrt(x) + 1/x)
    bool pass = true;
    for (unsigned int i = 0; i < arraySize; ++i) {
        float x        = input[i];
        float r        = 1.0f / x;
        float s        = std::sqrt(x);
        float prod     = r * s;
        float sum      = s + r;
        float expected = 1.0f / std::sqrt(prod + sum);
        float diff     = std::fabs(output[i] - expected);
        if (!std::isfinite(output[i]) || (diff > 0.01f && diff > 0.005f * std::fabs(expected))) {
            std::cerr << "Mismatch at " << i << ": expected " << expected << " got " << output[i] << " (diff " << diff
                      << ")" << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
