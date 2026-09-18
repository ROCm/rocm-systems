// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// Global-load → transcendental hazard: loads float values from global
// memory into VGPRs, then immediately consumes them in v_sin_f32 and
// v_cos_f32 transcendental instructions.  The products sin(x)*cos(x)
// are stored to the output buffer.
//
// Hazard surface:
//   - global_load writes VGPRs → v_sin_f32 / v_cos_f32 read them
//     (requires s_wait_loadcnt before transcendental issue)
//   - v_sin_f32 / v_cos_f32 write VGPRs → v_mul_f32 reads them
//     (trans→VALU dependency)
__global__ void trans_sin_cos(const float* __restrict__ input, float* __restrict__ output, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x   = input[i];  // global_load → VGPR
        float s   = __sinf(x); // v_sin_f32
        float c   = __cosf(x); // v_cos_f32
        output[i] = s * c;     // v_mul_f32 (trans→VALU hazard)
    }
}

int main() {
    constexpr unsigned int arraySize       = 256;
    constexpr int          threadsPerBlock = 64;
    constexpr int          numBlocks       = arraySize / threadsPerBlock;

    std::vector<float> input(arraySize), output(arraySize, 0.0f);
    float *            d_in, *d_out;

    for (unsigned int i = 0; i < arraySize; ++i) {
        input[i] = static_cast<float>(i) * 0.01f;
    }

    HIP_CHECK(hipMalloc(&d_in, arraySize * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, arraySize * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_in, input.data(), arraySize * sizeof(float), hipMemcpyHostToDevice));

    trans_sin_cos<<<numBlocks, threadsPerBlock>>>(d_in, d_out, arraySize);
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

    // Validate: sin(x)*cos(x) == 0.5*sin(2x)
    bool pass = true;
    for (unsigned int i = 0; i < arraySize; ++i) {
        float x        = input[i];
        float expected = 0.5f * std::sin(2.0f * x);
        float diff     = std::fabs(output[i] - expected);
        // __sinf / __cosf are lower-precision intrinsics; allow some ULP error.
        // A non-finite result fails every tolerance comparison, so it is
        // rejected outright rather than passing by default.
        if (!std::isfinite(output[i]) || (diff > 0.01f && diff > 0.001f * std::fabs(expected))) {
            std::cerr << "Mismatch at " << i << ": expected " << expected << " got " << output[i] << " (diff " << diff
                      << ")" << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
