// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// WMMA (Wave Matrix Multiply-Accumulate) followed by transcendental (exp).
// This exercises v_wmma instructions writing VGPR results that are then
// consumed by v_exp_f32 (transcendental unit), creating a potential
// data hazard if the proper waits/delays are missing between the WMMA
// writeback and the transcendental read.
//
// Each wave loads input tile fragments from global memory into VGPRs,
// performs a 16x16x32 f16 WMMA, then immediately feeds the accumulator
// VGPRs into v_exp_f32 transcendentals and stores the results.
//
// Uses the gfx1250-native __builtin_amdgcn_wmma_f32_16x16x32_f16 which
// does not require extra target-feature flags and avoids an ICE in
// AMD clang 23.0 that affects the older 16x16x16 variants on gfx1250. The
// builtin needs the "gfx1250-insts" target feature, so the harness is told to
// skip this kernel everywhere else:
// requires: gfx1250

using half16 = _Float16 __attribute__((__vector_size__(16 * sizeof(_Float16))));
using float8 = float __attribute__((__vector_size__(8 * sizeof(float))));

__global__ void wmma_exp(const _Float16* __restrict__ A,
                         const _Float16* __restrict__ B,
                         float* __restrict__ C,
                         unsigned int N) {
    const unsigned int lane = threadIdx.x; // 0..31 within wave

    // Each lane loads 16 halfs from A and B into VGPR fragments.
    half16 a_frag;
    half16 b_frag;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        unsigned int idx = lane * 16 + i;
        a_frag[i]        = (idx < N * N) ? A[idx] : (_Float16)0.0f;
        b_frag[i]        = (idx < N * N) ? B[idx] : (_Float16)0.0f;
    }

    // Accumulator starts at zero.
    float8 acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    acc = __builtin_amdgcn_wmma_f32_16x16x32_f16(0u, a_frag, 0u, b_frag, 0u, acc, false, false);

    // Apply transcendental (exp) to the WMMA result.
    // This creates the critical hazard: WMMA writes acc VGPRs,
    // then v_exp_f32 reads them with no intervening wait.
    float8 result;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        result[i] = __expf(acc[i]);
    }

    // Store results unconditionally. With 32 lanes × 8 elements = 256 = N²,
    // every store index is in-bounds. Removing the per-element bounds check
    // keeps all stores in a single basic block, avoiding redundant
    // s_wait_kmcnt instructions the compiler inserts at block boundaries.
    float* C_lane = C + lane * 8;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        C_lane[i] = result[i];
    }
}

int main() {
    constexpr unsigned int N          = 16;
    constexpr unsigned int waveSize   = 32;
    constexpr unsigned int elemsPerLn = 8;
    static_assert(waveSize * elemsPerLn == N * N, "store loop assumes full coverage");
    constexpr std::size_t matSize = N * N;

    std::vector<_Float16> h_A(matSize);
    std::vector<_Float16> h_B(matSize);
    std::vector<float>    h_C(matSize, 0.0f);

    // Small values so exp() stays in a reasonable range.
    for (std::size_t i = 0; i < matSize; ++i) {
        h_A[i] = (_Float16)0.01f;
        h_B[i] = (_Float16)0.01f;
    }

    _Float16 *d_A, *d_B;
    float*    d_C;
    HIP_CHECK(hipMalloc(&d_A, matSize * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_B, matSize * sizeof(_Float16)));
    HIP_CHECK(hipMalloc(&d_C, matSize * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), matSize * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), matSize * sizeof(_Float16), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, matSize * sizeof(float)));

    // One wave = 32 threads.
    wmma_exp<<<1, 32>>>(d_A, d_B, d_C, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, matSize * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    // Print result as JSON array.
    std::cout << "[";
    for (std::size_t i = 0; i < matSize; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << h_C[i];
    }
    std::cout << "]" << std::endl;

    // Validate: all exp() outputs must be finite and strictly positive. The
    // output buffer starts zeroed, so accepting zero would let a kernel that
    // stored nothing report PASS.
    bool pass = true;
    for (std::size_t i = 0; i < matSize; ++i) {
        if (!std::isfinite(h_C[i]) || h_C[i] <= 0.0f) {
            std::cerr << "Bad value at " << i << ": " << h_C[i] << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
