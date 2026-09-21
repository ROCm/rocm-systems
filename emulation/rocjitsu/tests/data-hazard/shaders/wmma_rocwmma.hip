// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "utils.hpp"

// rocWMMA-based GEMM followed by transcendental (exp).
//
// This uses the rocWMMA C++ template library (load_matrix_sync, mma_sync,
// store_matrix_sync) instead of raw builtins, exercising the same WMMA→
// transcendental data-hazard surface as wmma_exp.cpp but through the
// higher-level rocWMMA API.
//
// Each wavefront loads 16×16 tile fragments of A (f16, row-major) and
// B (f16, col-major) from global memory, performs a 16×16×16 f16 MMA
// that accumulates into f32, applies v_exp_f32 on every accumulator
// element, and stores the results back.
//
// Hazard surface:
//   - global_load → VGPR (load_matrix_sync)
//   - WMMA writes accumulator VGPRs (mma_sync)
//   - v_exp_f32 reads accumulator VGPRs immediately (WMMA→trans hazard)
//   - global_store reads result VGPRs (trans→store hazard)
//
// The launch below is one 32-thread block, which is a whole wave only where
// waves are 32 lanes wide. On a wave64 target those threads are half a wave
// and the rocWMMA fragments never cover the tile, so the harness is told to
// skip this kernel everywhere else:
// requires: gfx1250

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

using FragA   = rocwmma::fragment<rocwmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::row_major>;
using FragB   = rocwmma::fragment<rocwmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, rocwmma::col_major>;
using FragAcc = rocwmma::fragment<rocwmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>;

__global__ void wmma_rocwmma(const __half* __restrict__ A,
                             const __half* __restrict__ B,
                             float* __restrict__ C,
                             unsigned int M,
                             unsigned int N,
                             unsigned int K) {
    // Each wavefront computes one WMMA_M × WMMA_N tile of C.
    const unsigned int waveM   = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const unsigned int tileRow = (waveM / (N / WMMA_N)) * WMMA_M;
    const unsigned int tileCol = (waveM % (N / WMMA_N)) * WMMA_N;

    if (tileRow >= M || tileCol >= N)
        return;

    FragAcc acc;
    rocwmma::fill_fragment(acc, 0.0f);

    // Accumulate over K dimension in WMMA_K steps.
    for (unsigned int k = 0; k < K; k += WMMA_K) {
        FragA a_frag;
        FragB b_frag;
        rocwmma::load_matrix_sync(a_frag, A + tileRow * K + k, K);
        rocwmma::load_matrix_sync(b_frag, B + k * N + tileCol, N);
        rocwmma::mma_sync(acc, a_frag, b_frag, acc);
    }

    // Apply transcendental (exp) directly to accumulator registers.
    // This creates the critical hazard: mma_sync writes acc VGPRs,
    // then v_exp_f32 reads them with no intervening wait.
    for (int i = 0; i < acc.num_elements; ++i) {
        acc.x[i] = __expf(acc.x[i]);
    }

    // Store the result tile.
    rocwmma::store_matrix_sync(C + tileRow * N + tileCol, acc, N, rocwmma::mem_row_major);
}

int main() {
    constexpr unsigned int M     = 16;
    constexpr unsigned int N     = 16;
    constexpr unsigned int K     = 16;
    constexpr std::size_t  sizeA = M * K;
    constexpr std::size_t  sizeB = K * N;
    constexpr std::size_t  sizeC = M * N;

    std::vector<__half> h_A(sizeA);
    std::vector<__half> h_B(sizeB);
    std::vector<float>  h_C(sizeC, 0.0f);

    // Small values so exp(result) stays in a reasonable range.
    for (std::size_t i = 0; i < sizeA; ++i)
        h_A[i] = __float2half(0.01f);
    for (std::size_t i = 0; i < sizeB; ++i)
        h_B[i] = __float2half(0.01f);

    __half *d_A, *d_B;
    float*  d_C;
    HIP_CHECK(hipMalloc(&d_A, sizeA * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_B, sizeB * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_C, sizeC * sizeof(float)));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), sizeA * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), sizeB * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_C, 0, sizeC * sizeof(float)));

    // One tile = one wavefront = 32 threads (wave32 on gfx1250).
    // For a single 16×16 output we need one wave.
    wmma_rocwmma<<<1, 32>>>(d_A, d_B, d_C, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_C.data(), d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    // Print result as JSON array.
    std::cout << "[";
    for (std::size_t i = 0; i < sizeC; ++i) {
        if (i > 0)
            std::cout << ",";
        std::cout << h_C[i];
    }
    std::cout << "]" << std::endl;

    // Validate: all exp() outputs must be finite and strictly positive. The
    // output buffer starts zeroed, so accepting zero would let a kernel that
    // stored nothing report PASS.
    bool pass = true;
    for (std::size_t i = 0; i < sizeC; ++i) {
        if (!std::isfinite(h_C[i]) || h_C[i] <= 0.0f) {
            std::cerr << "Bad value at " << i << ": " << h_C[i] << std::endl;
            pass = false;
            break;
        }
    }
    std::cerr << (pass ? "PASS" : "FAIL") << std::endl;
    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}
