// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_PORTABLE_MATMUL_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_PORTABLE_MATMUL_H_

#include "util/simd.h"
#include <cmath>
#include <type_traits>

namespace rocjitsu::amdgpu {

// ---------------------------------------------------------------------------
// Execution kernels
// ---------------------------------------------------------------------------

/// Shared SIMD core for the MFMA/WMMA/SWMMAC executors. For every (row,col),
/// Cbuf[row*stride+col] += sum_k Abuf[row*K+k] * Bbuf[k*stride+col], run as
/// native-width rows over the col (N) dimension with a scalar tail. A/B/C must
/// already be hoisted into dense buffers (lane permutation, per-element scale,
/// and 2:4 sparsity gather folded in by the caller). Float uses fused FMA
/// (matching the hardware's single-rounding MACs; the scalar reference is
/// non-fused, so f32 agrees to a few ULP and packed f16/bf16 rounds identically);
/// integer MAC is exact, so the SIMD and scalar paths are bit-identical.
/// Templated so the `if constexpr (has_stdx_simd)` callers never instantiate it
/// on a platform without <experimental/simd>.
///
/// Cbuf is the in/out accumulator: each (row,col) is read as the starting value,
/// the K products are added, and the result written back. Callers come in two
/// flavors — most pre-seed Cbuf with the hardware C accumulator (D = C + A*B in
/// place), while the int32 WMMA paths instead leave Cbuf ZERO and add the
/// hardware accumulator afterward in wider precision (for saturation). That is
/// why the staging buffers are zero-initialized as a uniform convention: it is
/// load-bearing for the zero-start callers and harmless (redundant) for the
/// pre-seeded ones.
///
/// Read-bounds: only the used region is touched — Abuf[row*K+k] for row<M,k<K,
/// and Bbuf/Cbuf[..*stride+col] for col<N (the vectorized loop is bounded by
/// `col + W <= N`, the remainder is a scalar tail at col<N). The padding columns
/// [N, stride) — present only so each row starts W-aligned — are never read.
template <typename T>
void wmma_simd_matmul(uint32_t M, uint32_t N, uint32_t K, uint32_t W, uint32_t stride,
                      const T *Abuf, const T *Bbuf, T *Cbuf) {
  for (uint32_t row = 0; row < M; ++row) {
    uint32_t col = 0;
    for (; col + W <= N; col += W) {
      util::native<T> c;
      c.copy_from(&Cbuf[row * stride + col], util::stdx::vector_aligned);
      for (uint32_t k = 0; k < K; ++k) {
        util::native<T> a(Abuf[row * K + k]);
        util::native<T> bv;
        bv.copy_from(&Bbuf[k * stride + col], util::stdx::vector_aligned);
        if constexpr (std::is_floating_point_v<T>)
          c = util::stdx::fma(a, bv, c);
        else
          c += a * bv;
      }
      c.copy_to(&Cbuf[row * stride + col], util::stdx::vector_aligned);
    }
    for (; col < N; ++col) {
      T acc = Cbuf[row * stride + col];
      for (uint32_t k = 0; k < K; ++k) {
        if constexpr (std::is_floating_point_v<T>)
          acc = std::fma(Abuf[row * K + k], Bbuf[k * stride + col], acc);
        else
          acc += Abuf[row * K + k] * Bbuf[k * stride + col];
      }
      Cbuf[row * stride + col] = acc;
    }
  }
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_PORTABLE_MATMUL_H_
