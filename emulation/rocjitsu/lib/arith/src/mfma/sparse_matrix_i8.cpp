// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse signed-byte SMFMAC and mixed-signedness gfx1250 SWMMAC dispatch.
#include "common/internal.h"
#include "mfma/backends.h"
#include "mfma/sparse_integer.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
#include "rj_arith/detail/rj_arith_wmma.h"
namespace {
// Validate metadata controls before reading operands; integer arithmetic is exact in every mode.
template <int M, int K, bool Wmma>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *index) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL);
  constexpr uint64_t allowed = Wmma ? RJ_ARITH_WMMA_SIGNED_A | RJ_ARITH_WMMA_SIGNED_B |
                                          RJ_ARITH_WMMA_CLAMP | RJ_ARITH_SWMMAC_INDEX_KEY_1
                                    : 0x7fULL;
  const bool clamp = Wmma ? (controls & RJ_ARITH_WMMA_CLAMP) : (flags & RJ_ARITH_FP16_OVFL);
  if (int error = validate(flags, controls & ~allowed,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  // LLVM folds a per-word logical metadata shift into index_key:1. Normalize
  // that encoding before the common scalar/SIMD layout decoder consumes it.
  uint32_t shifted[2][32];
  const uint32_t *selected[2];
  if (Wmma && (controls & RJ_ARITH_SWMMAC_INDEX_KEY_1)) {
    for (int reg = 0; reg < (2); ++reg) {
      for (int lane = 0; lane < 32; ++lane) {
        shifted[reg][lane] = index[reg][lane] >> 16;
      }
      selected[reg] = shifted[reg];
    }
    index = selected;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    sparse_matrix_i8_x86_64_v4<M, K, Wmma>(controls, clamp, d, a, b, index);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    sparse_matrix_i8_x86_64_v3<M, K, Wmma>(controls, clamp, d, a, b, index);
    return RJ_ARITH_SUCCESS;
  }
#endif
  sparse_matrix_i8_kernel<M, K, Wmma>(controls, clamp, d, a, b, index);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Update D from compressed A, selected dense B, and its original accumulator.
int rj_arith_v_smfmac_i32_16x16x64_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *index) {
  return evaluate<16, 64, false>(flags, controls, d, a, b, index);
}

// Update D from compressed A, selected dense B, and its original accumulator.
int rj_arith_v_smfmac_i32_32x32x32_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *index) {
  return evaluate<32, 32, false>(flags, controls, d, a, b, index);
}

// Update D from compressed A, selected dense B, and its original accumulator.
int rj_arith_v_smfmac_i32_16x16x128_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index) {
  return evaluate<16, 128, false>(flags, controls, d, a, b, index);
}

// Update D from compressed A, selected dense B, and its original accumulator.
int rj_arith_v_smfmac_i32_32x32x64_i8_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *index) {
  return evaluate<32, 64, false>(flags, controls, d, a, b, index);
}

// Update D from compressed A, selected dense B, and its original accumulator.
int rj_arith_v_swmmac_i32_16x16x128_iu8_w32(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index) {
  return evaluate<16, 128, true>(flags, controls, d, a, b, index);
}
