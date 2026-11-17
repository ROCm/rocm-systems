// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sparse floating matrix validation and dispatch, retaining the legacy numerical model.
#include "mfma/backends.h"
#include "mfma/sparse_float.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
namespace {
// Validate policy and controls before accessing any operand.
template <int M, int K, int Bits, bool BfA, bool BfB, bool Fnuz>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *index) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, controls & ~0x7fULL,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  // Metadata sets occupy 8 bits for compact F16/BF16, 16 bits for
  // wide F16/BF16 or compact FP8/BF8, and all 32 bits for wide FP8/BF8.
  constexpr int set_bits = M * K / 64;
  const int shift = (controls & 3) == 0 ? set_bits * ((controls >> 3) & (32 / set_bits - 1)) : 0;
  uint32_t selected[64];
  const uint32_t *selected_index[] = {selected};
  if (shift) {
    for (int lane = 0; lane < 64; ++lane) {
      selected[lane] = index[0][lane] >> shift;
    }
    index = selected_index;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    smfmac_float_x86_64_v4<M, K, Bits, BfA, BfB, Fnuz>(d, a, b, index, flags & RJ_ARITH_FP16_OVFL);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    smfmac_float_x86_64_v3<M, K, Bits, BfA, BfB, Fnuz>(d, a, b, index, flags & RJ_ARITH_FP16_OVFL);
    return RJ_ARITH_SUCCESS;
  }
#endif
  smfmac_float_kernel<M, K, Bits, BfA, BfB, Fnuz>(d, a, b, index, flags & RJ_ARITH_FP16_OVFL);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x32_f16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index) {
  return evaluate<16, 32, 16, false, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x32_bf16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index) {
  return evaluate<16, 32, 16, true, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x16_f16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index) {
  return evaluate<32, 16, 16, false, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x16_bf16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index) {
  return evaluate<32, 16, 16, true, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_f16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index) {
  return evaluate<16, 64, 16, false, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_bf16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index) {
  return evaluate<16, 64, 16, true, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_f16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *index) {
  return evaluate<32, 32, 16, false, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_bf16_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                            const uint32_t *const *a, const uint32_t *const *b,
                                            const uint32_t *const *index) {
  return evaluate<32, 32, 16, true, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<16, 64, 8, false, false, true>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_fp8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index) {
  return evaluate<16, 64, 8, false, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<16, 64, 8, false, true, true>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_fp8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index) {
  return evaluate<16, 64, 8, false, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<16, 64, 8, true, false, true>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_bf8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index) {
  return evaluate<16, 64, 8, true, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<16, 64, 8, true, true, true>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x64_bf8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index) {
  return evaluate<16, 64, 8, true, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<32, 32, 8, false, false, true>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_fp8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index) {
  return evaluate<32, 32, 8, false, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<32, 32, 8, false, true, true>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_fp8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index) {
  return evaluate<32, 32, 8, false, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<32, 32, 8, true, false, true>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_bf8_fp8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index) {
  return evaluate<32, 32, 8, true, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<32, 32, 8, true, true, true>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x32_bf8_bf8_w64_gfx950(uint64_t flags, uint64_t controls,
                                                      uint32_t *const *d, const uint32_t *const *a,
                                                      const uint32_t *const *b,
                                                      const uint32_t *const *index) {
  return evaluate<32, 32, 8, true, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x128_fp8_fp8_w64(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<16, 128, 8, false, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x128_fp8_bf8_w64(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<16, 128, 8, false, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x128_bf8_fp8_w64(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<16, 128, 8, true, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_16x16x128_bf8_bf8_w64(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *index) {
  return evaluate<16, 128, 8, true, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x64_fp8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<32, 64, 8, false, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x64_fp8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<32, 64, 8, false, true, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x64_bf8_fp8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<32, 64, 8, true, false, false>(flags, controls, d, a, b, index);
}

// Accumulate the selected sparse products into the tied destination.
int rj_arith_v_smfmac_f32_32x32x64_bf8_bf8_w64(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b,
                                               const uint32_t *const *index) {
  return evaluate<32, 64, 8, true, true, false>(flags, controls, d, a, b, index);
}
