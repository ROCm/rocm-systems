// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP32 and XF32 MFMA use the shared floating-point reduction.
#include "common/internal.h"
#include "mfma/backends.h"
#include "mfma/gfx942_exact.h"
#include "mfma/reduction.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"

namespace {
// Keep the scalar stack buffer out of the SIMD dispatch prologue.
template <int M, int K, int Batch, bool Xf32, rocjitsu::arith::MfmaF32Model Model>
[[gnu::noinline]] void evaluate_scalar(uint64_t controls, uint64_t flags, bool clamp_overflow,
                                       uint32_t *const *d, const uint32_t *const *a,
                                       const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  constexpr uint32_t input_mask = Xf32 ? kMfmaXf32InputMask : UINT32_MAX;
  float left[M * K * Batch], right[M * K * Batch];
  for (int i = 0; i < M * K * Batch; ++i) {
    left[i] = as_float(a[i / 64][i % 64] & input_mask);
    right[i] = as_float(b[i / 64][i % 64] & input_mask);
    if constexpr (Model == MfmaF32Model::Gfx90a) {
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        left[i] = mfma_flush_f32(left[i]);
        right[i] = mfma_flush_f32(right[i]);
      }
    }
  }
  mfma_reduce_scalar<M, K, Batch, 1, false, Model>(controls, d, left, right, c, clamp_overflow,
                                                   flags);
}

// Validate the approximate model and evaluate a full matrix; errors do not touch operands.
template <int M, int K, int Batch, bool Xf32 = false, bool Gfx942 = false,
          rocjitsu::arith::MfmaF32Model Model = rocjitsu::arith::MfmaF32Model::Default>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, controls & ~0x3ffULL,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  // Captured gfx942 XF32 forms ignore CBSZ/ABID/BLGP in hardware; CBSZ <= 4 with
  // ABID < 2^CBSZ and BLGP 0..7 are qualified in exact modes. Compatibility retains
  // the existing fast approximate implementation.
  if constexpr (Gfx942) {
    if (is_exact(flags) && is_gfx942_mfma_controls_qualified(controls)) {
      const bool no_inf_nan = (flags & RJ_ARITH_FP16_OVFL) != 0;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
      if (!no_inf_nan && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
        mfma_gfx942_exact_x86_64_v4<MfmaGfx942Format::Xf32, M, K, Batch>(d, a, b, c);
        return RJ_ARITH_SUCCESS;
      }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
      if (!no_inf_nan && cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
        mfma_gfx942_exact_x86_64_v3<MfmaGfx942Format::Xf32, M, K, Batch>(d, a, b, c);
        return RJ_ARITH_SUCCESS;
      }
#endif
      mfma_gfx942_exact(MfmaGfx942Format::Xf32, M, K, Batch, no_inf_nan, d, a, b, c);
      return RJ_ARITH_SUCCESS;
    }
    controls = 0;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  const bool clamp_overflow =
      Model == MfmaF32Model::Default && !Xf32 && (flags & RJ_ARITH_FP16_OVFL);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    mfma_f32_x86_64_v4<M, K, Batch, Xf32, Model>(controls, flags, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    mfma_f32_x86_64_v3<M, K, Batch, Xf32, Model>(controls, flags, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  evaluate_scalar<M, K, Batch, Xf32, Model>(controls, flags, clamp_overflow, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate 16x16x8 XF32 MFMA.
int rj_arith_v_mfma_f32_16x16x8_xf32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<16, 8, 1, true, true>(flags, controls, d, a, b, c);
}
// Evaluate 32x32x4 XF32 MFMA.
int rj_arith_v_mfma_f32_32x32x4_xf32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<32, 4, 1, true, true>(flags, controls, d, a, b, c);
}

// Evaluate 2 independent 32x32x1 FP32 matrices.
int rj_arith_v_mfma_f32_32x32x1_2b_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c) {
  return evaluate<32, 1, 2>(flags, controls, d, a, b, c);
}

// Evaluate 4 independent 16x16x1 FP32 matrices.
int rj_arith_v_mfma_f32_16x16x1_4b_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                           const uint32_t *const *a, const uint32_t *const *b,
                                           const uint32_t *const *c) {
  return evaluate<16, 1, 4>(flags, controls, d, a, b, c);
}

// Evaluate 16 independent 4x4x1 FP32 matrices.
int rj_arith_v_mfma_f32_4x4x1_16b_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                          const uint32_t *const *a, const uint32_t *const *b,
                                          const uint32_t *const *c) {
  return evaluate<4, 1, 16>(flags, controls, d, a, b, c);
}

// Evaluate 1 independent 32x32x2 FP32 matrices.
int rj_arith_v_mfma_f32_32x32x2_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<32, 2, 1>(flags, controls, d, a, b, c);
}

// Evaluate 1 independent 16x16x4 FP32 matrices.
int rj_arith_v_mfma_f32_16x16x4_f32_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<16, 4, 1>(flags, controls, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_16x16x8_xf32_w64_gfx950(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return evaluate<16, 8, 1, true>(flags, controls, d, a, b, c);
}

// Evaluate the gfx950 form using the existing approximate model.
int rj_arith_v_mfma_f32_32x32x4_xf32_w64_gfx950(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return evaluate<32, 4, 1, true>(flags, controls, d, a, b, c);
}

// Retain gfx908's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_32x32x1_2b_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return evaluate<32, 1, 2, false, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d,
                                                                                 a, b, c);
}

// Retain gfx908's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_16x16x1_4b_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return evaluate<16, 1, 4, false, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d,
                                                                                 a, b, c);
}

// Retain gfx908's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_4x4x1_16b_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  return evaluate<4, 1, 16, false, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d,
                                                                                 a, b, c);
}

// Retain gfx908's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_32x32x2_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return evaluate<32, 2, 1, false, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d,
                                                                                 a, b, c);
}

// Retain gfx908's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx908(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return evaluate<16, 4, 1, false, false, rocjitsu::arith::MfmaF32Model::Gfx908>(flags, controls, d,
                                                                                 a, b, c);
}

// Retain gfx90a's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_32x32x1_2b_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return evaluate<32, 1, 2, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d,
                                                                                 a, b, c);
}

// Retain gfx90a's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_16x16x1_4b_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return evaluate<16, 1, 4, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d,
                                                                                 a, b, c);
}

// Retain gfx90a's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_4x4x1_16b_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  return evaluate<4, 1, 16, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d,
                                                                                 a, b, c);
}

// Retain gfx90a's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_32x32x2_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return evaluate<32, 2, 1, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d,
                                                                                 a, b, c);
}

// Retain gfx90a's input/output denormal and A broadcast policy.
int rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx90a(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return evaluate<16, 4, 1, false, false, rocjitsu::arith::MfmaF32Model::Gfx90a>(flags, controls, d,
                                                                                 a, b, c);
}

namespace {

int a_lane_f32_gfx950(int lane, int M, int cbsz, int abid) {
  if (cbsz <= 0 || cbsz > 4) {
    return lane;
  }
  const int num_blocks = 64 / M;
  const int group = 1 << cbsz;
  if (group > num_blocks) {
    return lane % 16;
  }
  const int block = lane / M;
  const int selected = (block & ~(group - 1)) | (abid & (group - 1));
  return selected * M + (lane % M);
}

int b_lane_perm_gfx950(int lane, int blgp) {
  if (blgp == 0) {
    return lane;
  }
  if (blgp <= 2) {
    return (lane % 32) + (blgp - 1) * 32;
  }
  if (blgp == 3) {
    return (lane + 16) % 64;
  }
  return (lane % 16) + (blgp - 4) * 16;
}

uint32_t step_fma32_gfx950(uint32_t a, uint32_t b, uint32_t c, bool flush_denorm) {
  using namespace rocjitsu::arith;
  auto is_sub = [](uint32_t u) { return (u & 0x7f800000U) == 0 && (u & 0x007fffffU) != 0; };
  auto is_nan = [](uint32_t u) {
    return (u & 0x7f800000U) == 0x7f800000U && (u & 0x007fffffU) != 0;
  };
  auto is_inf = [](uint32_t u) { return (u & 0x7fffffffU) == 0x7f800000U; };
  if (flush_denorm) {
    if (is_sub(a)) {
      a &= 0x80000000U;
    }
    if (is_sub(b)) {
      b &= 0x80000000U;
    }
    if (is_sub(c)) {
      c &= 0x80000000U;
    }
  }
  const bool na = is_nan(a), nb = is_nan(b), nc = is_nan(c);
  const bool ia = is_inf(a), ib = is_inf(b), ic = is_inf(c);
  const bool za = (a & 0x7fffffffU) == 0, zb = (b & 0x7fffffffU) == 0;
  if ((ia && zb) || (za && ib)) {
    if (na) {
      return a | 0x00400000U;
    }
    if (nb) {
      return b | 0x00400000U;
    }
    return 0xffc00000U;
  }
  if (na) {
    return a | 0x00400000U;
  }
  if (nb) {
    return b | 0x00400000U;
  }
  if (nc) {
    return c | 0x00400000U;
  }
  if (ia || ib) {
    const uint32_t sign = (a ^ b) & 0x80000000U;
    if (ic && ((c & 0x80000000U) != sign)) {
      return 0xffc00000U;
    }
    return sign | 0x7f800000U;
  }
  if (ic) {
    return c;
  }
  uint32_t out = as_bits(__builtin_fmaf(as_float(a), as_float(b), as_float(c)));
  if (flush_denorm && is_sub(out)) {
    out &= 0x80000000U;
  }
  return out;
}

// Keep exact and permuted/scalar stack buffers out of the SIMD dispatch prologue.
template <int M, int K, int Batch>
[[gnu::noinline]] void evaluate_gfx950_exact(int cbsz, int abid, int blgp, bool flush_in,
                                             uint32_t *const *d, const uint32_t *const *a,
                                             const uint32_t *const *b, const uint32_t *const *c) {
  constexpr int OutputRegs = M * M * Batch / 64;
  constexpr int multirows = 64 / M;
  constexpr int blocks_per_reg = (64 + M * M / 4 - 1) / (M * M / 4);
  uint32_t staged_a[64], staged_b[64], staged_d[OutputRegs][64];
  for (int lane = 0; lane < 64; ++lane) {
    staged_a[lane] = a[0][lane];
    staged_b[lane] = b[0][lane];
  }
  for (int reg = 0; reg < OutputRegs; ++reg) {
    for (int lane = 0; lane < 64; ++lane) {
      staged_d[reg][lane] = c[reg][lane];
    }
  }
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < M; ++row) {
      for (int col = 0; col < M; ++col) {
        const int oreg = batch * (M * M / 64) + (row / (4 * multirows)) * 4 + (row % 4);
        const int olane =
            (batch % blocks_per_reg) * M + ((row / 4) % multirows) * blocks_per_reg * M + col;
        uint32_t acc = staged_d[oreg][olane];
        for (int k = 0; k < K; ++k) {
          const int al = a_lane_f32_gfx950(batch * M + k * M * Batch + row, M, cbsz, abid);
          const int bl = b_lane_perm_gfx950(batch * M + k * M * Batch + col, blgp);
          acc = step_fma32_gfx950(staged_a[al], staged_b[bl], acc, flush_in);
        }
        staged_d[oreg][olane] = acc;
      }
    }
  }
  for (int reg = 0; reg < OutputRegs; ++reg) {
    __builtin_memcpy(d[reg], staged_d[reg], sizeof(staged_d[reg]));
  }
}

template <int M, int K, int Batch>
[[gnu::noinline]] void
evaluate_gfx950_loose_slow(uint64_t loose_controls, uint64_t loose_flags, int cbsz, int abid,
                           uint32_t *const *d, const uint32_t *const *a, const uint32_t *const *b,
                           const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  uint32_t perm_a[64];
  const uint32_t *perm_a_ptr = a[0];
  if (cbsz > 0 && cbsz <= 4) {
    for (int lane = 0; lane < 64; ++lane) {
      perm_a[lane] = a[0][a_lane_f32_gfx950(lane, M, cbsz, abid)];
    }
    perm_a_ptr = perm_a;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(loose_flags, RJ_ARITH_CPU_X86_64_V4)) {
    mfma_f32_x86_64_v4<M, K, Batch, false>(loose_controls, loose_flags, d, &perm_a_ptr, b, c);
    return;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(loose_flags, RJ_ARITH_CPU_X86_64_V3)) {
    mfma_f32_x86_64_v3<M, K, Batch, false>(loose_controls, loose_flags, d, &perm_a_ptr, b, c);
    return;
  }
#endif
  float left[64], right[64];
  for (int i = 0; i < 64; ++i) {
    left[i] = as_float(perm_a_ptr[i]);
    right[i] = as_float(b[0][i]);
  }
  mfma_reduce_scalar<M, K, Batch, 1>(loose_controls, d, left, right, c, false);
}

template <int M, int K, int Batch>
int evaluate_gfx950(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                    const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, controls & ~0x3ffULL,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  const int cbsz = int(controls & 7);
  const int abid = int((controls >> 3) & 15);
  const int blgp = int((controls >> 7) & 7);
  const bool flush_in = (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) != 0;
  const bool flush_out = (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) != 0;
  if (is_exact(flags) && cbsz <= 4 && flush_in == flush_out) {
    evaluate_gfx950_exact<M, K, Batch>(cbsz, abid, blgp, flush_in, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  const uint64_t loose_flags = flags & ~RJ_ARITH_FP16_OVFL;
  const uint64_t loose_controls = controls & ~0x7fULL;
  if (cbsz <= 0 || cbsz > 4) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if (cpu_level_at_least(loose_flags, RJ_ARITH_CPU_X86_64_V4)) {
      mfma_f32_x86_64_v4<M, K, Batch, false>(loose_controls, loose_flags, d, a, b, c);
      return RJ_ARITH_SUCCESS;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (cpu_level_at_least(loose_flags, RJ_ARITH_CPU_X86_64_V3)) {
      mfma_f32_x86_64_v3<M, K, Batch, false>(loose_controls, loose_flags, d, a, b, c);
      return RJ_ARITH_SUCCESS;
    }
#endif
  }
  evaluate_gfx950_loose_slow<M, K, Batch>(loose_controls, loose_flags, cbsz, abid, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}

} // namespace

int rj_arith_v_mfma_f32_32x32x1_2b_f32_w64_gfx950(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return evaluate_gfx950<32, 1, 2>(flags, controls, d, a, b, c);
}

int rj_arith_v_mfma_f32_16x16x1_4b_f32_w64_gfx950(uint64_t flags, uint64_t controls,
                                                  uint32_t *const *d, const uint32_t *const *a,
                                                  const uint32_t *const *b,
                                                  const uint32_t *const *c) {
  return evaluate_gfx950<16, 1, 4>(flags, controls, d, a, b, c);
}

int rj_arith_v_mfma_f32_4x4x1_16b_f32_w64_gfx950(uint64_t flags, uint64_t controls,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  return evaluate_gfx950<4, 1, 16>(flags, controls, d, a, b, c);
}

int rj_arith_v_mfma_f32_32x32x2_f32_w64_gfx950(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return evaluate_gfx950<32, 2, 1>(flags, controls, d, a, b, c);
}

int rj_arith_v_mfma_f32_16x16x4_f32_w64_gfx950(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return evaluate_gfx950<16, 4, 1>(flags, controls, d, a, b, c);
}
