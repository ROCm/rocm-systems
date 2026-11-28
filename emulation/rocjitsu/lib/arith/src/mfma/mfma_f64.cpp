// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared CDNA FP64 MFMA validation, baseline evaluation and SIMD dispatch.
#include "common/dense_f64.h"
#include "common/dense_f64_backends.h"
#include "common/internal.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_mfma.h"
namespace {
// Validate the inherited approximate model before accessing operands.
template <int M, int Batch>
int evaluate(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
             const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error =
          validate(flags, controls & ~7ULL, kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    dense_f64_x86_64_v4<M, Batch>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    dense_f64_x86_64_v3<M, Batch>(controls, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
#endif
  dense_f64_kernel<M, Batch, 64>(controls, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}
} // namespace

// Evaluate 1 independent 16x16 FP64 output matrices.
int rj_arith_v_mfma_f64_16x16x4_f64_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c) {
  return evaluate<16, 1>(flags, controls, d, a, b, c);
}

// Evaluate 4 independent 4x4 FP64 output matrices.
int rj_arith_v_mfma_f64_4x4x4_4b_f64_w64(uint64_t flags, uint64_t controls, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         const uint32_t *const *c) {
  return evaluate<4, 4>(flags, controls, d, a, b, c);
}

namespace {

uint64_t step_fma64_gfx950(uint64_t a, uint64_t b, uint64_t c) {
  using namespace rocjitsu::arith;
  auto is_nan = [](uint64_t u) {
    return (u & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL && (u & 0x000fffffffffffffULL) != 0;
  };
  auto is_inf = [](uint64_t u) { return (u & 0x7fffffffffffffffULL) == 0x7ff0000000000000ULL; };
  const bool na = is_nan(a), nb = is_nan(b), nc = is_nan(c);
  const bool ia = is_inf(a), ib = is_inf(b), ic = is_inf(c);
  const bool za = (a & 0x7fffffffffffffffULL) == 0, zb = (b & 0x7fffffffffffffffULL) == 0;
  if ((ia && zb) || (za && ib)) {
    if (na) {
      return a | 0x0008000000000000ULL;
    }
    if (nb) {
      return b | 0x0008000000000000ULL;
    }
    return 0xfff8000000000000ULL;
  }
  if (na) {
    return a | 0x0008000000000000ULL;
  }
  if (nb) {
    return b | 0x0008000000000000ULL;
  }
  if (nc) {
    return c | 0x0008000000000000ULL;
  }
  if (ia || ib) {
    const uint64_t sign = (a ^ b) & 0x8000000000000000ULL;
    if (ic && ((c & 0x8000000000000000ULL) != sign)) {
      return 0xfff8000000000000ULL;
    }
    return sign | 0x7ff0000000000000ULL;
  }
  if (ic) {
    return c;
  }
  return bit_cast<uint64_t>(__builtin_fma(as_double(a), as_double(b), as_double(c)));
}

// Keep exact and source-abs stack buffers out of the SIMD dispatch prologue.
template <int M, int Batch>
[[gnu::noinline]] void evaluate_gfx950_loose_abs(uint64_t flags, uint64_t neg, uint64_t abs_mask,
                                                 uint32_t *const *d, const uint32_t *const *a,
                                                 const uint32_t *const *b,
                                                 const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  constexpr int Outputs = 2 * M * M * Batch / 64;
  uint32_t abs_a_hi[64], abs_b_hi[64], abs_c_hi[Outputs / 2][64];
  const uint32_t *abs_a_ptrs[2] = {a[0], a[1]};
  const uint32_t *abs_b_ptrs[2] = {b[0], b[1]};
  const uint32_t *abs_c_ptrs[Outputs];
  for (int reg = 0; reg < Outputs; ++reg) {
    abs_c_ptrs[reg] = c[reg];
  }
  if (abs_mask & 1) {
    for (int lane = 0; lane < 64; ++lane) {
      abs_a_hi[lane] = a[1][lane] & 0x7fffffffU;
    }
    abs_a_ptrs[1] = abs_a_hi;
  }
  if (abs_mask & 2) {
    for (int lane = 0; lane < 64; ++lane) {
      abs_b_hi[lane] = b[1][lane] & 0x7fffffffU;
    }
    abs_b_ptrs[1] = abs_b_hi;
  }
  if (abs_mask & 4) {
    for (int reg = 0; reg < Outputs; reg += 2) {
      for (int lane = 0; lane < 64; ++lane) {
        abs_c_hi[reg / 2][lane] = c[reg + 1][lane] & 0x7fffffffU;
      }
      abs_c_ptrs[reg + 1] = abs_c_hi[reg / 2];
    }
  }
  const uint32_t *const *pa = (abs_mask & 1) ? abs_a_ptrs : a;
  const uint32_t *const *pb = (abs_mask & 2) ? abs_b_ptrs : b;
  const uint32_t *const *pc = (abs_mask & 4) ? abs_c_ptrs : c;
#if defined(RJ_ARITH_HAVE_X86_64_V4)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
    dense_f64_x86_64_v4<M, Batch>(neg, d, pa, pb, pc);
    return;
  }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
    dense_f64_x86_64_v3<M, Batch>(neg, d, pa, pb, pc);
    return;
  }
#endif
  dense_f64_kernel<M, Batch, 64>(neg, d, pa, pb, pc);
}

template <int M, int Batch>
[[gnu::noinline]] void evaluate_gfx950_exact(uint64_t neg, uint64_t abs_mask, uint32_t *const *d,
                                             const uint32_t *const *a, const uint32_t *const *b,
                                             const uint32_t *const *c) {
  constexpr int Outputs = 2 * M * M * Batch / 64;
  constexpr int groups = 64 / M;
  constexpr int batch_groups = (64 + M * M - 1) / (M * M);
  auto load_mod = [](const uint32_t *const *p, int reg, int lane, bool do_abs, bool do_neg) {
    uint64_t bits = uint64_t(p[reg][lane]) | (uint64_t(p[reg + 1][lane]) << 32);
    if (do_abs) {
      bits &= 0x7fffffffffffffffULL;
    }
    if (do_neg) {
      bits ^= 0x8000000000000000ULL;
    }
    return bits;
  };
  uint64_t staged_a[64], staged_b[64], staged_d[Outputs / 2][64];
  for (int lane = 0; lane < 64; ++lane) {
    staged_a[lane] = load_mod(a, 0, lane, (abs_mask & 1) != 0, (neg & 1) != 0);
    staged_b[lane] = load_mod(b, 0, lane, (abs_mask & 2) != 0, (neg & 2) != 0);
  }
  for (int reg = 0; reg < Outputs; reg += 2) {
    for (int lane = 0; lane < 64; ++lane) {
      staged_d[reg / 2][lane] = load_mod(c, reg, lane, (abs_mask & 4) != 0, (neg & 4) != 0);
    }
  }
  for (int batch = 0; batch < Batch; ++batch) {
    for (int row = 0; row < M; ++row) {
      const int d_idx = batch * (M * M / 64) + row / groups;
      for (int col = 0; col < M; ++col) {
        const int olane = (batch % batch_groups) * M + (row % groups) * batch_groups * M + col;
        uint64_t acc = staged_d[d_idx][olane];
        for (int k = 0; k < 4; ++k) {
          const int al = batch * M + k * M * Batch + row;
          const int bl = batch * M + k * M * Batch + col;
          acc = step_fma64_gfx950(staged_a[al], staged_b[bl], acc);
        }
        staged_d[d_idx][olane] = acc;
      }
    }
  }
  for (int reg = 0; reg < Outputs; reg += 2) {
    for (int lane = 0; lane < 64; ++lane) {
      const uint64_t bits = staged_d[reg / 2][lane];
      d[reg][lane] = uint32_t(bits);
      d[reg + 1][lane] = uint32_t(bits >> 32);
    }
  }
}

template <int M, int Batch>
int evaluate_gfx950(uint64_t flags, uint64_t controls, uint32_t *const *d, const uint32_t *const *a,
                    const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, controls & ~0x3fULL,
                           kMatrixIgnoredFpFlags | RJ_ARITH_FP_DX10_CLAMP_OFF)) {
    return error;
  }
  if (reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  const uint64_t neg = controls & 7ULL;
  const uint64_t abs_mask = (controls >> 3) & 7ULL;
  const bool qualified = neg == 0 && abs_mask <= 4;
  if (!qualified &&
      (flags & RJ_ARITH_SEMANTICS_MASK) >= RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!is_exact(flags) || !qualified) {
    if (abs_mask == 0) {
#if defined(RJ_ARITH_HAVE_X86_64_V4)
      if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
        dense_f64_x86_64_v4<M, Batch>(neg, d, a, b, c);
        return RJ_ARITH_SUCCESS;
      }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
      if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
        dense_f64_x86_64_v3<M, Batch>(neg, d, a, b, c);
        return RJ_ARITH_SUCCESS;
      }
#endif
      dense_f64_kernel<M, Batch, 64>(neg, d, a, b, c);
      return RJ_ARITH_SUCCESS;
    }
    evaluate_gfx950_loose_abs<M, Batch>(flags, neg, abs_mask, d, a, b, c);
    return RJ_ARITH_SUCCESS;
  }
  evaluate_gfx950_exact<M, Batch>(neg, abs_mask, d, a, b, c);
  return RJ_ARITH_SUCCESS;
}

} // namespace

int rj_arith_v_mfma_f64_16x16x4_f64_w64_gfx950(uint64_t flags, uint64_t controls,
                                               uint32_t *const *d, const uint32_t *const *a,
                                               const uint32_t *const *b, const uint32_t *const *c) {
  return evaluate_gfx950<16, 1>(flags, controls, d, a, b, c);
}

int rj_arith_v_mfma_f64_4x4x4_4b_f64_w64_gfx950(uint64_t flags, uint64_t controls,
                                                uint32_t *const *d, const uint32_t *const *a,
                                                const uint32_t *const *b,
                                                const uint32_t *const *c) {
  return evaluate_gfx950<4, 4>(flags, controls, d, a, b, c);
}
