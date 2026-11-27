// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// CDNA FP16/BF16 vector dots: host FP32 pair sum followed by accumulation.
// Inherits rocjitsu's arithmetic order, not the distinct gfx11/gfx12 exact models.

#include "common/float_encoding.h"
#include "common/internal.h"
#include "common/simd.h"
#include "rj_arith/detail/rj_arith_common.h"
#include "rj_arith/detail/rj_arith_dot.h"

#include <stdint.h>

namespace {

template <typename T> T min_val(T x, T y) { return x < y ? x : y; }
template <typename T> T max_val(T x, T y) { return x > y ? x : y; }
template <typename T> T clamp_val(T v, T lo, T hi) { return v < lo ? lo : (v > hi ? hi : v); }

uint32_t dot2_f16_gfx950_exact(uint16_t lo_a, uint16_t lo_b, uint16_t hi_a, uint16_t hi_b,
                               uint32_t c_bits) {
  using namespace rocjitsu::arith;
  const float a[2] = {f16_to_float(lo_a), f16_to_float(hi_a)};
  const float b[2] = {f16_to_float(lo_b), f16_to_float(hi_b)};
  auto decode = [](float val, int32_t &mant, int32_t &exp, bool &inf_nan) {
    const uint32_t bits = as_bits(val);
    const int32_t ef = int32_t((bits >> 23) & 0xffU);
    const int32_t e = max_val(ef, 1) - 127;
    const uint32_t fm = (bits & 0x7fffffU) | (ef != 0 ? 0x800000U : 0U);
    const int32_t below = clamp_val(-14 - e, 0, 31);
    const int32_t mag = int32_t((fm >> below) >> 13);
    mant = (bits >> 31) ? -mag : mag;
    exp = max_val(e, -14);
    inf_nan = (ef == 0xff);
  };
  int32_t am[2], ae[2], bm[2], be[2];
  bool ai0, ai1, bi0, bi1;
  decode(a[0], am[0], ae[0], ai0);
  decode(a[1], am[1], ae[1], ai1);
  decode(b[0], bm[0], be[0], bi0);
  decode(b[1], bm[1], be[1], bi1);

  int32_t pm[2], pe[2], top_e = -999;
  for (int i = 0; i < 2; ++i) {
    pm[i] = am[i] * bm[i];
    pe[i] = pm[i] != 0 ? ae[i] + be[i] : -999;
    top_e = max_val(top_e, pe[i]);
  }
  int64_t aligned = 0;
  for (int i = 0; i < 2; ++i) {
    if (pm[i] == 0) {
      continue;
    }
    const int32_t rsh = top_e - pe[i];
    const int64_t scaled = int64_t(pm[i]) * 16;
    if (rsh <= 0) {
      aligned += scaled;
    } else if (rsh < 62) {
      const int64_t tz = scaled < 0 ? ((int64_t{1} << rsh) - 1) : 0;
      aligned += (scaled + tz) >> rsh;
    }
  }
  const int32_t dot_sum = int32_t(aligned);
  const int64_t cef = (c_bits >> 23) & 0xffU;
  const int64_t ce = max_val<int64_t>(cef, 1) - 127;
  const int64_t cmag = (c_bits & 0x7fffffU) | (cef != 0 ? 0x800000U : 0U);
  const int64_t cm = (c_bits >> 31) ? -cmag : cmag;
  const int64_t te = max_val<int64_t>(top_e, ce);
  const int64_t dot_part = (int64_t(dot_sum) * 256) >> min_val<int64_t>(te - top_e, 63);
  const int64_t c_part = ((cm * 2) >> min_val<int64_t>(te - ce, 63)) * 256;
  const int64_t total = dot_part + c_part;
  const uint64_t tmag = uint64_t(total < 0 ? -total : total);
  const int64_t ttop = 63 - __builtin_clzll(tmag | 1ULL);
  const int64_t ne = max_val<int64_t>(ttop + te - 32, -126);
  const int64_t right = ne - te + 1;
  const int64_t lsh = clamp_val<int64_t>(-right, 0, 31);
  const int64_t rsh = clamp_val<int64_t>(right, 0, 63);
  const int64_t fixed = int64_t(uint64_t(total) << lsh) >> rsh;
  const uint64_t mag = uint64_t(fixed < 0 ? -fixed : fixed);
  const int64_t tb = 63 - __builtin_clzll(mag | 1ULL);
  const int64_t exp = max_val<int64_t>(tb + ne - 31, -126);
  const int64_t dropped = clamp_val<int64_t>(exp - 23 - ne + 31, 1, 63);
  const uint64_t kept = mag >> dropped;
  const uint64_t rem = mag & ((uint64_t{1} << dropped) - 1);
  const uint64_t half = uint64_t{1} << (dropped - 1);
  const uint64_t rup = (rem > half || (rem == half && (kept & 1))) ? 1 : 0;
  const uint64_t bits = (uint64_t(exp + 126) << 23) + kept + rup;
  const uint64_t bounded = min_val<uint64_t>(bits, 0x7f800000ULL);
  const uint64_t sbits = bounded | (fixed < 0 ? 0x80000000ULL : 0ULL);
  uint32_t res = (mag != 0 && bounded != 0) ? uint32_t(sbits) : 0U;

  if (ai0 || ai1 || bi0 || bi1 || cef == 0xff) {
    bool nan = false, pos = false, neg = false;
    auto add_inf = [&](bool s) { (s ? neg : pos) = true; };
    for (int i = 0; i < 2; ++i) {
      const uint32_t ab = as_bits(a[i]), bb = as_bits(b[i]);
      const bool s = ((ab ^ bb) >> 31) != 0;
      const bool as = ((ab >> 23) & 0xffU) == 0xffU, bs = ((bb >> 23) & 0xffU) == 0xffU;
      const bool an = as && (ab & 0x7fffffU), bn = bs && (bb & 0x7fffffU);
      const bool az = (ab & 0x7fffffffU) == 0, bz = (bb & 0x7fffffffU) == 0;
      if (an || bn || (as && bz) || (bs && az)) {
        nan = true;
      } else if (as || bs) {
        add_inf(s);
      }
    }
    if (cef == 0xff) {
      if (c_bits & 0x7fffffU) {
        nan = true;
      } else {
        add_inf((c_bits >> 31) != 0);
      }
    }
    if (nan || (pos && neg)) {
      res = 0xffc00000U;
    } else if (pos || neg) {
      res = pos ? 0x7f800000U : 0xff800000U;
    }
  }
  return res;
}

} // namespace

namespace {

// Compute selected lanes after validating the complete request. Guest FP modes and
// CLAMP do not affect this inherited model; unsupported exact requests leave D unchanged.
template <bool Bf16, bool Accumulate, typename Mask>
int dot(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d, const uint32_t *const *a,
        const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  [[maybe_unused]] const bool preserve_compatibility =
      (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_LOOSE);
  if (int error = validate(flags, mode & ~(Accumulate ? 0ULL : 0x7ffULL), kMatrixIgnoredFpFlags)) {
    return error;
  }
  if ((flags & RJ_ARITH_SEMANTICS_MASK) > RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE) {
    return RJ_ARITH_ERROR_UNSUPPORTED_SEMANTICS;
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
#if defined(RJ_ARITH_HAVE_X86_64_V4_BF16)
  if constexpr (Bf16) {
    if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_ZEN4) && !preserve_compatibility) {
      dot2_bf16_x86_64_v4_bf16<int(sizeof(Mask) * 8), false>(exec_mask, uint32_t(mode), d[0], a[0],
                                                             b[0], Accumulate ? d[0] : c[0]);
      return RJ_ARITH_SUCCESS;
    }
  }
#endif
  for (int base = 0; base < int(sizeof(Mask) * 8); base += 32) {
    const uint32_t *av = a[0] + base, *bv = b[0] + base;
    uint32_t *dv = d[0] + base;
    const uint32_t *cv = Accumulate ? dv : c[0] + base;
    const uint32_t active = uint32_t(exec_mask >> base);
#if defined(RJ_ARITH_HAVE_X86_64_V4)
    if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V4)) {
      cdna_dot2_x86_64_v4(Bf16, active, uint32_t(mode), dv, av, bv, cv);
      continue;
    }
#endif
#if defined(RJ_ARITH_HAVE_X86_64_V3)
    if (cpu_level_at_least(flags, RJ_ARITH_CPU_X86_64_V3)) {
      cdna_dot2_x86_64_v3(Bf16, active, uint32_t(mode), dv, av, bv, cv);
      continue;
    }
#endif
    const int a0 = mode & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0;
    const int b0 = mode & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0;
    const int a1 = mode & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16;
    const int b1 = mode & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16;
    auto factor = [](uint32_t bits, int shift, bool negate) {
      uint16_t half = uint16_t(bits >> shift) ^ (negate ? 0x8000U : 0U);
      return Bf16 ? bf16_to_float(half) : f16_to_float(half);
    };
    for (int lane = 0; lane < 32; ++lane) {
      if (!((active >> lane) & 1)) {
        continue;
      }
      const float low = factor(av[lane], a0, mode & RJ_ARITH_DOT_NEG_LO_A) *
                        factor(bv[lane], b0, mode & RJ_ARITH_DOT_NEG_LO_B);
      const float high = factor(av[lane], a1, mode & RJ_ARITH_DOT_NEG_HI_A) *
                         factor(bv[lane], b1, mode & RJ_ARITH_DOT_NEG_HI_B);
      const uint32_t acc = (cv[lane] & (mode & RJ_ARITH_DOT_ABS_C ? 0x7fffffffU : ~0U)) ^
                           (mode & RJ_ARITH_DOT_NEG_C ? 0x80000000U : 0U);
      dv[lane] = as_bits((low + high) + as_float(acc));
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2_f32_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *trapsts) {
  return dot<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2_f32_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *trapsts) {
  return dot<false, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2c_f32_f16_w32_gfx942(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot<false, true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2c_f32_f16_w64_gfx942(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot<false, true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2_f32_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2_f32_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        const uint32_t *const *c,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot<true, false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2c_f32_bf16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         [[maybe_unused]] uint32_t *trapsts) {
  return dot<true, true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

// Accumulate two packed products; unsupported requests do not change D or FP status.
int rj_arith_v_dot2c_f32_bf16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                         uint64_t instruction_flags, uint32_t *const *d,
                                         const uint32_t *const *a, const uint32_t *const *b,
                                         [[maybe_unused]] uint32_t *trapsts) {
  return dot<true, true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

namespace {

template <bool Accumulate, typename Mask>
int dot_f16_gfx950(uint64_t flags, Mask exec_mask, uint64_t mode, uint32_t *const *d,
                   const uint32_t *const *a, const uint32_t *const *b, const uint32_t *const *c) {
  using namespace rocjitsu::arith;
  flags = resolve_compatibility(flags, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS);
  if (int error = validate(flags, mode & ~(Accumulate ? 0ULL : 0x7ffULL), kMatrixIgnoredFpFlags)) {
    return error;
  }
  if (reports_fp_status(flags)) {
    return RJ_ARITH_ERROR_UNSUPPORTED_FP_STATUS;
  }
  if (!is_exact(flags)) {
    return dot<false, Accumulate>(flags, exec_mask, mode, d, a, b, c);
  }
  if (!exec_mask) {
    return RJ_ARITH_SUCCESS;
  }
  const int a0 = mode & RJ_ARITH_DOT_LO_A_HIGH ? 16 : 0;
  const int b0 = mode & RJ_ARITH_DOT_LO_B_HIGH ? 16 : 0;
  const int a1 = mode & RJ_ARITH_DOT_HI_A_LOW ? 0 : 16;
  const int b1 = mode & RJ_ARITH_DOT_HI_B_LOW ? 0 : 16;
  for (int base = 0; base < int(sizeof(Mask) * 8); base += 32) {
    const uint32_t *av = a[0] + base, *bv = b[0] + base;
    uint32_t *dv = d[0] + base;
    const uint32_t *cv = Accumulate ? dv : c[0] + base;
    const uint32_t active = uint32_t(exec_mask >> base);
    for (int lane = 0; lane < 32; ++lane) {
      if (!((active >> lane) & 1)) {
        continue;
      }
      const uint32_t aw = av[lane], bw = bv[lane], cw = cv[lane];
      const uint16_t lo_a = uint16_t(aw >> a0) ^ (mode & RJ_ARITH_DOT_NEG_LO_A ? 0x8000U : 0U);
      const uint16_t lo_b = uint16_t(bw >> b0) ^ (mode & RJ_ARITH_DOT_NEG_LO_B ? 0x8000U : 0U);
      const uint16_t hi_a = uint16_t(aw >> a1) ^ (mode & RJ_ARITH_DOT_NEG_HI_A ? 0x8000U : 0U);
      const uint16_t hi_b = uint16_t(bw >> b1) ^ (mode & RJ_ARITH_DOT_NEG_HI_B ? 0x8000U : 0U);
      const uint32_t acc = (cw & (mode & RJ_ARITH_DOT_ABS_C ? 0x7fffffffU : ~0U)) ^
                           (mode & RJ_ARITH_DOT_NEG_C ? 0x80000000U : 0U);
      dv[lane] = dot2_f16_gfx950_exact(lo_a, lo_b, hi_a, hi_b, acc);
    }
  }
  return RJ_ARITH_SUCCESS;
}

} // namespace

int rj_arith_v_dot2_f32_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *trapsts) {
  return dot_f16_gfx950<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int rj_arith_v_dot2_f32_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                       uint64_t instruction_flags, uint32_t *const *d,
                                       const uint32_t *const *a, const uint32_t *const *b,
                                       const uint32_t *const *c,
                                       [[maybe_unused]] uint32_t *trapsts) {
  return dot_f16_gfx950<false>(flags, exec_mask, instruction_flags, d, a, b, c);
}

int rj_arith_v_dot2c_f32_f16_w32_gfx950(uint64_t flags, uint32_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot_f16_gfx950<true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}

int rj_arith_v_dot2c_f32_f16_w64_gfx950(uint64_t flags, uint64_t exec_mask,
                                        uint64_t instruction_flags, uint32_t *const *d,
                                        const uint32_t *const *a, const uint32_t *const *b,
                                        [[maybe_unused]] uint32_t *trapsts) {
  return dot_f16_gfx950<true>(flags, exec_mask, instruction_flags, d, a, b, nullptr);
}
