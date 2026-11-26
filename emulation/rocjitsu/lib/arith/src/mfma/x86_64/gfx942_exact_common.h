// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared compile-time metadata and cold non-finite fallback helpers for x86-64
// SIMD implementations of captured gfx942 exact MFMA and SMFMAC.
#pragma once
#include "common/internal.h"
#include "mfma/gfx942_exact.h"
#include "mfma/sparse_layout.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

enum class Gfx942Elem { F16, Bf16, Xf32, Fp8Fnuz, Bf8Fnuz };

template <MfmaGfx942Format Format> constexpr Gfx942Elem gfx942_elem_a() {
  if constexpr (Format == MfmaGfx942Format::F16) {
    return Gfx942Elem::F16;
  } else if constexpr (Format == MfmaGfx942Format::Bf16) {
    return Gfx942Elem::Bf16;
  } else if constexpr (Format == MfmaGfx942Format::Xf32) {
    return Gfx942Elem::Xf32;
  } else if constexpr (Format == MfmaGfx942Format::Fp8Fp8 || Format == MfmaGfx942Format::Fp8Bf8) {
    return Gfx942Elem::Fp8Fnuz;
  } else {
    return Gfx942Elem::Bf8Fnuz;
  }
}

template <MfmaGfx942Format Format> constexpr Gfx942Elem gfx942_elem_b() {
  if constexpr (Format == MfmaGfx942Format::F16) {
    return Gfx942Elem::F16;
  } else if constexpr (Format == MfmaGfx942Format::Bf16) {
    return Gfx942Elem::Bf16;
  } else if constexpr (Format == MfmaGfx942Format::Xf32) {
    return Gfx942Elem::Xf32;
  } else if constexpr (Format == MfmaGfx942Format::Fp8Fp8 || Format == MfmaGfx942Format::Bf8Fp8) {
    return Gfx942Elem::Fp8Fnuz;
  } else {
    return Gfx942Elem::Bf8Fnuz;
  }
}

template <MfmaGfx942Format Format> constexpr int gfx942_per_word() {
  if constexpr (Format == MfmaGfx942Format::Xf32) {
    return 1;
  } else if constexpr (Format == MfmaGfx942Format::F16 || Format == MfmaGfx942Format::Bf16) {
    return 2;
  } else {
    return 4;
  }
}

template <MfmaGfx942Format Format> constexpr int gfx942_products_per_step() {
  if constexpr (Format == MfmaGfx942Format::Xf32) {
    return 4;
  } else if constexpr (Format == MfmaGfx942Format::F16 || Format == MfmaGfx942Format::Bf16) {
    return 8;
  } else {
    return 16;
  }
}

template <MfmaGfx942Format Format> constexpr bool gfx942_even_odd() {
  return Format != MfmaGfx942Format::F16 && Format != MfmaGfx942Format::Bf16 &&
         Format != MfmaGfx942Format::Xf32;
}

template <MfmaGfx942Format Format> constexpr bool gfx942_can_product_overflow() {
  return Format == MfmaGfx942Format::Bf16 || Format == MfmaGfx942Format::Xf32;
}

inline constexpr int32_t kGfx942ZeroProductExponent = -999;
inline constexpr uint32_t kGfx942PosInf = 0x7f800000u;
inline constexpr uint32_t kGfx942NegInf = 0xff800000u;
inline constexpr uint32_t kGfx942Nan = 0xffc00000u;

struct Gfx942Classified {
  bool is_nan;
  bool is_inf;
  bool is_zero;
  bool sign;
  int32_t mag;
  int32_t exp;
};

template <Gfx942Elem Elem> inline Gfx942Classified classify_gfx942_raw(uint32_t raw) {
  if constexpr (Elem == Gfx942Elem::F16) {
    const uint32_t u = raw & 0xffffu;
    const uint32_t e = (u >> 10) & 0x1fu;
    const uint32_t f = u & 0x3ffu;
    const bool special = e == 31u;
    return {special && f != 0u, special && f == 0u, (u & 0x7fffu) == 0u, (u >> 15) != 0u, 0, 0};
  } else if constexpr (Elem == Gfx942Elem::Bf16) {
    const uint32_t u = raw & 0xffffu;
    const uint32_t e = (u >> 7) & 0xffu;
    const uint32_t f = u & 0x7fu;
    const bool special = e == 0xffu;
    const int32_t mag = static_cast<int32_t>(((e != 0u ? 0x80u : 0u) | f) << 3);
    const int32_t exp = (e != 0u ? static_cast<int32_t>(e) : 1) - 127;
    return {special && f != 0u, special && f == 0u, (u & 0x7fffu) == 0u, (u >> 15) != 0u, mag, exp};
  } else if constexpr (Elem == Gfx942Elem::Xf32) {
    const uint32_t u = raw & 0xffffe000u;
    const uint32_t e = (u >> 23) & 0xffu;
    const uint32_t f = (u >> 13) & 0x3ffu;
    const bool special = e == 0xffu;
    const int32_t mag = static_cast<int32_t>((e != 0u ? 0x400u : 0u) | f);
    const int32_t exp = (e != 0u ? static_cast<int32_t>(e) : 1) - 127;
    return {
        special && f != 0u, special && f == 0u, (u & 0x7fffffffu) == 0u, (u >> 31) != 0u, mag, exp};
  } else {
    const uint32_t u = raw & 0xffu;
    return {u == 0x80u, false, u == 0u, (u >> 7) != 0u, 0, 0};
  }
}

template <MfmaGfx942Format Format>
inline uint32_t read_raw_elem(const uint32_t *const *source, int lane, int local) {
  constexpr int per_word = gfx942_per_word<Format>();
  const uint32_t word = source[local / per_word][lane];
  if constexpr (per_word == 1) {
    return word;
  } else if constexpr (per_word == 2) {
    return (word >> (16 * (local & 1))) & 0xffffu;
  } else {
    return (word >> (8 * (local & 3))) & 0xffu;
  }
}

template <MfmaGfx942Format Format, int M, int K, int Batch>
[[gnu::noinline]] inline void gfx942_non_finite_dense_one(int batch, int row, int col, int start,
                                                          int products, const uint32_t *const *a,
                                                          const uint32_t *const *b, uint32_t c_bits,
                                                          uint32_t &result) {
  constexpr int per_lane = K * M * Batch / 64;
  constexpr Gfx942Elem elem_a = gfx942_elem_a<Format>();
  constexpr Gfx942Elem elem_b = gfx942_elem_b<Format>();
  bool nan = false;
  bool positive = false;
  bool negative = false;
  const auto add_inf = [&](bool sign) { (sign ? negative : positive) = true; };

  for (int i = 0; i < products; ++i) {
    const int depth = start + i;
    const int lane_base = batch * M + (depth / per_lane) * M * Batch;
    const int local = depth % per_lane;
    const Gfx942Classified ca =
        classify_gfx942_raw<elem_a>(read_raw_elem<Format>(a, lane_base + row, local));
    const Gfx942Classified cb =
        classify_gfx942_raw<elem_b>(read_raw_elem<Format>(b, lane_base + col, local));
    const bool sign = ca.sign != cb.sign;
    const bool a_spec = ca.is_nan || ca.is_inf;
    const bool b_spec = cb.is_nan || cb.is_inf;
    if (ca.is_nan || cb.is_nan || (a_spec && cb.is_zero) || (b_spec && ca.is_zero)) {
      nan = true;
    } else if (a_spec || b_spec) {
      add_inf(sign);
    } else if constexpr (gfx942_can_product_overflow<Format>()) {
      if (!ca.is_zero && !cb.is_zero) {
        const int32_t prod_mag = ca.mag * cb.mag;
        if (prod_mag != 0 && ca.exp + cb.exp + (prod_mag >> 21) >= 128) {
          add_inf(sign);
        }
      }
    }
  }
  if (((c_bits >> 23) & 0xffu) == 0xffu) {
    if ((c_bits & 0x7fffffu) != 0u) {
      nan = true;
    } else {
      add_inf((c_bits >> 31) != 0u);
    }
  }
  if (nan || (positive && negative)) {
    result = kGfx942Nan;
  } else if (positive || negative) {
    result = positive ? kGfx942PosInf : kGfx942NegInf;
  }
}

template <MfmaGfx942Format Format, int M, int K>
[[gnu::noinline]] inline void
gfx942_non_finite_sparse_one(int row, int col, int start, int products, const uint32_t *const *a,
                             const uint32_t *const *b, const uint32_t *selected_index,
                             uint32_t c_bits, uint32_t &result) {
  constexpr bool half = Format == MfmaGfx942Format::F16 || Format == MfmaGfx942Format::Bf16;
  constexpr SmfmacLayout layout = half ? SmfmacLayout::Cdna3F16 : SmfmacLayout::Cdna3Fp8;
  constexpr Gfx942Elem elem_a = gfx942_elem_a<Format>();
  constexpr Gfx942Elem elem_b = gfx942_elem_b<Format>();
  bool nan = false;
  bool positive = false;
  bool negative = false;
  const auto add_inf = [&](bool sign) { (sign ? negative : positive) = true; };

  for (int i = 0; i < products; ++i) {
    const int t = start + i;
    const int q_log = t / 2;
    const int q =
        (M == 16 && K == 64) ? ((q_log & 3) + ((q_log & 4) << 1) + ((q_log & 8) >> 1)) : q_log;
    const int s = t & 1;
    const SmfmacSparseLoc sloc = smfmac_sparse_loc<layout, M, K>(row, q, s);
    const uint32_t raw_a = read_raw_elem<Format>(a, sloc.lane, sloc.element);
    const int sel = 4 * q + int((selected_index[sloc.lane] >> (4 * sloc.nibble + 2 * s)) & 3u);
    const SmfmacDenseLoc dloc = smfmac_dense_loc<layout, M>(sel, col);
    const uint32_t raw_b = read_raw_elem<Format>(b, dloc.lane, dloc.element);
    const Gfx942Classified ca = classify_gfx942_raw<elem_a>(raw_a);
    const Gfx942Classified cb = classify_gfx942_raw<elem_b>(raw_b);
    const bool sign = ca.sign != cb.sign;
    const bool a_spec = ca.is_nan || ca.is_inf;
    const bool b_spec = cb.is_nan || cb.is_inf;
    if (ca.is_nan || cb.is_nan || (a_spec && cb.is_zero) || (b_spec && ca.is_zero)) {
      nan = true;
    } else if (a_spec || b_spec) {
      add_inf(sign);
    } else if constexpr (gfx942_can_product_overflow<Format>()) {
      if (!ca.is_zero && !cb.is_zero) {
        const int32_t prod_mag = ca.mag * cb.mag;
        if (prod_mag != 0 && ca.exp + cb.exp + (prod_mag >> 21) >= 128) {
          add_inf(sign);
        }
      }
    }
  }
  if (((c_bits >> 23) & 0xffu) == 0xffu) {
    if ((c_bits & 0x7fffffu) != 0u) {
      nan = true;
    } else {
      add_inf((c_bits >> 31) != 0u);
    }
  }
  if (nan || (positive && negative)) {
    result = kGfx942Nan;
  } else if (positive || negative) {
    result = positive ? kGfx942PosInf : kGfx942NegInf;
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
