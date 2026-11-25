// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 signed/unsigned 32-bit selection, including saturating add-then-select.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer_minmax.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Select each lane's minimum using the template-selected signedness.
template <bool Signed> __m512i minimum(__m512i a, __m512i b) {
  if constexpr (Signed) {
    return _mm512_min_epi32(a, b);
  } else {
    return _mm512_min_epu32(a, b);
  }
}

// Select each lane's maximum using the template-selected signedness.
template <bool Signed> __m512i maximum(__m512i a, __m512i b) {
  if constexpr (Signed) {
    return _mm512_max_epi32(a, b);
  } else {
    return _mm512_max_epu32(a, b);
  }
}

// Saturate signed or unsigned sums to their 32-bit range before min/max selection.
template <bool Signed> __m512i saturating_sum(__m512i a, __m512i b) {
  auto sum = _mm512_add_epi32(a, b);
  if constexpr (Signed) {
    auto overflow_bits = _mm512_and_si512(_mm512_xor_si512(sum, a), _mm512_xor_si512(sum, b));
    auto limit = _mm512_xor_si512(_mm512_srai_epi32(a, 31), _mm512_set1_epi32(0x7fffffff));
    return _mm512_mask_mov_epi32(sum, _mm512_movepi32_mask(overflow_bits), limit);
  } else {
    return _mm512_mask_set1_epi32(sum, _mm512_cmp_epu32_mask(sum, a, _MM_CMPINT_LT), -1);
  }
}

// Select integer minima or maxima for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <IntegerMinmax Op, bool Signed>
void run(uint32_t exec_mask, bool clamp, uint32_t *d, const uint32_t *a, const uint32_t *b,
         const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_loadu_si512(a + lane);
    auto y = _mm512_loadu_si512(b + lane);
    __m512i z = _mm512_setzero_si512(), value;
    if constexpr (Op != IntegerMinmax::Min && Op != IntegerMinmax::Max) {
      z = _mm512_loadu_si512(c + lane);
    }
    if constexpr (Op == IntegerMinmax::Min) {
      value = minimum<Signed>(x, y);
    }
    if constexpr (Op == IntegerMinmax::Max) {
      value = maximum<Signed>(x, y);
    }
    if constexpr (Op == IntegerMinmax::Min3) {
      value = minimum<Signed>(minimum<Signed>(x, y), z);
    }
    if constexpr (Op == IntegerMinmax::Max3) {
      value = maximum<Signed>(maximum<Signed>(x, y), z);
    }
    if constexpr (Op == IntegerMinmax::Minmax) {
      value = maximum<Signed>(minimum<Signed>(x, y), z);
    }
    if constexpr (Op == IntegerMinmax::Maxmin) {
      value = minimum<Signed>(maximum<Signed>(x, y), z);
    }
    if constexpr (Op == IntegerMinmax::AddMin) {
      value = minimum<Signed>(saturating_sum<Signed>(x, y), z);
    }
    if constexpr (Op == IntegerMinmax::AddMax) {
      value = maximum<Signed>(saturating_sum<Signed>(x, y), z);
    }
    if constexpr (Op == IntegerMinmax::Median) {
      value = maximum<Signed>(minimum<Signed>(maximum<Signed>(x, y), z), minimum<Signed>(x, y));
    }
    if constexpr (Signed && (Op == IntegerMinmax::AddMin || Op == IntegerMinmax::AddMax)) {
      if (clamp) {
        value = _mm512_max_epi32(value, _mm512_setzero_si512());
      }
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), value);
  }
}

} // namespace

// Select integer minima or maxima. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerMinmax Op, bool Signed>
void integer_minmax_x86_64_v4(uint32_t exec_mask, bool clamp, uint32_t *d, const uint32_t *a,
                              const uint32_t *b, const uint32_t *c) {
  run<Op, Signed>(exec_mask, clamp, d, a, b, c);
}

template void integer_minmax_x86_64_v4<IntegerMinmax::Min, false>(uint32_t, bool, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Min, true>(uint32_t, bool, uint32_t *,
                                                                 const uint32_t *, const uint32_t *,
                                                                 const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Max, false>(uint32_t, bool, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Max, true>(uint32_t, bool, uint32_t *,
                                                                 const uint32_t *, const uint32_t *,
                                                                 const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Min3, false>(uint32_t, bool, uint32_t *,
                                                                   const uint32_t *,
                                                                   const uint32_t *,
                                                                   const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Min3, true>(uint32_t, bool, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Max3, false>(uint32_t, bool, uint32_t *,
                                                                   const uint32_t *,
                                                                   const uint32_t *,
                                                                   const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Max3, true>(uint32_t, bool, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Minmax, false>(uint32_t, bool, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Minmax, true>(uint32_t, bool, uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Maxmin, false>(uint32_t, bool, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Maxmin, true>(uint32_t, bool, uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Median, false>(uint32_t, bool, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::Median, true>(uint32_t, bool, uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *);

template void integer_minmax_x86_64_v4<IntegerMinmax::AddMin, false>(uint32_t, bool, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::AddMin, true>(uint32_t, bool, uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::AddMax, false>(uint32_t, bool, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
template void integer_minmax_x86_64_v4<IntegerMinmax::AddMax, true>(uint32_t, bool, uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *);

} // namespace rocjitsu::arith
