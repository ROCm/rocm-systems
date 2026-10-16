// SPDX-License-Identifier: MIT

// AVX2 signed/unsigned two-input and packed 16-bit min/max.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/integer_minmax.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Select each lane's minimum using the template-selected signedness.
template <bool Signed> __m256i minimum(__m256i a, __m256i b) {
  if constexpr (Signed) {
    return _mm256_min_epi32(a, b);
  } else {
    return _mm256_min_epu32(a, b);
  }
}

// Select each lane's maximum using the template-selected signedness.
template <bool Signed> __m256i maximum(__m256i a, __m256i b) {
  if constexpr (Signed) {
    return _mm256_max_epi32(a, b);
  } else {
    return _mm256_max_epu32(a, b);
  }
}

// Select integer minima or maxima for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <IntegerMinmax Op, bool Signed>
void run(uint32_t exec_mask, uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    __m256i z = _mm256_setzero_si256(), value;
    if constexpr (Op != IntegerMinmax::Min && Op != IntegerMinmax::Max) {
      z = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
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
    if constexpr (Op == IntegerMinmax::Median) {
      value = maximum<Signed>(minimum<Signed>(maximum<Signed>(x, y), z), minimum<Signed>(x, y));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, value);
  }
}

} // namespace

// Select integer minima or maxima. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <IntegerMinmax Op, bool Signed>
void integer_minmax_x86_64_v3(uint32_t exec_mask, uint32_t *d, const uint32_t *a, const uint32_t *b,
                              const uint32_t *c) {
  run<Op, Signed>(exec_mask, d, a, b, c);
}

template void integer_minmax_x86_64_v3<IntegerMinmax::Min, false>(uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Min, true>(uint32_t, uint32_t *,
                                                                 const uint32_t *, const uint32_t *,
                                                                 const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Max, false>(uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Max, true>(uint32_t, uint32_t *,
                                                                 const uint32_t *, const uint32_t *,
                                                                 const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Min3, false>(uint32_t, uint32_t *,
                                                                   const uint32_t *,
                                                                   const uint32_t *,
                                                                   const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Min3, true>(uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Max3, false>(uint32_t, uint32_t *,
                                                                   const uint32_t *,
                                                                   const uint32_t *,
                                                                   const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Max3, true>(uint32_t, uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *,
                                                                  const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Minmax, false>(uint32_t, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Minmax, true>(uint32_t, uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Maxmin, false>(uint32_t, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Maxmin, true>(uint32_t, uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Median, false>(uint32_t, uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *,
                                                                     const uint32_t *);
template void integer_minmax_x86_64_v3<IntegerMinmax::Median, true>(uint32_t, uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *,
                                                                    const uint32_t *);

} // namespace rocjitsu::arith
