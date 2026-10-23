// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 conversions between FP64 and FP32 or 32-bit integers in split VGPR layouts.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/alu_x86_64_v3.h"
#include "common/x86_64/alu_x86_64_v4.h"
#include "conversion/conversion64.h"
#include "conversion/float_conversion.h"
#include "rj_arith/rj_arith.h"
#include <experimental/simd>

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {
// Eight unsigned lanes for the shared conversion model. Use native vector operators
// here: Clang with libstdc++ 15 cannot instantiate experimental::simd<uint64_t>'s
// AVX-512 comparison masks (its comparison result uses a different signed type).
class ConversionLanes64 {
public:
  using value_type = uint64_t;
  using Bits = uint64_t __attribute__((vector_size(64)));
  Bits bits;

  ConversionLanes64() = default;
  ConversionLanes64(uint64_t x) : bits{x, x, x, x, x, x, x, x} {}
  explicit ConversionLanes64(Bits x) : bits(x) {}

  // Arithmetic and bit operations act independently in each lane.
  friend ConversionLanes64 operator+(ConversionLanes64 a, ConversionLanes64 b) {
    return ConversionLanes64(a.bits + b.bits);
  }
  friend ConversionLanes64 operator-(ConversionLanes64 a, ConversionLanes64 b) {
    return ConversionLanes64(a.bits - b.bits);
  }
  friend ConversionLanes64 operator&(ConversionLanes64 a, ConversionLanes64 b) {
    return ConversionLanes64(a.bits & b.bits);
  }
  friend ConversionLanes64 operator|(ConversionLanes64 a, ConversionLanes64 b) {
    return ConversionLanes64(a.bits | b.bits);
  }
  friend ConversionLanes64 operator^(ConversionLanes64 a, ConversionLanes64 b) {
    return ConversionLanes64(a.bits ^ b.bits);
  }
  friend ConversionLanes64 operator~(ConversionLanes64 a) { return ConversionLanes64(~a.bits); }
  friend ConversionLanes64 operator<<(ConversionLanes64 a, ConversionLanes64 b) {
    return ConversionLanes64(a.bits << b.bits);
  }
  friend ConversionLanes64 operator>>(ConversionLanes64 a, ConversionLanes64 b) {
    return ConversionLanes64(a.bits >> b.bits);
  }
  ConversionLanes64 &operator|=(ConversionLanes64 b) {
    bits |= b.bits;
    return *this;
  }

  // Comparisons yield all-one or all-zero lanes, also used as Boolean masks.
  friend ConversionLanes64 operator==(ConversionLanes64 a, ConversionLanes64 b) {
    return ConversionLanes64(std::bit_cast<Bits>(a.bits == b.bits));
  }
  friend ConversionLanes64 operator!=(ConversionLanes64 a, ConversionLanes64 b) {
    return ~(a == b);
  }
  friend ConversionLanes64 operator<(ConversionLanes64 a, ConversionLanes64 b) {
    return ConversionLanes64(std::bit_cast<Bits>(a.bits < b.bits));
  }
  friend ConversionLanes64 operator>(ConversionLanes64 a, ConversionLanes64 b) { return b < a; }
  friend ConversionLanes64 operator>=(ConversionLanes64 a, ConversionLanes64 b) { return ~(a < b); }
  friend ConversionLanes64 operator!(ConversionLanes64 a) { return ~a; }
  friend ConversionLanes64 operator&&(ConversionLanes64 a, ConversionLanes64 b) { return a & b; }
  friend ConversionLanes64 operator||(ConversionLanes64 a, ConversionLanes64 b) { return a | b; }
};

// Select raw lanes without depending on a standard-library SIMD mask ABI.
class SelectedLanes64 {
public:
  SelectedLanes64(ConversionLanes64 mask, ConversionLanes64 &destination)
      : mask_(mask), destination_(destination) {}
  void operator=(ConversionLanes64 value) {
    destination_ = (value & mask_) | (destination_ & ~mask_);
  }

private:
  ConversionLanes64 mask_;
  ConversionLanes64 &destination_;
};

// ADL hook used by the shared scalar/SIMD selection helper.
SelectedLanes64 where(ConversionLanes64 mask, ConversionLanes64 &destination) {
  return {mask, destination};
}
} // namespace

namespace {

// Extract each 64-bit lane's low 32-bit word into consecutive lanes.
__m256i low_words(__m512i words) { return _mm512_cvtepi64_epi32(words); }

} // namespace

// Convert FP64, FP32, or 32-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion64 Op>
void conversion64_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                            const uint32_t *const *a, bool quiet_nan) {
  constexpr bool from_integer =
      Op == Conversion64::SignedToDouble || Op == Conversion64::UnsignedToDouble;
  constexpr bool to_double = from_integer || Op == Conversion64::FloatToDouble;
  if constexpr (to_double || Op == Conversion64::DoubleToFloat) {
    namespace sx = std::experimental;
    using V = ConversionLanes64;
    uint32_t result[to_double ? 2 : 1][32];
    for (int lane = 0; lane < 32; lane += 8) {
      V raw(std::bit_cast<V::Bits>(_mm512_cvtepu32_epi64(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane)))));
      if constexpr (!to_double) {
        raw |= V(std::bit_cast<V::Bits>(_mm512_cvtepu32_epi64(
                   _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[1] + lane)))))
               << 32;
      }
      V bits = conversion_detail::convert64<Op>(raw, flags, mode, quiet_nan);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane),
                          _mm512_cvtepi64_epi32(std::bit_cast<__m512i>(bits.bits)));
      if constexpr (to_double) {
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                            _mm512_cvtepi64_epi32(std::bit_cast<__m512i>((bits >> 32).bits)));
      }
    }
    // Snapshot the wave before storing: either destination may alias either source VGPR.
    using W = sx::native_simd<uint32_t>;
    for (int reg = 0; reg < (to_double ? 2 : 1); ++reg) {
      for (int lane = 0; lane < 32; lane += W::size()) {
        W value(result[reg] + lane, sx::element_aligned);
        W indices([](auto i) { return uint32_t(i); });
        auto active = ((W(exec_mask >> lane) >> indices) & W(1U)) != W(0U);
        sx::where(active, value).copy_to(d[reg] + lane, sx::element_aligned);
      }
    }
    return;
  }
  const auto zero = _mm512_setzero_pd();
  const double scales[] = {1, 2, 4, 0.5};
  uint32_t result[to_double ? 2 : 1][32];
  for (int lane = 0; lane < 32; lane += 8) {
    auto low = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[0] + lane));
    __m512d value;
    if constexpr (from_integer) {
      if constexpr (Op == Conversion64::SignedToDouble) {
        value = _mm512_cvtepi32_pd(low);
      } else {
        value = _mm512_cvtepu32_pd(low);
      }
    } else if constexpr (Op == Conversion64::FloatToDouble) {
      low = _mm256_and_si256(low, _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1));
      low = _mm256_xor_si256(low, _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0));
      value = _mm512_cvtps_pd(_mm256_castsi256_ps(low));
    } else {
      auto high = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a[1] + lane));
      auto raw = _mm512_or_si512(_mm512_cvtepu32_epi64(low),
                                 _mm512_slli_epi64(_mm512_cvtepu32_epi64(high), 32));
      raw = _mm512_and_si512(raw, _mm512_set1_epi64(mode & RJ_ARITH_ALU_ABS_A ? INT64_MAX : -1));
      raw = _mm512_xor_si512(raw, _mm512_set1_epi64(mode & RJ_ARITH_ALU_NEG_A ? INT64_MIN : 0));
      value = _mm512_castsi512_pd(raw);
    }
    if constexpr (to_double) {
      if (mode & RJ_ARITH_ALU_OMOD_HALF) {
        value = prepare_omod_f64(value, mode);
        value = _mm512_mul_pd(value, _mm512_set1_pd(scales[(mode >> 6) & 3]));
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        value = _mm512_min_pd(_mm512_max_pd(value, zero), _mm512_set1_pd(1));
      }
      auto raw = _mm512_castpd_si512(value);
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane), low_words(raw));
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[1] + lane),
                          low_words(_mm512_srli_epi64(raw, 32)));
    } else {
      __m256i output;
      if constexpr (Op == Conversion64::DoubleToFloat) {
        auto x = _mm512_cvtpd_ps(value);
        if (mode & RJ_ARITH_ALU_OMOD_HALF) {
          auto magnitude = _mm512_castsi512_pd(
              _mm512_and_si512(_mm512_castpd_si512(value), _mm512_set1_epi64(INT64_MAX)));
          auto tiny = _mm512_cmp_pd_mask(magnitude, _mm512_set1_pd(0x1p-126), _CMP_LT_OQ);
          x = _mm256_mask_mov_ps(x, tiny, _mm256_setzero_ps());
          x = prepare_omod_f32(x, mode);
          x = _mm256_mul_ps(x, _mm256_set1_ps(float(scales[(mode >> 6) & 3])));
        }
        if (mode & RJ_ARITH_ALU_CLAMP) {
          x = _mm256_min_ps(_mm256_max_ps(x, _mm256_setzero_ps()), _mm256_set1_ps(1));
        }
        output = _mm256_castps_si256(x);
      } else if constexpr (Op == Conversion64::DoubleToUnsigned) {
        value = _mm512_min_pd(_mm512_max_pd(value, zero), _mm512_set1_pd(4294967295.0));
        output = _mm512_cvttpd_epu32(value);
      } else {
        auto nan = _mm512_cmp_pd_mask(value, value, _CMP_UNORD_Q);
        value = _mm512_min_pd(_mm512_max_pd(value, _mm512_set1_pd(-2147483648.0)),
                              _mm512_set1_pd(2147483647.0));
        output = _mm512_cvttpd_epi32(value);
        output = _mm256_mask_mov_epi32(output, nan, _mm256_setzero_si256());
      }
      _mm256_storeu_si256(reinterpret_cast<__m256i *>(result[0] + lane), output);
    }
  }
  // Delay stores to allow cross-half source/destination aliases. Store D0
  // before D1, so D1 wins when both destination halves share storage.
  for (int reg = 0; reg < (to_double ? 2 : 1); ++reg) {
    for (int lane = 0; lane < 32; lane += 16) {
      _mm512_mask_storeu_epi32(d[reg] + lane, __mmask16(exec_mask >> lane),
                               _mm512_loadu_si512(result[reg] + lane));
    }
  }
}

template void conversion64_x86_64_v4<Conversion64::SignedToDouble>(uint64_t, uint32_t, uint32_t,
                                                                   uint32_t *const *,
                                                                   const uint32_t *const *, bool);
template void conversion64_x86_64_v4<Conversion64::UnsignedToDouble>(uint64_t, uint32_t, uint32_t,
                                                                     uint32_t *const *,
                                                                     const uint32_t *const *, bool);
template void conversion64_x86_64_v4<Conversion64::DoubleToSigned>(uint64_t, uint32_t, uint32_t,
                                                                   uint32_t *const *,
                                                                   const uint32_t *const *, bool);
template void conversion64_x86_64_v4<Conversion64::DoubleToUnsigned>(uint64_t, uint32_t, uint32_t,
                                                                     uint32_t *const *,
                                                                     const uint32_t *const *, bool);
template void conversion64_x86_64_v4<Conversion64::FloatToDouble>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *const *,
                                                                  const uint32_t *const *, bool);
template void conversion64_x86_64_v4<Conversion64::DoubleToFloat>(uint64_t, uint32_t, uint32_t,
                                                                  uint32_t *const *,
                                                                  const uint32_t *const *, bool);

} // namespace rocjitsu::arith
