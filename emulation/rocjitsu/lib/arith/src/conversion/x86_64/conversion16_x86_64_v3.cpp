// SPDX-License-Identifier: MIT

// AVX2 conversions between FP16 and FP32 or 16-bit integers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/conversion16.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Widen low-halfword FP16 encodings to FP32 vector lanes.
__m256 widen(__m256i halves) {
  return _mm256_cvtph_ps(
      _mm_packus_epi32(_mm256_castsi256_si128(halves), _mm256_extracti128_si256(halves, 1)));
}

// Round FP32 vector lanes to FP16, optionally saturating finite overflow.
__m256i narrow(__m256 x, bool saturate) {
  auto result =
      _mm256_cvtepu16_epi32(_mm256_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  if (saturate) {
    auto magnitude = _mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(INT32_MAX));
    auto finite = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude);
    auto infinity = _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                       _mm256_set1_epi32(0x7c00));
    result = _mm256_add_epi32(result, _mm256_and_si256(finite, infinity));
  }
  return result;
}

// Convert FP16, FP32, or 16-bit integers for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <Conversion16 Op, bool GuestWidening = false>
void convert(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
             uint64_t flags = 0) {
  constexpr bool from_integer =
      Op == Conversion16::SignedToHalf || Op == Conversion16::UnsignedToHalf;
  constexpr bool to_half = from_integer || Op == Conversion16::FloatToHalf;
  auto sa = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
  auto sd = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  const float scales[] = {1, 2, 4, 0.5f};
  unsigned omod = (mode >> 6) & 3;
  auto scale = _mm256_set1_ps(scales[omod]);
  const auto zero = _mm256_setzero_ps();
  for (int lane = 0; lane < 32; lane += 8) {
    auto raw = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    if constexpr (Op != Conversion16::FloatToHalf) {
      raw = _mm256_and_si256(_mm256_srl_epi32(raw, sa), _mm256_set1_epi32(65535));
    }
    __m256 value;
    if constexpr (from_integer) {
      if constexpr (Op == Conversion16::SignedToHalf) {
        raw = _mm256_srai_epi32(_mm256_slli_epi32(raw, 16), 16);
      }
      value = _mm256_cvtepi32_ps(raw);
    } else {
      constexpr uint32_t sign = Op == Conversion16::FloatToHalf ? 0x80000000 : 0x8000;
      raw = _mm256_and_si256(
          raw, _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_A ? ~sign : UINT32_MAX)));
      raw = _mm256_xor_si256(raw, _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_A ? sign : 0)));
      if constexpr (Op == Conversion16::FloatToHalf) {
        value = _mm256_castsi256_ps(raw);
      } else {
        if constexpr (Op == Conversion16::HalfToFloat && GuestWidening) {
          if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
            auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x400),
                                           _mm256_and_si256(raw, _mm256_set1_epi32(0x7fff)));
            raw = _mm256_blendv_epi8(raw, _mm256_and_si256(raw, _mm256_set1_epi32(0x8000)), tiny);
          }
        }
        value = widen(raw);
        if constexpr (Op == Conversion16::HalfToFloat && GuestWidening) {
          // F16C quiets NaNs; restore the raw encoding for RDNA3 IEEE=0.
          if (!(flags & RJ_ARITH_FP_IEEE_MODE)) {
            auto nan = _mm256_cmpgt_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(0x7fff)),
                                          _mm256_set1_epi32(0x7c00));
            auto bits = _mm256_or_si256(
                _mm256_set1_epi32(0x7f800000),
                _mm256_or_si256(
                    _mm256_slli_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(0x8000)), 16),
                    _mm256_slli_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(0x3ff)), 13)));
            value = _mm256_castsi256_ps(_mm256_blendv_epi8(_mm256_castps_si256(value), bits, nan));
          }
        }
      }
    }
    __m256i result;
    if constexpr (to_half) {
      result = narrow(value, saturate);
      if (omod) {
        auto magnitude = _mm256_and_si256(result, _mm256_set1_epi32(0x7fff));
        auto boundary = _mm256_cmpeq_epi32(magnitude, _mm256_set1_epi32(0x400));
        auto below = _mm256_cmpgt_epi32(
            _mm256_set1_epi32(0x387ff000),
            _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(INT32_MAX)));
        auto tiny = _mm256_or_si256(_mm256_cmpgt_epi32(_mm256_set1_epi32(0x400), magnitude),
                                    _mm256_and_si256(boundary, below));
        auto initial = _mm256_andnot_si256(tiny, result);
        result = narrow(_mm256_mul_ps(widen(initial), scale), saturate);
        if (omod == 3) {
          auto underflow = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800), magnitude);
          result = _mm256_blendv_epi8(result, _mm256_and_si256(initial, _mm256_set1_epi32(0x8000)),
                                      underflow);
        }
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        // Positive NaNs and every negative encoding clamp to +0.
        auto discard = _mm256_cmpgt_epi32(result, _mm256_set1_epi32(0x7c00));
        result = _mm256_min_epi32(_mm256_andnot_si256(discard, result), _mm256_set1_epi32(0x3c00));
      }
    } else if constexpr (Op == Conversion16::HalfToFloat) {
      const auto input_bits = _mm256_castps_si256(value);
      if (!GuestWidening || omod || (flags & RJ_ARITH_FP_IEEE_MODE)) {
        value = _mm256_mul_ps(value, scale);
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        value = _mm256_min_ps(_mm256_max_ps(value, zero), _mm256_set1_ps(1));
      }
      if (omod) {
        value = _mm256_andnot_ps(_mm256_cmp_ps(value, zero, _CMP_EQ_OQ), value);
      }
      result = _mm256_castps_si256(value);
      if constexpr (GuestWidening) {
        // Preserve the widening NaN policy through OMOD and DX10_CLAMP-off.
        if (!(mode & RJ_ARITH_ALU_CLAMP) || (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
          auto nan = _mm256_cmpgt_epi32(_mm256_and_si256(input_bits, _mm256_set1_epi32(0x7fffffff)),
                                        _mm256_set1_epi32(0x7f800000));
          result = _mm256_blendv_epi8(result, input_bits, nan);
        }
      }
    } else if constexpr (Op == Conversion16::HalfToUnsigned) {
      value = _mm256_min_ps(_mm256_max_ps(value, zero), _mm256_set1_ps(65535));
      result = _mm256_cvttps_epi32(value);
    } else {
      auto nan = _mm256_cmp_ps(value, value, _CMP_UNORD_Q);
      value = _mm256_min_ps(_mm256_max_ps(value, _mm256_set1_ps(-32768)), _mm256_set1_ps(32767));
      result = _mm256_and_si256(_mm256_cvttps_epi32(value), _mm256_set1_epi32(65535));
      result = _mm256_andnot_si256(_mm256_castps_si256(nan), result);
    }
    if constexpr (Op != Conversion16::HalfToFloat) {
      auto original = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
      auto selected_mask = _mm256_sll_epi32(_mm256_set1_epi32(65535), sd);
      result = _mm256_or_si256(_mm256_andnot_si256(selected_mask, original),
                               _mm256_sll_epi32(result, sd));
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

} // namespace

// Convert FP16, FP32, or 16-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion16 Op>
void conversion16_x86_64_v3(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a) {
  convert<Op>(saturate, exec_mask, mode, d, a);
}

// Convert FP16, FP32, or 16-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void conversion16_widen_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                  const uint32_t *a) {
  convert<Conversion16::HalfToFloat, true>(false, exec_mask, mode, d, a, flags);
}

template void conversion16_x86_64_v3<Conversion16::SignedToHalf>(bool, uint32_t, uint32_t,
                                                                 uint32_t *, const uint32_t *);
template void conversion16_x86_64_v3<Conversion16::UnsignedToHalf>(bool, uint32_t, uint32_t,
                                                                   uint32_t *, const uint32_t *);
template void conversion16_x86_64_v3<Conversion16::HalfToSigned>(bool, uint32_t, uint32_t,
                                                                 uint32_t *, const uint32_t *);
template void conversion16_x86_64_v3<Conversion16::HalfToUnsigned>(bool, uint32_t, uint32_t,
                                                                   uint32_t *, const uint32_t *);
template void conversion16_x86_64_v3<Conversion16::FloatToHalf>(bool, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *);
template void conversion16_x86_64_v3<Conversion16::HalfToFloat>(bool, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
