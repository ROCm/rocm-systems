// SPDX-License-Identifier: MIT

// AVX-512 conversions between FP16 and FP32 or 16-bit integers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "conversion/conversion16.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Widen low-halfword FP16 encodings to FP32 vector lanes.
__m512 widen(__m512i halves) { return _mm512_cvtph_ps(_mm512_cvtepi32_epi16(halves)); }

// Round FP32 vector lanes to FP16, optionally saturating finite overflow.
__m512i narrow(__m512 x, bool saturate) {
  auto result =
      _mm512_cvtepu16_epi32(_mm512_cvtps_ph(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
  if (saturate) {
    auto magnitude = _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(INT32_MAX));
    auto finite = _mm512_cmp_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000), _MM_CMPINT_LT);
    auto infinity = _mm512_cmpeq_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7fff)),
                                            _mm512_set1_epi32(0x7c00));
    result = _mm512_mask_sub_epi32(result, finite & infinity, result, _mm512_set1_epi32(1));
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
  auto scale = _mm512_set1_ps(scales[omod]);
  const auto zero = _mm512_setzero_ps();
  for (int lane = 0; lane < 32; lane += 16) {
    auto raw = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(a + lane));
    if constexpr (Op != Conversion16::FloatToHalf) {
      raw = _mm512_and_si512(_mm512_srl_epi32(raw, sa), _mm512_set1_epi32(65535));
    }
    __m512 value;
    if constexpr (from_integer) {
      if constexpr (Op == Conversion16::SignedToHalf) {
        raw = _mm512_srai_epi32(_mm512_slli_epi32(raw, 16), 16);
      }
      value = _mm512_cvtepi32_ps(raw);
    } else {
      constexpr uint32_t sign = Op == Conversion16::FloatToHalf ? 0x80000000 : 0x8000;
      raw = _mm512_and_si512(
          raw, _mm512_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_A ? ~sign : UINT32_MAX)));
      raw = _mm512_xor_si512(raw, _mm512_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_A ? sign : 0)));
      if constexpr (Op == Conversion16::FloatToHalf) {
        value = _mm512_castsi512_ps(raw);
      } else {
        if constexpr (Op == Conversion16::HalfToFloat && GuestWidening) {
          if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
            auto tiny = _mm512_cmp_epi32_mask(_mm512_and_si512(raw, _mm512_set1_epi32(0x7fff)),
                                              _mm512_set1_epi32(0x400), _MM_CMPINT_LT);
            raw =
                _mm512_mask_mov_epi32(raw, tiny, _mm512_and_si512(raw, _mm512_set1_epi32(0x8000)));
          }
        }
        value = widen(raw);
        if constexpr (Op == Conversion16::HalfToFloat && GuestWidening) {
          // F16C quiets NaNs; restore the raw encoding for RDNA3 IEEE=0.
          if (!(flags & RJ_ARITH_FP_IEEE_MODE)) {
            auto nan = _mm512_cmp_epi32_mask(_mm512_and_si512(raw, _mm512_set1_epi32(0x7fff)),
                                             _mm512_set1_epi32(0x7c00), _MM_CMPINT_GT);
            auto bits = _mm512_or_si512(
                _mm512_set1_epi32(0x7f800000),
                _mm512_or_si512(
                    _mm512_slli_epi32(_mm512_and_si512(raw, _mm512_set1_epi32(0x8000)), 16),
                    _mm512_slli_epi32(_mm512_and_si512(raw, _mm512_set1_epi32(0x3ff)), 13)));
            value =
                _mm512_castsi512_ps(_mm512_mask_mov_epi32(_mm512_castps_si512(value), nan, bits));
          }
        }
      }
    }
    __m512i result;
    if constexpr (to_half) {
      result = narrow(value, saturate);
      if (omod) {
        auto magnitude = _mm512_and_si512(result, _mm512_set1_epi32(0x7fff));
        auto boundary = _mm512_cmpeq_epi32_mask(magnitude, _mm512_set1_epi32(0x400));
        auto below = _mm512_cmp_epi32_mask(
            _mm512_and_si512(_mm512_castps_si512(value), _mm512_set1_epi32(INT32_MAX)),
            _mm512_set1_epi32(0x387ff000), _MM_CMPINT_LT);
        auto tiny = _mm512_cmp_epi32_mask(magnitude, _mm512_set1_epi32(0x400), _MM_CMPINT_LT) |
                    (boundary & below);
        auto initial = _mm512_mask_mov_epi32(result, tiny, _mm512_setzero_si512());
        result = narrow(_mm512_mul_ps(widen(initial), scale), saturate);
        if (omod == 3) {
          auto underflow =
              _mm512_cmp_epi32_mask(magnitude, _mm512_set1_epi32(0x800), _MM_CMPINT_LT);
          result = _mm512_mask_mov_epi32(result, underflow,
                                         _mm512_and_si512(initial, _mm512_set1_epi32(0x8000)));
        }
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        auto discard = _mm512_cmp_epi32_mask(result, _mm512_set1_epi32(0x7c00), _MM_CMPINT_GT);
        result = _mm512_min_epi32(_mm512_mask_mov_epi32(result, discard, _mm512_setzero_si512()),
                                  _mm512_set1_epi32(0x3c00));
      }
    } else if constexpr (Op == Conversion16::HalfToFloat) {
      const auto input_bits = _mm512_castps_si512(value);
      if (!GuestWidening || omod || (flags & RJ_ARITH_FP_IEEE_MODE)) {
        value = _mm512_mul_ps(value, scale);
      }
      if (mode & RJ_ARITH_ALU_CLAMP) {
        value = _mm512_min_ps(_mm512_max_ps(value, zero), _mm512_set1_ps(1));
      }
      if (omod) {
        value = _mm512_mask_mov_ps(value, _mm512_cmp_ps_mask(value, zero, _CMP_EQ_OQ), zero);
      }
      result = _mm512_castps_si512(value);
      if constexpr (GuestWidening) {
        // Preserve the widening NaN policy through OMOD and DX10_CLAMP-off.
        if (!(mode & RJ_ARITH_ALU_CLAMP) || (flags & RJ_ARITH_FP_DX10_CLAMP_OFF)) {
          auto nan =
              _mm512_cmpgt_epi32_mask(_mm512_and_si512(input_bits, _mm512_set1_epi32(0x7fffffff)),
                                      _mm512_set1_epi32(0x7f800000));
          result = _mm512_mask_mov_epi32(result, nan, input_bits);
        }
      }
    } else if constexpr (Op == Conversion16::HalfToUnsigned) {
      value = _mm512_min_ps(_mm512_max_ps(value, zero), _mm512_set1_ps(65535));
      result = _mm512_cvttps_epi32(value);
    } else {
      auto nan = _mm512_cmp_ps_mask(value, value, _CMP_UNORD_Q);
      value = _mm512_min_ps(_mm512_max_ps(value, _mm512_set1_ps(-32768)), _mm512_set1_ps(32767));
      result = _mm512_and_si512(_mm512_cvttps_epi32(value), _mm512_set1_epi32(65535));
      result = _mm512_mask_mov_epi32(result, nan, _mm512_setzero_si512());
    }
    if constexpr (Op != Conversion16::HalfToFloat) {
      auto original = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(d + lane));
      auto selected_mask = _mm512_sll_epi32(_mm512_set1_epi32(65535), sd);
      result = _mm512_or_si512(_mm512_andnot_si512(selected_mask, original),
                               _mm512_sll_epi32(result, sd));
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

} // namespace

// Convert FP16, FP32, or 16-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <Conversion16 Op>
void conversion16_x86_64_v4(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                            const uint32_t *a) {
  convert<Op>(saturate, exec_mask, mode, d, a);
}

// Convert FP16, FP32, or 16-bit integers. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void conversion16_widen_x86_64_v4(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                                  const uint32_t *a) {
  convert<Conversion16::HalfToFloat, true>(false, exec_mask, mode, d, a, flags);
}

template void conversion16_x86_64_v4<Conversion16::SignedToHalf>(bool, uint32_t, uint32_t,
                                                                 uint32_t *, const uint32_t *);
template void conversion16_x86_64_v4<Conversion16::UnsignedToHalf>(bool, uint32_t, uint32_t,
                                                                   uint32_t *, const uint32_t *);
template void conversion16_x86_64_v4<Conversion16::HalfToSigned>(bool, uint32_t, uint32_t,
                                                                 uint32_t *, const uint32_t *);
template void conversion16_x86_64_v4<Conversion16::HalfToUnsigned>(bool, uint32_t, uint32_t,
                                                                   uint32_t *, const uint32_t *);
template void conversion16_x86_64_v4<Conversion16::FloatToHalf>(bool, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *);
template void conversion16_x86_64_v4<Conversion16::HalfToFloat>(bool, uint32_t, uint32_t,
                                                                uint32_t *, const uint32_t *);

} // namespace rocjitsu::arith
