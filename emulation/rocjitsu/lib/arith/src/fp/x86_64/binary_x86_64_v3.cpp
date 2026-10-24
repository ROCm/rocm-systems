// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 FP32 add/subtract/multiply and DX9-zero multiplication and min/max selection.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/x86_64/alu_x86_64_v3.h"
#include "fp/binary.h"
#include "fp/x86_64/minmax_x86_64_v3.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <initializer_list>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Apply OMOD using exponent arithmetic. Only finite normal lanes are scaled;
// prepared zeros and nonfinite inputs retain their bits. No host mode changes.
inline __m256 scale_omod(__m256 value, uint32_t mode, unsigned rounding) {
  const unsigned omod = (mode >> 6) & 3;
  value = prepare_omod_f32(value, mode);
  auto bits = _mm256_castps_si256(value);
  auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
  auto normal = _mm256_and_si256(_mm256_cmpgt_epi32(magnitude, _mm256_setzero_si256()),
                                 _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000), magnitude));
  auto scaled = _mm256_add_epi32(bits, _mm256_set1_epi32(omod == 3 ? -0x800000 : int(omod << 23)));
  if (omod != 3) {
    auto overflow =
        _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f800000 - int(omod << 23) - 1));
    auto negative = _mm256_srai_epi32(bits, 31);
    auto infinity = rounding == 0   ? _mm256_set1_epi32(-1)
                    : rounding == 1 ? _mm256_xor_si256(negative, _mm256_set1_epi32(-1))
                    : rounding == 2 ? negative
                                    : _mm256_setzero_si256();
    auto limit = _mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN)),
        _mm256_blendv_epi8(_mm256_set1_epi32(0x7f7fffff), _mm256_set1_epi32(0x7f800000), infinity));
    scaled = _mm256_blendv_epi8(scaled, limit, overflow);
  }
  return _mm256_castsi256_ps(_mm256_blendv_epi8(bits, scaled, normal));
}

// Evaluate FP32 binary arithmetic for the selected lanes. Requires validated controls;
// preserve inactive destination lanes.
template <Binary Op>
void run(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
         const uint32_t *b) {
  const auto ka = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  const auto kb = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_B ? INT32_MAX : -1);
  const auto na = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  const auto nb = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)), ka), na));
    auto y = _mm256_castsi256_ps(_mm256_xor_si256(
        _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)), kb), nb));
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      x = flush_denorm_f32(x);
      y = flush_denorm_f32(y);
    }
    __m256 value;
    if constexpr (Op == Binary::Add) {
      value = _mm256_add_ps(x, y);
    }
    if constexpr (Op == Binary::Sub) {
      value = _mm256_sub_ps(x, y);
    }
    if constexpr (Op == Binary::Subrev) {
      value = _mm256_sub_ps(y, x);
    }
    if constexpr (Op == Binary::MulDx9Zero) {
      auto zero = _mm256_setzero_ps();
      auto has_zero =
          _mm256_or_ps(_mm256_cmp_ps(x, zero, _CMP_EQ_OQ), _mm256_cmp_ps(y, zero, _CMP_EQ_OQ));
      value = _mm256_andnot_ps(has_zero, _mm256_mul_ps(x, y));
    }
    if constexpr (Op == Binary::Mul) {
      value = _mm256_mul_ps(x, y);
    }
    if constexpr (Op == Binary::MinNum) {
      value = minmax<false, false>(x, y);
    }
    if constexpr (Op == Binary::MaxNum) {
      value = minmax<true, false>(x, y);
    }
    if constexpr (Op == Binary::Minimum) {
      value = minmax<false, true>(x, y);
    }
    if constexpr (Op == Binary::Maximum) {
      value = minmax<true, true>(x, y);
    }
    if constexpr (Op == Binary::MinLegacy || Op == Binary::MaxLegacy) {
      value = minmax_rdna3<Op == Binary::MaxLegacy>(x, y, flags & RJ_ARITH_FP_IEEE_MODE);
    }
    if (mode & RJ_ARITH_ALU_OMOD_HALF) {
      value = scale_omod(value, mode, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto clamped = _mm256_min_ps(_mm256_max_ps(value, _mm256_setzero_ps()), _mm256_set1_ps(1));
      value = flags & RJ_ARITH_FP_DX10_CLAMP_OFF
                  ? _mm256_blendv_ps(clamped, value, _mm256_cmp_ps(value, value, _CMP_UNORD_Q))
                  : clamped;
    }
    constexpr bool arithmetic = Op == Binary::Add || Op == Binary::Sub || Op == Binary::Subrev ||
                                Op == Binary::Mul || Op == Binary::MulDx9Zero;
    if (arithmetic && (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS)) {
      value = flush_denorm_f32(value);
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask,
                           _mm256_castps_si256(value));
  }
}

} // namespace

// Evaluate FP32 binary arithmetic. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void binary_x86_64_v3(Binary op, uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                      const uint32_t *a, const uint32_t *b) {
  switch (op) {
  case Binary::Add:
    return run<Binary::Add>(flags, exec_mask, mode, d, a, b);
  case Binary::Sub:
    return run<Binary::Sub>(flags, exec_mask, mode, d, a, b);
  case Binary::Subrev:
    return run<Binary::Subrev>(flags, exec_mask, mode, d, a, b);
  case Binary::MinNum:
    return run<Binary::MinNum>(flags, exec_mask, mode, d, a, b);
  case Binary::MaxNum:
    return run<Binary::MaxNum>(flags, exec_mask, mode, d, a, b);
  case Binary::Minimum:
    return run<Binary::Minimum>(flags, exec_mask, mode, d, a, b);
  case Binary::Maximum:
    return run<Binary::Maximum>(flags, exec_mask, mode, d, a, b);
  case Binary::MulDx9Zero:
    return run<Binary::MulDx9Zero>(flags, exec_mask, mode, d, a, b);
  case Binary::MinLegacy:
    return run<Binary::MinLegacy>(flags, exec_mask, mode, d, a, b);
  case Binary::MaxLegacy:
    return run<Binary::MaxLegacy>(flags, exec_mask, mode, d, a, b);
  case Binary::Mul:
    return run<Binary::Mul>(flags, exec_mask, mode, d, a, b);
  }
}

// The inherited FMA(A, 1, B) model equals native addition for finite inputs
// with nearest-even rounding. Reject nonfinite operands before any stores so
// the scalar path retains empirical NaN rules. Default host FP state is required.
bool binary_exact_x86_64_v3(Binary op, uint64_t flags, uint32_t exec_mask, uint32_t mode,
                            uint32_t *d, const uint32_t *a, const uint32_t *b) {
  if (flags & RJ_ARITH_FP_ROUND_MASK) {
    return false;
  }
  for (const uint32_t *input : {a, b}) {
    for (int lane = 0; lane < 32; lane += 8) {
      auto bits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(input + lane));
      auto exponent = _mm256_and_si256(bits, _mm256_set1_epi32(0x7f800000));
      if (_mm256_movemask_epi8(_mm256_cmpeq_epi32(exponent, _mm256_set1_epi32(0x7f800000)))) {
        return false;
      }
    }
  }
  binary_x86_64_v3(op, flags, exec_mask, mode, d, a, b);
  return true;
}

} // namespace rocjitsu::arith
