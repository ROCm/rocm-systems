// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Sixteen-lane loose FP32 ADD/SUB/MUL, including DX9 zero products.
// Source modifiers, guest flushing, OMOD and masked writes stay vectorized.

#include "common/x86_64/alu_x86_64_v4.h"
#include "fp/binary.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Apply power-of-two output scaling with guest-directed overflow encodings.
// OMOD flushes tiny inputs independently of the guest denormal controls.
__m512 scale_omod(__m512 value, uint32_t instruction_flags, unsigned rounding) {
  const unsigned omod = (instruction_flags >> 6) & 3;
  value = prepare_omod_f32(value, instruction_flags);
  auto bits = _mm512_castps_si512(value);
  auto magnitude = _mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX));
  auto normal = _mm512_cmpgt_epi32_mask(magnitude, _mm512_setzero_si512()) &
                _mm512_cmplt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000));
  auto scaled = _mm512_add_epi32(bits, _mm512_set1_epi32(omod == 3 ? -0x800000 : int(omod << 23)));
  if (omod != 3) {
    auto overflow =
        _mm512_cmpgt_epi32_mask(magnitude, _mm512_set1_epi32(0x7f800000 - int(omod << 23) - 1));
    auto negative = _mm512_cmplt_epi32_mask(bits, _mm512_setzero_si512());
    __mmask16 infinity = rounding == 0   ? __mmask16(0xffff)
                         : rounding == 1 ? __mmask16(~negative)
                         : rounding == 2 ? negative
                                         : __mmask16(0);
    auto limit = _mm512_or_si512(_mm512_and_si512(bits, _mm512_set1_epi32(INT32_MIN)),
                                 _mm512_mask_blend_epi32(infinity, _mm512_set1_epi32(0x7f7fffff),
                                                         _mm512_set1_epi32(0x7f800000)));
    scaled = _mm512_mask_mov_epi32(scaled, overflow, limit);
  }
  return _mm512_castsi512_ps(_mm512_mask_mov_epi32(bits, normal, scaled));
}

// Evaluate one validated operation, allowing D to alias either complete input VGPR.
template <Binary Op>
void run(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags, uint32_t *d,
         const uint32_t *a, const uint32_t *b) {
  auto ka = _mm512_set1_epi32(instruction_flags & RJ_ARITH_ALU_ABS_A ? INT32_MAX : -1);
  auto kb = _mm512_set1_epi32(instruction_flags & RJ_ARITH_ALU_ABS_B ? INT32_MAX : -1);
  auto na = _mm512_set1_epi32(instruction_flags & RJ_ARITH_ALU_NEG_A ? INT32_MIN : 0);
  auto nb = _mm512_set1_epi32(instruction_flags & RJ_ARITH_ALU_NEG_B ? INT32_MIN : 0);
  for (int lane = 0; lane < 32; lane += 16) {
    auto x = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(a + lane), ka), na));
    auto y = _mm512_castsi512_ps(
        _mm512_xor_si512(_mm512_and_si512(_mm512_loadu_si512(b + lane), kb), nb));
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      x = flush_denorm_f32(x);
      y = flush_denorm_f32(y);
    }
    __m512 value;
    if constexpr (Op == Binary::Add) {
      value = _mm512_add_ps(x, y);
    } else if constexpr (Op == Binary::Sub) {
      value = _mm512_sub_ps(x, y);
    } else if constexpr (Op == Binary::Subrev) {
      value = _mm512_sub_ps(y, x);
    } else {
      value = _mm512_mul_ps(x, y);
      if constexpr (Op == Binary::MulDx9Zero) {
        auto has_zero = _mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_EQ_OQ) |
                        _mm512_cmp_ps_mask(y, _mm512_setzero_ps(), _CMP_EQ_OQ);
        value = _mm512_mask_mov_ps(value, has_zero, _mm512_setzero_ps());
      }
    }
    if (instruction_flags & RJ_ARITH_ALU_OMOD_MASK) {
      value =
          scale_omod(value, instruction_flags, unsigned((flags & RJ_ARITH_FP_ROUND_MASK) >> 24));
    }
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
    if (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS) {
      value = flush_denorm_f32(value);
    }
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), _mm512_castps_si512(value));
  }
}

} // namespace

// Execute only ADD/SUB/SUBREV/MUL/DX9-MUL with validated flags and AVX-512 available.
void binary_x86_64_v4(Binary op, uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                      uint32_t *d, const uint32_t *a, const uint32_t *b) {
  switch (op) {
  case Binary::Add:
    return run<Binary::Add>(flags, exec_mask, instruction_flags, d, a, b);
  case Binary::Sub:
    return run<Binary::Sub>(flags, exec_mask, instruction_flags, d, a, b);
  case Binary::Subrev:
    return run<Binary::Subrev>(flags, exec_mask, instruction_flags, d, a, b);
  case Binary::Mul:
    return run<Binary::Mul>(flags, exec_mask, instruction_flags, d, a, b);
  case Binary::MulDx9Zero:
    return run<Binary::MulDx9Zero>(flags, exec_mask, instruction_flags, d, a, b);
  default:
    __builtin_unreachable();
  }
}

} // namespace rocjitsu::arith
