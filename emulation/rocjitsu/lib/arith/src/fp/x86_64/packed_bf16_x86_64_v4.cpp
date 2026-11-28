// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native Loose packed BF16 ADD/MUL/FMA on 16 lanes, including selectors and saturation.
#include "common/packed_alu.h"
#include "fp/packed_bf16.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Widen the selected BF16 half exactly and apply its packed sign modifier.
__m512 input(__m512i bits, uint32_t mode) {
  bits = mode & RJ_ARITH_ALU_HIGH_A ? _mm512_and_si512(bits, _mm512_set1_epi32(-65536))
                                    : _mm512_slli_epi32(bits, 16);
  if (mode & RJ_ARITH_ALU_NEG_A) {
    bits = _mm512_xor_si512(bits, _mm512_set1_epi32(INT32_MIN));
  }
  return _mm512_castsi512_ps(bits);
}

// Evaluate one result half; FP16_OVFL saturates only finite-input overflows.
template <PackedBf16 Op>
__m512i compute(__m512i a, __m512i b, __m512i c, uint32_t mode, bool saturate) {
  auto x = input(a, mode), y = input(b, mode >> 1);
  auto finite = (_mm512_cmpgt_epi32_mask(
                     _mm512_set1_epi32(0x7f800000),
                     _mm512_and_si512(_mm512_castps_si512(x), _mm512_set1_epi32(INT32_MAX))) &
                 _mm512_cmpgt_epi32_mask(
                     _mm512_set1_epi32(0x7f800000),
                     _mm512_and_si512(_mm512_castps_si512(y), _mm512_set1_epi32(INT32_MAX))));
  __m512 value;
  if constexpr (Op == PackedBf16::Fma) {
    auto z = input(c, mode >> 2);
    finite = (finite & _mm512_cmpgt_epi32_mask(
                           _mm512_set1_epi32(0x7f800000),
                           _mm512_and_si512(_mm512_castps_si512(z), _mm512_set1_epi32(INT32_MAX))));
    value = _mm512_fmadd_ps(x, y, z);
  } else if constexpr (Op == PackedBf16::Add) {
    value = _mm512_add_ps(x, y);
  } else {
    value = _mm512_mul_ps(x, y);
  }
  // Round once to BF16; keep infinities and quiet NaNs without scalar lane repair.
  auto bits = _mm512_castps_si512(value);
  auto increment =
      _mm512_add_epi32(_mm512_set1_epi32(0x7fff),
                       _mm512_and_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(1)));
  auto result = _mm512_srli_epi32(_mm512_add_epi32(bits, increment), 16);
  auto nan = _mm512_cmpgt_epi32_mask(_mm512_and_si512(bits, _mm512_set1_epi32(INT32_MAX)),
                                     _mm512_set1_epi32(0x7f800000));
  result = _mm512_mask_mov_epi32(
      result, nan, _mm512_or_si512(_mm512_srli_epi32(bits, 16), _mm512_set1_epi32(0x40)));
  if (saturate) {
    auto overflow =
        (finite & _mm512_cmpeq_epi32_mask(_mm512_and_si512(result, _mm512_set1_epi32(0x7fff)),
                                          _mm512_set1_epi32(0x7f80)));
    result =
        _mm512_mask_mov_epi32(result, overflow, _mm512_sub_epi32(result, _mm512_set1_epi32(1)));
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    // All negative encodings and positive NaNs clamp to zero, positive infinity to one.
    auto zero = _mm512_setzero_si512();
    auto negative = _mm512_cmpgt_epi32_mask(result, _mm512_set1_epi32(0x7fff));
    auto clamped = _mm512_min_epi32(result, _mm512_set1_epi32(0x3f80));
    result = _mm512_mask_mov_epi32(clamped, negative, zero);
    result = _mm512_mask_mov_epi32(result, nan, zero);
  }
  return result;
}

// Consume both operand halves before publishing this chunk, allowing complete VGPR overlap.
template <PackedBf16 Op>
void run(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c) {
  const auto low_mode = packed_half_instruction_flags(mode, false);
  const auto high_mode = packed_half_instruction_flags(mode, true);
  for (int lane = 0; lane < 32; lane += 16) {
    auto va = _mm512_loadu_si512(a + lane), vb = _mm512_loadu_si512(b + lane);
    auto vc = _mm512_setzero_si512();
    if constexpr (Op == PackedBf16::Fma) {
      vc = _mm512_loadu_si512(c + lane);
    }
    auto low = compute<Op>(va, vb, vc, low_mode, saturate);
    auto high = compute<Op>(va, vb, vc, high_mode, saturate);
    auto result = _mm512_or_si512(low, _mm512_slli_epi32(high, 16));
    _mm512_mask_storeu_epi32(d + lane, __mmask16(exec_mask >> lane), result);
  }
}

} // namespace

// Execute one validated operation with the CPU features named by this backend.
void packed_bf16_x86_64_v4(PackedBf16 op, bool saturate, uint32_t exec_mask, uint32_t mode,
                           uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c) {
  switch (op) {
  case PackedBf16::Add:
    return run<PackedBf16::Add>(saturate, exec_mask, mode, d, a, b, c);
  case PackedBf16::Mul:
    return run<PackedBf16::Mul>(saturate, exec_mask, mode, d, a, b, c);
  case PackedBf16::Fma:
    return run<PackedBf16::Fma>(saturate, exec_mask, mode, d, a, b, c);
  }
  __builtin_unreachable();
}

} // namespace rocjitsu::arith
