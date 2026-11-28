// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Native Loose packed BF16 ADD/MUL/FMA on 8 lanes, including selectors and saturation.
#include "common/packed_alu.h"
#include "fp/packed_bf16.h"
#include "rj_arith/detail/rj_arith_common.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {
namespace {

// Widen the selected BF16 half exactly and apply its packed sign modifier.
__m256 input(__m256i bits, uint32_t mode) {
  bits = mode & RJ_ARITH_ALU_HIGH_A ? _mm256_and_si256(bits, _mm256_set1_epi32(-65536))
                                    : _mm256_slli_epi32(bits, 16);
  if (mode & RJ_ARITH_ALU_NEG_A) {
    bits = _mm256_xor_si256(bits, _mm256_set1_epi32(INT32_MIN));
  }
  return _mm256_castsi256_ps(bits);
}

// Evaluate one result half; FP16_OVFL saturates only finite-input overflows.
template <PackedBf16 Op>
__m256i compute(__m256i a, __m256i b, __m256i c, uint32_t mode, bool saturate) {
  auto x = input(a, mode), y = input(b, mode >> 1);
  auto finite = _mm256_and_si256(
      _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000),
                         _mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(INT32_MAX))),
      _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000),
                         _mm256_and_si256(_mm256_castps_si256(y), _mm256_set1_epi32(INT32_MAX))));
  __m256 value;
  if constexpr (Op == PackedBf16::Fma) {
    auto z = input(c, mode >> 2);
    finite = _mm256_and_si256(
        finite,
        _mm256_cmpgt_epi32(_mm256_set1_epi32(0x7f800000),
                           _mm256_and_si256(_mm256_castps_si256(z), _mm256_set1_epi32(INT32_MAX))));
    value = _mm256_fmadd_ps(x, y, z);
  } else if constexpr (Op == PackedBf16::Add) {
    value = _mm256_add_ps(x, y);
  } else {
    value = _mm256_mul_ps(x, y);
  }
  // Round once to BF16; keep infinities and quiet NaNs without scalar lane repair.
  auto bits = _mm256_castps_si256(value);
  auto increment =
      _mm256_add_epi32(_mm256_set1_epi32(0x7fff),
                       _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1)));
  auto result = _mm256_srli_epi32(_mm256_add_epi32(bits, increment), 16);
  auto nan = _mm256_cmpgt_epi32(_mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX)),
                                _mm256_set1_epi32(0x7f800000));
  result = _mm256_blendv_epi8(
      result, _mm256_or_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(0x40)), nan);
  if (saturate) {
    auto overflow = _mm256_and_si256(
        finite, _mm256_cmpeq_epi32(_mm256_and_si256(result, _mm256_set1_epi32(0x7fff)),
                                   _mm256_set1_epi32(0x7f80)));
    result = _mm256_blendv_epi8(result, _mm256_sub_epi32(result, _mm256_set1_epi32(1)), overflow);
  }
  if (mode & RJ_ARITH_ALU_CLAMP) {
    // All negative encodings and positive NaNs clamp to zero, positive infinity to one.
    auto zero = _mm256_setzero_si256();
    auto negative = _mm256_cmpgt_epi32(result, _mm256_set1_epi32(0x7fff));
    auto clamped = _mm256_min_epi32(result, _mm256_set1_epi32(0x3f80));
    result = _mm256_blendv_epi8(clamped, zero, negative);
    result = _mm256_blendv_epi8(result, zero, nan);
  }
  return result;
}

// Consume both operand halves before publishing this chunk, allowing complete VGPR overlap.
template <PackedBf16 Op>
void run(bool saturate, uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
         const uint32_t *b, const uint32_t *c) {
  const auto low_mode = packed_half_instruction_flags(mode, false);
  const auto high_mode = packed_half_instruction_flags(mode, true);
  for (int lane = 0; lane < 32; lane += 8) {
    auto va = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
         vb = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    auto vc = _mm256_setzero_si256();
    if constexpr (Op == PackedBf16::Fma) {
      vc = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane));
    }
    auto low = compute<Op>(va, vb, vc, low_mode, saturate);
    auto high = compute<Op>(va, vb, vc, high_mode, saturate);
    auto result = _mm256_or_si256(low, _mm256_slli_epi32(high, 16));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, result);
  }
}

} // namespace

// Execute one validated operation with the CPU features named by this backend.
void packed_bf16_x86_64_v3(PackedBf16 op, bool saturate, uint32_t exec_mask, uint32_t mode,
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
