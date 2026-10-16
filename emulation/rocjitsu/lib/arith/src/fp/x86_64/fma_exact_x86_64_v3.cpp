// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 guarded exact FP32 FMA using native fused arithmetic and scalar exceptional-lane
// repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "common/x86_64/alu_x86_64_v3.h"
#include "fp/scalar_repair.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Native nearest FMA implements the inherited normal-result model. Repair
// special/tiny lanes with that same scalar model; stage every result before
// masked stores so repair remains safe when operands alias D. Directed guest
// rounding requires AVX512 instruction-local rounding or the scalar path.
bool fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                         const uint32_t *a, const uint32_t *b, const uint32_t *c, bool dx9_zero,
                         bool rdna3) {
  if (flags & RJ_ARITH_FP_ROUND_MASK) {
    return false;
  }
  uint32_t result[32];
  for (int lane = 0; lane < 32; lane += 8) {
    const uint32_t *inputs[] = {a, b, c};
    __m256 values[3];
    auto bad = _mm256_setzero_si256();
    for (int operand = 0; operand < 3; ++operand) {
      auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(inputs[operand] + lane));
      if (mode & (RJ_ARITH_ALU_ABS_A << operand)) {
        x = _mm256_and_si256(x, _mm256_set1_epi32(INT32_MAX));
      }
      if (mode & (RJ_ARITH_ALU_NEG_A << operand)) {
        x = _mm256_xor_si256(x, _mm256_set1_epi32(INT32_MIN));
      }
      auto mag = _mm256_and_si256(x, _mm256_set1_epi32(INT32_MAX));
      bad = _mm256_or_si256(bad, _mm256_cmpgt_epi32(mag, _mm256_set1_epi32(0x7f7fffff)));
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        x = _mm256_blendv_epi8(x, _mm256_and_si256(x, _mm256_set1_epi32(INT32_MIN)),
                               _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800000), mag));
      }
      values[operand] = _mm256_castsi256_ps(x);
    }
    if (dx9_zero) {
      auto zeros = _mm256_or_ps(_mm256_cmp_ps(values[0], _mm256_setzero_ps(), _CMP_EQ_OQ),
                                _mm256_cmp_ps(values[1], _mm256_setzero_ps(), _CMP_EQ_OQ));
      values[0] = _mm256_blendv_ps(values[0], _mm256_setzero_ps(), zeros);
      values[1] = _mm256_blendv_ps(values[1], _mm256_set1_ps(1), zeros);
    }
    auto value = _mm256_fmadd_ps(values[0], values[1], values[2]);
    auto mag = _mm256_and_si256(_mm256_castps_si256(value), _mm256_set1_epi32(INT32_MAX));
    bad =
        _mm256_or_si256(bad, _mm256_or_si256(_mm256_cmpgt_epi32(mag, _mm256_set1_epi32(0x7f7fffff)),
                                             _mm256_cmpgt_epi32(_mm256_set1_epi32(0x800001), mag)));
    if (unsigned omod = (mode >> 6) & 3) {
      value = prepare_omod_f32(value, mode);
      value = _mm256_mul_ps(value, _mm256_set1_ps(omod == 3 ? 0.5f : omod == 1 ? 2.0f : 4.0f));
    }
    if (mode & RJ_ARITH_ALU_CLAMP) {
      value = clamp_f32(value, flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result + lane), _mm256_castps_si256(value));
    // Repair exceptional or tiny lanes with the scalar model; ordinary lanes retain the native
    // fused result.
    unsigned repair = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(bad)));
    if (repair) {
      for (int i = 0; i < 8; ++i) {
        if ((repair >> i) & 1) {
          result[lane + i] =
              repair_fma_f32(a[lane + i], b[lane + i], c[lane + i], flags, mode, rdna3, dx9_zero);
        }
      }
    }
  }
  for (int lane = 0; lane < 32; lane += 8) {
    auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                  _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), mask,
                           _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result + lane)));
  }
  return true;
}
} // namespace rocjitsu::arith
