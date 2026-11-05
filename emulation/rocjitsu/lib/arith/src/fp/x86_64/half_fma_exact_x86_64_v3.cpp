// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 exact FP16 FMA and packed FMA using wider arithmetic with scalar repair.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/internal.h"
#include "common/simd.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "fp/packed_fma.h"
#include "fp/scalar_repair.h"
#include "fp/x86_64/mixed_half_x86_64_v3.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Reuse the established compensated FP64/round-to-odd half-output arithmetic.
// Directed rounding and unsuppressed reporting retain the scalar implementation.
// Repair special lanes with the ordinary half-FMA model, not MIX's NaN policy.
void half_fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t instruction_flags,
                              uint32_t *d, const uint32_t *a, const uint32_t *b, const uint32_t *c,
                              bool rdna3) {
  uint32_t output[32];
  for (int lane = 0; lane < 32; lane += 8) {
    const uint32_t *sources[] = {a, b, c};
    __m256 values[3];
    auto bad = _mm256_setzero_si256();
    for (int src = 0; src < 3; ++src) {
      auto x = half_input(
          _mm256_loadu_si256(reinterpret_cast<const __m256i *>(sources[src] + lane)),
          instruction_flags & (RJ_ARITH_ALU_HIGH_A << src) ? 16 : 0, instruction_flags >> src);
      auto bits = _mm256_castps_si256(x);
      auto magnitude = _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX));
      bad = _mm256_or_si256(bad, _mm256_cmpgt_epi32(magnitude, _mm256_set1_epi32(0x7f7fffff)));
      if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
        x = flush_denorm_f16(x);
      }

      // The mixed helper's special repair implements a different model. Feed
      // it harmless finite values; ordinary-FMA repair below owns these lanes.
      values[src] = x;
    }
    for (auto &value : values) {
      value = _mm256_andnot_ps(_mm256_castsi256_ps(bad), value);
    }
    unsigned omod = (instruction_flags >> 6) & 3;
    __m128i halves[2];
    for (int part = 0; part < 2; ++part) {
      auto x = part ? _mm256_extractf128_ps(values[0], 1) : _mm256_castps256_ps128(values[0]);
      auto y = part ? _mm256_extractf128_ps(values[1], 1) : _mm256_castps256_ps128(values[1]);
      auto z = part ? _mm256_extractf128_ps(values[2], 1) : _mm256_castps256_ps128(values[2]);
      halves[part] = mixed_half_value4<false>(x, y, z, flags & RJ_ARITH_FP16_OVFL, false,
                                              omod || (flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS));
    }
    auto result = _mm256_inserti128_si256(_mm256_castsi128_si256(halves[0]), halves[1], 1);
    if (omod) {
      const float scales[] = {1, 2, 4, 0.5f};
      auto wide = prepare_omod_f16(half_input(result, 0, 0), instruction_flags);
      wide = _mm256_mul_ps(wide, _mm256_set1_ps(scales[omod]));
      result = half_narrow(wide, flags & RJ_ARITH_FP16_OVFL);
    }
    if (instruction_flags & RJ_ARITH_ALU_CLAMP) {
      auto negative = _mm256_cmpgt_epi32(result, _mm256_set1_epi32(0x7fff));
      result = _mm256_andnot_si256(negative, _mm256_min_epi32(result, _mm256_set1_epi32(0x3c00)));
    }
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(output + lane), result);
    unsigned repair = unsigned(_mm256_movemask_ps(_mm256_castsi256_ps(bad)));
    if (repair) {
      for (int i = 0; i < 8; ++i) {
        if (!((repair >> i) & 1)) {
          continue;
        }
        uint16_t x = uint16_t(a[lane + i] >> (instruction_flags & RJ_ARITH_ALU_HIGH_A ? 16 : 0));
        uint16_t y = uint16_t(b[lane + i] >> (instruction_flags & RJ_ARITH_ALU_HIGH_B ? 16 : 0));
        uint16_t z = uint16_t(c[lane + i] >> (instruction_flags & RJ_ARITH_ALU_HIGH_C ? 16 : 0));
        output[lane + i] = repair_fma_f16(x, y, z, flags, instruction_flags, rdna3);
      }
    }
  }
  int shift = instruction_flags & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto result =
        _mm256_sll_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(output + lane)),
                         _mm_cvtsi32_si128(shift));
    result = _mm256_or_si256(
        result, _mm256_and_si256(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane)),
                                 _mm256_set1_epi32(shift ? 0xffff : 0xffff0000U)));
    auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                  _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), mask, result);
  }
}

// Evaluate packed FP16 FMA, preserving inactive lanes and allowing destination overlap.
// Requires AVX2 and the rounding/reporting domain documented in fp/packed_fma.h.
// Return false without changing D when scalar evaluation is required.
bool packed_fma_exact_x86_64_v3(uint64_t flags, uint32_t exec_mask, const uint32_t modes[2],
                                uint32_t *d, const uint32_t *a, const uint32_t *b,
                                const uint32_t *c, uint32_t *, bool rdna3) {
  uint32_t staged[32];
  for (int lane = 0; lane < 32; ++lane) {
    staged[lane] = d[lane];
  }
  if ((flags & RJ_ARITH_FP_ROUND_MASK) ||
      (reports_fp_status(flags) &&
       (!(modes[0] & RJ_ARITH_ALU_CLAMP) || !(modes[1] & RJ_ARITH_ALU_CLAMP)))) {
    return false;
  }
  half_fma_exact_x86_64_v3(flags, exec_mask, modes[0], staged, a, b, c, rdna3);
  half_fma_exact_x86_64_v3(flags, exec_mask, modes[1] | RJ_ARITH_ALU_HIGH_D, staged, a, b, c,
                           rdna3);

  for (int lane = 0; lane < 32; lane += 8) {
    auto mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                  _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), mask,
                           _mm256_loadu_si256(reinterpret_cast<const __m256i *>(staged + lane)));
  }
  return true;
}

} // namespace rocjitsu::arith
