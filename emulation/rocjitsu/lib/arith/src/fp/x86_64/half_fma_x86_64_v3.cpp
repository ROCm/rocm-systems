// SPDX-License-Identifier: MIT

// AVX2 FP16 FMA/FMAC with guest rounding and optional FP status.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "fp/x86_64/half_fma_x86_64_v3.h"
#include "common/x86_64/half_x86_64_v3.h"
#include "fp/fma.h"
#include "fp/half_fma.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate FP16 fused multiply-add. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <FmaOperands Operands>
void half_fma_x86_64_v3(uint64_t flags, uint32_t exec_mask, uint32_t mode, uint32_t *d,
                        const uint32_t *a, const uint32_t *b, const uint32_t *c, uint16_t literal) {
  if constexpr (Operands != FmaOperands::Registers) {
    mode &= RJ_ARITH_ALU_HIGH_A | RJ_ARITH_ALU_HIGH_B | RJ_ARITH_ALU_HIGH_D;
  }
  int a_shift = mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0;
  int b_shift = mode & RJ_ARITH_ALU_HIGH_B ? 16 : 0;
  int c_shift = mode & RJ_ARITH_ALU_HIGH_C ? 16 : 0;
  int d_shift = mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0;
  auto keep = _mm256_set1_epi32(d_shift ? 65535 : -65536);
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane)),
                               a_shift, mode);
    auto y = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane)),
                               b_shift, mode >> 1);
    __m256 z;
    if constexpr (Operands == FmaOperands::MultiplyLiteral) {
      z = y;
      y = half_input<false>(_mm256_set1_epi32(literal), 0, 0);
    } else if constexpr (Operands == FmaOperands::AddLiteral) {
      z = half_input<false>(_mm256_set1_epi32(literal), 0, 0);
    } else {
      z = half_input<false>(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(c + lane)),
                            c_shift, mode >> 2);
    }
    if (flags & RJ_ARITH_FP_FLUSH_INPUT_DENORMALS) {
      // Replace subnormal FP32 lanes with signed zero.
      auto flush = [](__m256 input) {
        auto bits = _mm256_castps_si256(input);
        auto tiny = _mm256_cmpgt_epi32(_mm256_set1_epi32(0x38800000),
                                       _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MAX)));
        return _mm256_castsi256_ps(
            _mm256_blendv_epi8(bits, _mm256_and_si256(bits, _mm256_set1_epi32(INT32_MIN)), tiny));
      };
      x = flush(x);
      y = flush(y);
      z = flush(z);
    }
    auto result = half_fma_value(x, y, z, flags & RJ_ARITH_FP16_OVFL, mode,
                                 flags & RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS,
                                 flags & RJ_ARITH_FP_DX10_CLAMP_OFF);
    result = _mm256_sll_epi32(result, _mm_cvtsi32_si128(d_shift));
    auto old = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    result = _mm256_or_si256(result, _mm256_and_si256(old, keep));
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), lane_exec_mask, result);
  }
}

template void half_fma_x86_64_v3<FmaOperands::Registers>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                         const uint32_t *, const uint32_t *,
                                                         const uint32_t *, uint16_t);

template void half_fma_x86_64_v3<FmaOperands::MultiplyLiteral>(uint64_t, uint32_t, uint32_t,
                                                               uint32_t *, const uint32_t *,
                                                               const uint32_t *, const uint32_t *,
                                                               uint16_t);

template void half_fma_x86_64_v3<FmaOperands::AddLiteral>(uint64_t, uint32_t, uint32_t, uint32_t *,
                                                          const uint32_t *, const uint32_t *,
                                                          const uint32_t *, uint16_t);

} // namespace rocjitsu::arith
