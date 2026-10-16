// SPDX-License-Identifier: MIT

// AVX2 per-lane conditional selection between two source registers.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "register/cndmask.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Select between two sources by lane condition. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
void cndmask_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a,
                       const uint32_t *b, uint32_t condition) {
  const uint32_t sign = 0x80000000;
  auto clear_a = _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_A ? ~sign : UINT32_MAX));
  auto clear_b = _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_ABS_B ? ~sign : UINT32_MAX));
  auto flip_a = _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_A ? sign : 0));
  auto flip_b = _mm256_set1_epi32(int(mode & RJ_ARITH_ALU_NEG_B ? sign : 0));
  for (int lane = 0; lane < 32; lane += 8) {
    auto x = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto y = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(b + lane));
    x = _mm256_xor_si256(_mm256_and_si256(x, clear_a), flip_a);
    y = _mm256_xor_si256(_mm256_and_si256(y, clear_b), flip_b);
    auto select = _mm256_sllv_epi32(_mm256_set1_epi32(int(condition >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    auto value = _mm256_blendv_epi8(x, y, _mm256_srai_epi32(select, 31));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, value);
  }
}

} // namespace rocjitsu::arith
