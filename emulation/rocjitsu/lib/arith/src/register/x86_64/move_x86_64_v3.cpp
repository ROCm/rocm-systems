// SPDX-License-Identifier: MIT

// AVX2 sIMD halfword moves/swaps and Wave64 half-wave exchange.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate move half for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void move_half_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *d, const uint32_t *a) {
  auto sa = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
  auto sd = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  auto destination = _mm256_sll_epi32(_mm256_set1_epi32(65535), sd);
  auto keep = _mm256_set1_epi32(mode & RJ_ARITH_ALU_ABS_A ? 32767 : 65535);
  auto sign = _mm256_set1_epi32(mode & RJ_ARITH_ALU_NEG_A ? 32768 : 0);
  for (int lane = 0; lane < 32; lane += 8) {
    auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    value = _mm256_xor_si256(_mm256_and_si256(_mm256_srl_epi32(value, sa), keep), sign);
    auto original = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    value =
        _mm256_or_si256(_mm256_andnot_si256(destination, original), _mm256_sll_epi32(value, sd));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(exec_mask >> lane),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, value);
  }
}

// Evaluate swap half for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void swap_half_x86_64_v3(int lanes, uint64_t exec_mask, uint32_t mode, uint32_t *d, uint32_t *a) {
  auto sa = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_A ? 16 : 0);
  auto sd = _mm_cvtsi32_si128(mode & RJ_ARITH_ALU_HIGH_D ? 16 : 0);
  auto destination = _mm256_sll_epi32(_mm256_set1_epi32(65535), sd);
  auto source = _mm256_sll_epi32(_mm256_set1_epi32(65535), sa);
  auto same_register = _mm256_set1_epi32(d == a ? -1 : 0);
  for (int lane = 0; lane < lanes; lane += 8) {
    auto old_a = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto old_d = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(d + lane));
    auto new_d = _mm256_or_si256(
        _mm256_andnot_si256(destination, old_d),
        _mm256_and_si256(destination, _mm256_sll_epi32(_mm256_srl_epi32(old_a, sa), sd)));
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(uint32_t(exec_mask >> lane)),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, new_d);

    // Preserve the first write within one VGPR without reloading a masked store.
    // Inactive results are immaterial: both destination stores apply EXEC.
    auto preserved = _mm256_blendv_epi8(old_a, new_d, same_register);
    auto new_a = _mm256_or_si256(
        _mm256_andnot_si256(source, preserved),
        _mm256_and_si256(source, _mm256_sll_epi32(_mm256_srl_epi32(old_d, sd), sa)));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(a + lane), active, new_a);
  }
}

// Evaluate permlane64 for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void permlane64_x86_64_v3(uint64_t exec_mask, uint32_t *d, const uint32_t *a) {
  for (int lane = 0; lane < 32; lane += 8) {
    // Read both sides of the exchange before either store, including D == A.
    auto lower = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane));
    auto upper = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(a + lane + 32));
    auto low_active = _mm256_sllv_epi32(_mm256_set1_epi32(uint32_t(exec_mask >> lane)),
                                        _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    auto high_active = _mm256_sllv_epi32(_mm256_set1_epi32(uint32_t(exec_mask >> (lane + 32))),
                                         _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), low_active, upper);
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane + 32), high_active, lower);
  }
}

} // namespace rocjitsu::arith
