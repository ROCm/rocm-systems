// SPDX-License-Identifier: MIT

// AVX2 integer add/subtract with per-lane carry/borrow input and output masks.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "integer/carry.h"
#include "rj_arith/rj_arith.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Return the lane mask for unsigned a < b.
inline __m256i less_unsigned(__m256i a, __m256i b) {
  auto sign = _mm256_set1_epi32(INT32_MIN);
  return _mm256_cmpgt_epi32(_mm256_xor_si256(b, sign), _mm256_xor_si256(a, sign));
}

} // namespace

namespace rocjitsu::arith {

// Add/subtract with carry or borrow masks. Preserve destination lanes outside exec_mask.
// Requires the named CPU features and validated instruction controls.
template <CarryOp Op, bool WithCarry>
uint32_t carry_x86_64_v3(uint32_t exec_mask, uint32_t mode, uint32_t *const *d,
                         const uint32_t *const *a, const uint32_t *const *b, uint32_t input_carry) {
  uint32_t output_carry = 0;
  for (int lane = 0; lane < 32; lane += 8) {
    auto av = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>((Op == CarryOp::Subrev ? b : a)[0] + lane));
    auto bv = _mm256_loadu_si256(
        reinterpret_cast<const __m256i *>((Op == CarryOp::Subrev ? a : b)[0] + lane));
    __m256i result;
    __m256i co;
    if constexpr (Op == CarryOp::Add) {
      result = _mm256_add_epi32(av, bv);
      co = less_unsigned(result, av);
    } else {
      result = _mm256_sub_epi32(av, bv);
      co = less_unsigned(av, bv);
    }
    if constexpr (WithCarry) {
      auto ci = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(input_carry >> lane)),
                                                   _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)),
                                 _mm256_set1_epi32(1));
      if constexpr (Op == CarryOp::Add) {
        auto sum = _mm256_add_epi32(result, ci);
        co = _mm256_or_si256(co, less_unsigned(sum, result));
        result = sum;
      } else {
        co = _mm256_or_si256(co, less_unsigned(result, ci));
        result = _mm256_sub_epi32(result, ci);
      }
    }
    output_carry |= uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(co))) << lane;
    if (mode & RJ_ARITH_ALU_CLAMP) {
      auto limit = _mm256_set1_epi32(Op == CarryOp::Add ? -1 : 0);
      result = _mm256_blendv_epi8(result, limit, co);
    }
    auto lane_exec_mask = _mm256_sllv_epi32(_mm256_set1_epi32(int(exec_mask >> lane)),
                                            _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d[0] + lane), lane_exec_mask, result);
  }
  return output_carry & exec_mask;
}

template uint32_t carry_x86_64_v3<CarryOp::Add, false>(uint32_t, uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v3<CarryOp::Add, true>(uint32_t, uint32_t, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v3<CarryOp::Sub, false>(uint32_t, uint32_t, uint32_t *const *,
                                                       const uint32_t *const *,
                                                       const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v3<CarryOp::Sub, true>(uint32_t, uint32_t, uint32_t *const *,
                                                      const uint32_t *const *,
                                                      const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v3<CarryOp::Subrev, false>(uint32_t, uint32_t, uint32_t *const *,
                                                          const uint32_t *const *,
                                                          const uint32_t *const *, uint32_t);
template uint32_t carry_x86_64_v3<CarryOp::Subrev, true>(uint32_t, uint32_t, uint32_t *const *,
                                                         const uint32_t *const *,
                                                         const uint32_t *const *, uint32_t);

} // namespace rocjitsu::arith
