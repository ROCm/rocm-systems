// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX2 sIMD DS gather/swizzle routing with staged source reads.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Select source EXEC bits independently of destination participation.
__m256i gather(const uint32_t *a, __m256i source, uint64_t exec_mask) {
  auto value = _mm256_i32gather_epi32(reinterpret_cast<const int *>(a), source, 4);
  auto upper = _mm256_cmpgt_epi32(source, _mm256_set1_epi32(31));
  auto bits = _mm256_blendv_epi8(_mm256_set1_epi32(int(uint32_t(exec_mask))),
                                 _mm256_set1_epi32(int(uint32_t(exec_mask >> 32))), upper);
  auto active =
      _mm256_and_si256(_mm256_srlv_epi32(bits, _mm256_and_si256(source, _mm256_set1_epi32(31))),
                       _mm256_set1_epi32(1));
  value = _mm256_and_si256(value, _mm256_sub_epi32(_mm256_setzero_si256(), active));
  return value;
}

// Delay writes until every address and source has been read, including aliases.
void store(int lanes, uint64_t exec_mask, uint32_t *d, const uint32_t *result) {
  for (int lane = 0; lane < lanes; lane += 8) {
    auto active = _mm256_sllv_epi32(_mm256_set1_epi32(int(uint32_t(exec_mask >> lane))),
                                    _mm256_setr_epi32(31, 30, 29, 28, 27, 26, 25, 24));
    auto value = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(result + lane));
    _mm256_maskstore_epi32(reinterpret_cast<int *>(d + lane), active, value);
  }
}

} // namespace

namespace rocjitsu::arith {

// Evaluate ds bpermute for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void ds_bpermute_x86_64_v3(int lanes, int group, uint64_t exec_mask, uint32_t *d,
                           const uint32_t *addr, const uint32_t *data, uint16_t offset) {
  uint32_t result[64];
  for (int lane = 0; lane < lanes; lane += 8) {
    auto address =
        _mm256_add_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(addr + lane)),
                         _mm256_set1_epi32(offset));
    auto source = _mm256_or_si256(
        _mm256_and_si256(_mm256_srli_epi32(address, 2), _mm256_set1_epi32(group - 1)),
        _mm256_set1_epi32(lane & ~(group - 1)));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result + lane),
                        gather(data, source, exec_mask));
  }
  store(lanes, exec_mask, d, result);
}

// Evaluate ds swizzle for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void ds_swizzle_x86_64_v3(int lanes, uint64_t exec_mask, uint32_t *d, const uint32_t *a,
                          uint16_t offset) {
  unsigned mask = offset & 31;
  unsigned count = unsigned(__builtin_popcount(mask));
  unsigned rotate = (offset >> 5) & 31;
  if (offset & 1024) {
    rotate = 0U - rotate;
  }
  uint32_t result[64];
  for (int lane = 0; lane < lanes; lane += 8) {
    auto indices =
        _mm256_add_epi32(_mm256_set1_epi32(lane), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    __m256i source;
    if (offset >= 0xe000) {
      auto reversed = _mm256_or_si256(_mm256_setr_epi32(0, 16, 8, 24, 4, 20, 12, 28),
                                      _mm256_set1_epi32(((lane & 8) >> 2) | ((lane & 16) >> 4)));
      source = _mm256_or_si256(_mm256_srl_epi32(reversed, _mm_cvtsi32_si128(count)),
                               _mm256_and_si256(indices, _mm256_set1_epi32(mask)));
    } else if (offset >= 0xc000) {
      source = _mm256_or_si256(
          _mm256_and_si256(indices, _mm256_set1_epi32(mask)),
          _mm256_andnot_si256(_mm256_set1_epi32(mask),
                              _mm256_add_epi32(indices, _mm256_set1_epi32(int(rotate)))));
    } else if (offset & 0x8000) {
      auto selected = _mm256_and_si256(
          _mm256_srlv_epi32(_mm256_set1_epi32(offset), _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6)),
          _mm256_set1_epi32(3));
      source = _mm256_or_si256(_mm256_and_si256(indices, _mm256_set1_epi32(-4)), selected);
    } else {
      source = _mm256_xor_si256(_mm256_or_si256(_mm256_and_si256(indices, _mm256_set1_epi32(mask)),
                                                _mm256_set1_epi32((offset >> 5) & 31)),
                                _mm256_set1_epi32((offset >> 10) & 31));
    }
    source = _mm256_or_si256(_mm256_and_si256(source, _mm256_set1_epi32(31)),
                             _mm256_set1_epi32(lane & 32));
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(result + lane), gather(a, source, exec_mask));
  }
  store(lanes, exec_mask, d, result);
}

} // namespace rocjitsu::arith
