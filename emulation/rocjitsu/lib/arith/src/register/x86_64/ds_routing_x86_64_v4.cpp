// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 SIMD DS gather/swizzle routing with staged source reads.
// CPU-specific arithmetic stays here; public entry points select this backend.

#include "common/simd.h"

#include <immintrin.h>
#include <stdint.h>

namespace {

// Select source EXEC bits independently of destination participation.
__m512i gather(const uint32_t *a, __m512i source, uint64_t exec_mask, bool fetch_inactive = false) {
  auto value = _mm512_i32gather_epi32(source, a, 4);
  if (fetch_inactive) {
    return value;
  }
  auto upper = _mm512_cmpgt_epi32_mask(source, _mm512_set1_epi32(31));
  auto bits = _mm512_mask_blend_epi32(upper, _mm512_set1_epi32(int(uint32_t(exec_mask))),
                                      _mm512_set1_epi32(int(uint32_t(exec_mask >> 32))));
  auto active = _mm512_test_epi32_mask(
      _mm512_srlv_epi32(bits, _mm512_and_si512(source, _mm512_set1_epi32(31))),
      _mm512_set1_epi32(1));
  value = _mm512_maskz_mov_epi32(active, value);
  return value;
}

// Delay writes until every address and source has been read, including aliases.
void store(int lanes, uint64_t exec_mask, uint32_t *d, const uint32_t *result) {
  for (int lane = 0; lane < lanes; lane += 16) {
    const __mmask16 active = __mmask16(exec_mask >> lane);
    auto value = _mm512_loadu_si512(reinterpret_cast<const __m512i *>(result + lane));
    _mm512_mask_storeu_epi32(d + lane, active, value);
  }
}

} // namespace

namespace rocjitsu::arith {

// Evaluate ds bpermute for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void ds_bpermute_x86_64_v4(int lanes, int group, uint64_t exec_mask, uint32_t *d,
                           const uint32_t *addr, const uint32_t *data, uint16_t offset,
                           bool fetch_inactive) {
  uint32_t result[64];
  for (int lane = 0; lane < lanes; lane += 16) {
    auto address =
        _mm512_add_epi32(_mm512_loadu_si512(reinterpret_cast<const __m512i *>(addr + lane)),
                         _mm512_set1_epi32(offset));
    auto source = _mm512_or_si512(
        _mm512_and_si512(_mm512_srli_epi32(address, 2), _mm512_set1_epi32(group - 1)),
        _mm512_set1_epi32(lane & ~(group - 1)));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(result + lane),
                        gather(data, source, exec_mask, fetch_inactive));
  }
  store(lanes, exec_mask, d, result);
}

// Evaluate ds swizzle for the caller-selected operation. Preserve destination lanes outside
// exec_mask.
// Requires the named CPU features and validated instruction controls.
void ds_swizzle_x86_64_v4(int lanes, uint64_t exec_mask, uint32_t *d, const uint32_t *a,
                          uint16_t offset) {
  unsigned mask = offset & 31;
  unsigned count = unsigned(__builtin_popcount(mask));
  unsigned rotate = (offset >> 5) & 31;
  if (offset & 1024) {
    rotate = 0U - rotate;
  }
  uint32_t result[64];
  for (int lane = 0; lane < lanes; lane += 16) {
    auto indices =
        _mm512_add_epi32(_mm512_set1_epi32(lane),
                         _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
    __m512i source;
    if (offset >= 0xe000) {
      auto reversed = _mm512_or_si512(
          _mm512_setr_epi32(0, 16, 8, 24, 4, 20, 12, 28, 2, 18, 10, 26, 6, 22, 14, 30),
          _mm512_set1_epi32((lane & 16) >> 4));
      source = _mm512_or_si512(_mm512_srl_epi32(reversed, _mm_cvtsi32_si128(count)),
                               _mm512_and_si512(indices, _mm512_set1_epi32(mask)));
    } else if (offset >= 0xc000) {
      source = _mm512_or_si512(
          _mm512_and_si512(indices, _mm512_set1_epi32(mask)),
          _mm512_andnot_si512(_mm512_set1_epi32(mask),
                              _mm512_add_epi32(indices, _mm512_set1_epi32(int(rotate)))));
    } else if (offset & 0x8000) {
      auto selected = _mm512_and_si512(
          _mm512_srlv_epi32(_mm512_set1_epi32(offset),
                            _mm512_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6, 0, 2, 4, 6)),
          _mm512_set1_epi32(3));
      source = _mm512_or_si512(_mm512_and_si512(indices, _mm512_set1_epi32(-4)), selected);
    } else {
      source = _mm512_xor_si512(_mm512_or_si512(_mm512_and_si512(indices, _mm512_set1_epi32(mask)),
                                                _mm512_set1_epi32((offset >> 5) & 31)),
                                _mm512_set1_epi32((offset >> 10) & 31));
    }
    source = _mm512_or_si512(_mm512_and_si512(source, _mm512_set1_epi32(31)),
                             _mm512_set1_epi32(lane & 32));
    _mm512_storeu_si512(reinterpret_cast<__m512i *>(result + lane), gather(a, source, exec_mask));
  }
  store(lanes, exec_mask, d, result);
}

} // namespace rocjitsu::arith
