// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// AVX-512 tile selection for sixteen batched 4x4 MFMA operations.
#pragma once
#include "mfma/layout.h"
#include <immintrin.h>
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Tile starts differ in lane bits 2 and 3. Mapping tile 0 and tile 3
// identifies which of those bits survive the A-lane broadcast.
inline __m512i mfma_batch4_groups(int lane, int first_a, uint64_t controls,
                                  int matrix_block_lanes = 0) {
  int tile_bits = first_a ^ mfma_a_lane(lane + 12, controls, matrix_block_lanes);
  auto groups = _mm512_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
  return _mm512_or_si512(_mm512_and_si512(groups, _mm512_set1_epi32(tile_bits)),
                         _mm512_set1_epi32(first_a & 15));
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
