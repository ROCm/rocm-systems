// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rj_arith/rj_arith.h"
#include "rocjitsu/base/api.h"

#include <cstdint>

namespace rocjitsu::amdgpu {

// Keep architecture selection shared by execution and decoded-access checking.
inline uint32_t mfma_a_block_lanes(rj_code_arch_t arch, uint32_t matrix_rows) {
  return arch == ROCJITSU_CODE_ARCH_CDNA2 ? matrix_rows : 0u;
}

// The retained CDNA1/3/4 model uses a full-wave group; CDNA2 uses adjacent
// M-lane blocks. Hardware qualification of that choice is separate from
// keeping arithmetic, register reads and dependency checks consistent.
inline uint32_t permute_a_lane(uint32_t lane, uint32_t cbsz, uint32_t abid,
                               uint32_t matrix_block_lanes = 0) {
  return rj_arith_mfma_a_source_lane(lane, cbsz, abid, matrix_block_lanes);
}

/// @brief Permute the B-matrix lane based on the blgp field.
///
/// @details Per AMD ISA Table 29:
///   0: identity (l_b)
///   1: broadcast first 32 lanes  (l_b % 32)
///   2: broadcast second 32 lanes (l_b % 32 + 32)
///   3: rotate 16 lanes left      ((l_b + 16) % 64)
///   4: broadcast first 16 lanes  (l_b % 16)
///   5: broadcast second 16 lanes (l_b % 16 + 16)
///   6: broadcast third 16 lanes  (l_b % 16 + 32)
///   7: broadcast fourth 16 lanes (l_b % 16 + 48)
inline uint32_t permute_b_lane(uint32_t lane, uint32_t blgp) {
  return rj_arith_mfma_b_source_lane(lane, blgp);
}

/// Lanes read after applying a permutation to a full wave64 operand.
inline uint64_t mfma_permuted_lane_mask(auto permute) {
  uint64_t lanes = 0;
  for (uint32_t lane = 0; lane < 64; ++lane)
    lanes |= uint64_t{1} << permute(lane);
  return lanes;
}

inline uint64_t mfma_a_lane_mask(uint32_t cbsz, uint32_t abid, uint32_t matrix_block_lanes = 0) {
  return cbsz == 0 ? ~uint64_t{0} : mfma_permuted_lane_mask([&](uint32_t lane) {
    return permute_a_lane(lane, cbsz, abid, matrix_block_lanes);
  });
}

inline uint64_t mfma_b_lane_mask(uint32_t blgp) {
  return blgp == 0
             ? ~uint64_t{0}
             : mfma_permuted_lane_mask([&](uint32_t lane) { return permute_b_lane(lane, blgp); });
}

} // namespace rocjitsu::amdgpu
