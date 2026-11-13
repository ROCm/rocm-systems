// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace rocjitsu::amdgpu {

/// AMD's matrix calculator at 2ef91896 applies adjacent M-lane block groups
/// to gfx908, gfx90a and gfx942. CDNA2 follows that mapping. MI100 and MI200
/// describe the same fields. The gfx908 model retains full-wave broadcast
/// pending a hardware comparison. The gfx942/gfx950 model follows the
/// full-wave formula in MI300 §7.1.5.1 and CDNA4 §7.1.6.1. Both guides'
/// §13.3.6.1 ABID tables describe adjacent groups.
inline uint32_t permute_a_lane(uint32_t lane, uint32_t cbsz, uint32_t abid,
                               uint32_t matrix_block_lanes = 0) {
  if (cbsz == 0 || cbsz > 4)
    return lane;
  const uint32_t blocks = 1u << cbsz;
  const uint32_t width = matrix_block_lanes ? matrix_block_lanes : 64u / blocks;
  if (width > 64u / blocks)
    return lane;
  const uint32_t group_size = width * blocks;
  return (lane / group_size) * group_size + (abid % blocks) * width + (lane % width);
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
  switch (blgp) {
  case 0:
    return lane;
  case 1:
    return lane % 32;
  case 2:
    return lane % 32 + 32;
  case 3:
    return (lane + 16) % 64;
  case 4:
    return lane % 16;
  case 5:
    return lane % 16 + 16;
  case 6:
    return lane % 16 + 32;
  case 7:
    return lane % 16 + 48;
  default:
    return lane;
  }
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
