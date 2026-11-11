// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// GFX9 dense matrix register layout and MFMA source-lane controls.
#pragma once
#include "common/isa.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

struct MfmaWord {
  int reg;
  int lane;
};

// Locate one output element, including batched small matrices sharing each VGPR.
template <int M, int Batch> constexpr MfmaWord mfma_output(int row, int col, int batch) {
  constexpr int multirows = 64 / M;
  constexpr int blocks_per_reg = (64 + M * M / 4 - 1) / (M * M / 4);
  return {batch * (M * M / 64) + (row / (4 * multirows)) * 4 + row % 4,
          (batch % blocks_per_reg) * M + ((row / 4) % multirows) * blocks_per_reg * M + col};
}

// Locate an element in a flat unpacked source, before source-lane permutation.
template <int M, int K, int Batch, int PerWord>
constexpr int mfma_input_index(int index, int k, int batch) {
  constexpr int per_lane = K / (64 / (M * Batch));
  int local = k % per_lane;
  int lane = batch * M + (k / per_lane) * M * Batch + index;
  return (local / PerWord * 64 + lane) * PerWord + local % PerWord;
}

// Apply encoded A broadcast; reserved sizes retain the legacy identity behavior.
inline int mfma_a_lane(int lane, uint64_t controls) {
  int cbsz = int(controls & 7);
  if (!cbsz || cbsz > 4) {
    return lane;
  }
  int blocks = 1 << cbsz;
  return lane % (64 / blocks) + (64 / blocks) * (int(controls >> 3 & 15) % blocks);
}

// Apply the eight encoded B lane permutations, keeping the lane inside Wave64.
inline int mfma_b_lane(int lane, uint64_t controls) {
  int blgp = int(controls >> 7 & 7);
  if (blgp == 0) {
    return lane;
  }
  if (blgp <= 2) {
    return lane % 32 + (blgp - 1) * 32;
  }
  if (blgp == 3) {
    return (lane + 16) % 64;
  }
  return lane % 16 + (blgp - 4) * 16;
}

// Apply lane controls without changing the element position inside its source word.
template <int PerWord> inline int mfma_permute_input(int index, uint64_t controls, bool a) {
  int lane = index / PerWord % 64;
  int mapped = a ? mfma_a_lane(lane, controls) : mfma_b_lane(lane, controls);
  return index + PerWord * (mapped - lane);
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
