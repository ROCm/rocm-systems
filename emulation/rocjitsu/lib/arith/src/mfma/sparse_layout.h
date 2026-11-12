// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Inherited sparse A/metadata and dense B register layouts for floating SMFMAC.
#pragma once

#include "common/isa.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
enum class SmfmacLayout { Cdna3F16, Cdna4F16, Cdna3Fp8, Cdna4Fp8 };

struct SmfmacSparseLoc {
  int lane;
  int nibble;
  int element;
};

struct SmfmacDenseLoc {
  int lane;
  int element;
};

/// The sparse A element and its 2:4 selector occupy the same source lane.
/// These four layouts preserve the original scalar executor's register permutations.
template <SmfmacLayout Layout, int M, int K>
inline SmfmacSparseLoc smfmac_sparse_loc(int row, int q, int s) {
  if constexpr (Layout == SmfmacLayout::Cdna3F16 || Layout == SmfmacLayout::Cdna4F16) {
    constexpr int groups_per_lane = Layout == SmfmacLayout::Cdna3F16 ? 2 : 4;
    return {(q / groups_per_lane) * M + row, q % groups_per_lane,
            (2 * q + s) % (2 * groups_per_lane)};
  } else if constexpr (Layout == SmfmacLayout::Cdna3Fp8) {
    constexpr int half_groups = K / 8;
    return {((q % half_groups) / 2) * M + row, 2 * (q / half_groups) + (q % 2),
            4 * (q / half_groups) + (2 * q + s) % 4};
  } else if constexpr (M == 16) {
    const int lane_group = 2 * (q / 16) + ((q / 4) % 2);
    const int nibble = 2 * ((q / 8) % 2) + 4 * ((q % 4) / 2) + (q % 2);
    const int byte = 4 * (2 * ((q / 2) % 2) + ((q / 8) % 2)) + (2 * q + s) % 4;
    return {lane_group * M + row, nibble, byte};
  } else {
    static_assert(M == 32);
    const int lane_group = q / 8;
    const int nibble = (q % 2) + 2 * ((q / 4) % 2) + 4 * ((q / 2) % 2);
    const int byte = 4 * (2 * ((q / 2) % 2) + ((q / 4) % 2)) + (2 * q + s) % 4;
    return {lane_group * M + row, nibble, byte};
  }
}

// Locate a dense B element for one selected inner-product coordinate.
template <SmfmacLayout Layout, int N> inline SmfmacDenseLoc smfmac_dense_loc(int k, int col) {
  if constexpr (Layout == SmfmacLayout::Cdna3F16) {
    if constexpr (N == 16) {
      return {(k / 8) * 16 + col, k % 8};
    } else {
      return {16 * ((col / 16) + 2 * (k / 8)) + (col % 16), k % 8};
    }
  } else if constexpr (N == 16) {
    return {16 * ((k % 32) / 8) + col, 8 * (k / 32) + (k % 8)};
  } else {
    static_assert(N == 32);
    return {32 * ((k % 16) / 8) + col, 8 * (k / 16) + (k % 8)};
  }
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
