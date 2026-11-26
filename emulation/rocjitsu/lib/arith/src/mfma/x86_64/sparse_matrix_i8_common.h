// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared pair-grouped layout and byte-shuffle table for CDNA3/CDNA4 sparse i8 SMFMAC.
#pragma once
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

struct SmfmacI8PairLoc {
  int a_lane;
  int a_reg;
  int nib_shift;
  int b_lane;
  int b_reg;
};

// Locate a consecutive pair of 2:4 sparse groups (2*p, 2*p+1) for CDNA3/CDNA4 SMFMAC i8.
// Every pair shares one A word (4 compressed bytes), one selector byte, and two adjacent B words.
template <int M, int K> constexpr SmfmacI8PairLoc smfmac_i8_pair_loc(int p) {
  constexpr bool wide = M * K == 2048;
  int a_lane = 0, a_reg = 0, a_nib = 0;
  if constexpr (!wide) {
    a_lane = (p % (K / 16)) * M;
    a_nib = 2 * (p / (K / 16));
    a_reg = p / (K / 16);
  } else if constexpr (M == 16) {
    a_lane = (2 * (p / 8) + ((p / 2) & 1)) * 16;
    a_nib = 2 * ((p / 4) & 1) + 4 * (p & 1);
    a_reg = 2 * (p & 1) + ((p / 4) & 1);
  } else {
    a_lane = (p / 4) * 32;
    a_nib = 2 * ((p / 2) & 1) + 4 * (p & 1);
    a_reg = 2 * (p & 1) + ((p / 2) & 1);
  }
  const int b_lane = (M == 16) ? (16 * (p & 3)) : (32 * (p & 1));
  const int b_reg = 2 * (p / (M == 16 ? 4 : 2));
  return {a_lane, a_reg, 4 * a_nib, b_lane, b_reg};
}

struct SmfmacPairShufTable {
  alignas(64) uint8_t row[16][64];
};

constexpr SmfmacPairShufTable make_smfmac_pair_shuf() {
  SmfmacPairShufTable tbl{};
  for (int s = 0; s < 16; ++s) {
    const uint8_t s0 = static_cast<uint8_t>(s & 3);
    const uint8_t s1 = static_cast<uint8_t>((s >> 2) & 3);
    for (int sub = 0; sub < 4; ++sub) {
      for (int dw = 0; dw < 4; ++dw) {
        const uint8_t base = static_cast<uint8_t>(4 * dw);
        const int off = sub * 16 + dw * 4;
        tbl.row[s][off + 0] = static_cast<uint8_t>(base + s0);
        tbl.row[s][off + 1] = static_cast<uint8_t>(base + s1);
        tbl.row[s][off + 2] = static_cast<uint8_t>(base + s0);
        tbl.row[s][off + 3] = static_cast<uint8_t>(base + s1);
      }
    }
  }
  return tbl;
}

inline constexpr SmfmacPairShufTable kSmfmacPairShuf = make_smfmac_pair_shuf();

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
