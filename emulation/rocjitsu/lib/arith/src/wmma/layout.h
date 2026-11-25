// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Operand coordinates and arithmetic stage ordering shared by dense WMMA and sparse SWMMAC
// backends.
#pragma once

#include "common/isa.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Location of a contiguous sixteen-lane section within a WMMA operand.
struct WmmaRegisterSection {
  int reg;
  int lane;
};

// Each packed A/B word supplies a K pair. gfx11 replicates the same inputs
// across all sixteen-lane groups, so one copy supplies the whole tile.
template <int Lanes, bool Rdna3> constexpr WmmaRegisterSection wmma_half_input(int k) {
  constexpr int Registers = 256 / Lanes;
  return Rdna3 ? WmmaRegisterSection{k / 2, 0}
               : WmmaRegisterSection{(k % Registers) / 2, 16 * (k / Registers)};
}

// Location of an unpacked output row. gfx12 packed C/D combine neighboring
// register rows into low and high halfwords at the same lane offset.
template <int Lanes, bool Rdna3> constexpr WmmaRegisterSection wmma_half_output(int row) {
  constexpr int Groups = Lanes / 16;
  if constexpr (Rdna3) {
    return {row / Groups, 16 * (row % Groups)};
  } else {
    return {row % (256 / Lanes), 16 * (row / 8) + (Lanes == 64 ? 32 * ((row / 4) % 2) : 0)};
  }
}

// Map an accumulation step in [0,16) to gfx12 Wave32's physical K coordinate.
// DOT4 blocks visit K=0,8,4,12; packed output narrows at each block boundary.
constexpr int wmma_gfx12_w32_k(int step) {
  return (step & 3) | ((step & 4) << 1) | ((step & 8) >> 1);
}

// Map a gfx12 packed 4/8-bit K coordinate to its element index within a lane.
constexpr int wmma_gfx12_packed_local(int k) { return k % 8 + 8 * (k / 16); }

// Map a product within one CLAMP stage to dense K or a sparse compressed position.
constexpr int wmma_gfx12_integer_stage_k(int stage, int local) {
  return 16 * (local / 8) + 8 * stage + local % 8;
}

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
