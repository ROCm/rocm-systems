// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Operand coordinates and arithmetic stage ordering shared by dense WMMA backends.
#pragma once

namespace rocjitsu::arith {

// Map an accumulation step in [0,16) to gfx12 Wave32's physical K coordinate.
// DOT4 blocks visit K=0,8,4,12; packed output narrows at each block boundary.
constexpr int wmma_gfx12_w32_k(int step) {
  return (step & 3) | ((step & 4) << 1) | ((step & 8) >> 1);
}

// Map a gfx12 integer K coordinate to its packed element within a lane.
constexpr int wmma_gfx12_integer_local(int k) { return k % 8 + 8 * (k / 16); }

// Map a product within one CLAMP stage to its logical K coordinate.
constexpr int wmma_gfx12_integer_stage_k(int stage, int local) {
  return 16 * (local / 8) + 8 * stage + local % 8;
}

} // namespace rocjitsu::arith
