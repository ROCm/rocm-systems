// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Arithmetic stage ordering shared by dense WMMA backends.
#pragma once

namespace rocjitsu::arith {

// Map an accumulation step in [0,16) to gfx12 Wave32's physical K coordinate.
// DOT4 blocks visit K=0,8,4,12; packed output narrows at each block boundary.
constexpr int wmma_gfx12_w32_k(int step) {
  return (step & 3) | ((step & 4) << 1) | ((step & 8) >> 1);
}

} // namespace rocjitsu::arith
