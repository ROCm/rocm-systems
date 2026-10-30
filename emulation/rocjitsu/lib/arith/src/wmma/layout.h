// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Operand coordinates and arithmetic stage ordering shared by dense WMMA and sparse SWMMAC
// backends.
#pragma once

#include "common/isa.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

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
