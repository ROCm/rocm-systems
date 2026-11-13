// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// gfx1250 FP8/BF8 operand indexing for K64 and K128 Wave32 matrices.
#pragma once
#include "common/internal.h"
namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {
// Locate a byte in a flattened packed Wave32 operand; K blocks alternate lane halves.
inline int gfx1250_fp8_index(int index, int k) {
  int local = (k / 32) * 16 + k % 16;
  return (local / 4 * 32 + index + 16 * (k / 16 % 2)) * 4 + local % 4;
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
