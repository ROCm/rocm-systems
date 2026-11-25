// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Operand alignment for width-sensitive wave operations.
#pragma once
#include "common/isa.h"
#include "rj_arith/detail/rj_arith_common.h"
#include <stdint.h>

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// A VGPR file normally gives the whole wave's operands the same alignment.
// Test the streaming source and, for three-input selection, its output.
inline bool aligned_for_512([[maybe_unused]] const uint32_t *first,
                            [[maybe_unused]] const uint32_t *second = nullptr) {
#if defined(RJ_ARITH_HAVE_X86_64_V3)
  return ((reinterpret_cast<uintptr_t>(first) | reinterpret_cast<uintptr_t>(second)) & 63) == 0;
#else
  return true;
#endif
}

// Select the register width once. Both supported levels then take the same
// dispatch path when a narrower kernel handles split cache-line accesses.
inline uint64_t wave_cpu_level(uint64_t flags, bool use_wide) {
  uint64_t level = flags & RJ_ARITH_CPU_MASK;
  uint64_t limit = use_wide ? RJ_ARITH_CPU_X86_64_ZEN4 : RJ_ARITH_CPU_X86_64_V3;
  return level < limit ? level : limit;
}
} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
