// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Explicit enumeration of CPU levels supported by the current host.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Compare only CPU levels; ignore any independent feature bits in either value.
// supported_flags is a previously obtained result of rj_arith_init_cpu_flags().
constexpr bool cpu_level_supported(uint64_t cpu_level, uint64_t supported_flags) {
  return (cpu_level & RJ_ARITH_CPU_MASK) <= (supported_flags & RJ_ARITH_CPU_MASK);
}

// Discover the host CPU once and return supported levels in ascending order.
inline std::vector<uint64_t> supported_cpu_levels() {
  const uint64_t supported_flags = rj_arith_init_cpu_flags();
  std::vector<uint64_t> levels;
  for (uint64_t level : {RJ_ARITH_CPU_BASELINE, RJ_ARITH_CPU_X86_64_V3, RJ_ARITH_CPU_X86_64_V4,
                         RJ_ARITH_CPU_X86_64_ZEN4}) {
    if (!cpu_level_supported(level, supported_flags)) {
      break;
    }
    levels.push_back(level);
  }
  return levels;
}

static_assert(cpu_level_supported(RJ_ARITH_CPU_BASELINE, RJ_ARITH_CPU_BASELINE));
static_assert(cpu_level_supported(RJ_ARITH_CPU_X86_64_V3, RJ_ARITH_CPU_X86_64_V4));
static_assert(!cpu_level_supported(RJ_ARITH_CPU_X86_64_ZEN4, RJ_ARITH_CPU_X86_64_V4));

// Additional feature bits must neither enable higher levels nor reject lower ones.
static_assert(!cpu_level_supported(RJ_ARITH_CPU_X86_64_V3,
                                   RJ_ARITH_CPU_BASELINE | RJ_ARITH_CPU_FEATURES_MASK));
static_assert(cpu_level_supported(RJ_ARITH_CPU_X86_64_V3 | RJ_ARITH_CPU_FEATURES_MASK,
                                  RJ_ARITH_CPU_X86_64_V4));

} // namespace rocjitsu::arith::test
