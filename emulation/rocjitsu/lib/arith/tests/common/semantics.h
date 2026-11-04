// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared enumeration of supported semantics modes, including compatibility.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Return all defined semantics modes, including temporary compatibility mode.
inline std::vector<uint64_t> semantics_modes() {
  return {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL, RJ_ARITH_SEMANTICS_COMPATIBILITY};
}

} // namespace rocjitsu::arith::test
