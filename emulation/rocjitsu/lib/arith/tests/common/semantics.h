// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared enumeration of supported semantics modes.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Return all supported semantics modes.
inline std::vector<uint64_t> semantics_modes() {
  return {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL};
}

} // namespace rocjitsu::arith::test
