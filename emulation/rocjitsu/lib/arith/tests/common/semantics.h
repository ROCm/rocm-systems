// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared enumeration of supported semantics modes.

#pragma once

#include "rj_arith/rj_arith.h"

#include <stdint.h>
#include <vector>

namespace rocjitsu::arith::test {

// Return the supported FMA modes without FP-status reporting.
inline std::vector<uint64_t> fma_semantics_modes() {
  return {RJ_ARITH_SEMANTICS_LOOSE, RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_ALLOW_LOOSE,
          RJ_ARITH_SEMANTICS_BIT_EXACT_EMPIRICAL_NO_FP_STATUS};
}

} // namespace rocjitsu::arith::test
