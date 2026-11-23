// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Share gfx1150 constant-compare operand reads across execution, DPP and wait checks.
#pragma once

#include "rj_arith/detail/rj_arith_common.h"

#include <stdint.h>

namespace rocjitsu::amdgpu {

// Constant predicates read operands only for Compatibility's inherited FP status.
inline bool gfx1150_constant_compare_reads_inputs(uint64_t flags, bool dx10_clamp) {
  return !dx10_clamp && (flags & RJ_ARITH_SEMANTICS_MASK) == RJ_ARITH_SEMANTICS_COMPATIBILITY;
}

} // namespace rocjitsu::amdgpu
