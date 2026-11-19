// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Default FP-control acceptance for floating matrix instructions.

#pragma once

#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Fixed matrix models ignore these controls unless a target-specific contract
// consumes them (for example, gfx90a FP32 denormals). gfx11 also ignores DX10.
// FP16_OVFL is accepted separately and handled by the instructions that use it.
inline constexpr uint64_t kMatrixIgnoredFpFlags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                  RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                  RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_ROUND_MASK;

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
