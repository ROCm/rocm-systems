// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared WMMA operand modifiers, separate from architecture-neutral validation.

#pragma once

#include "rj_arith/detail/rj_arith_wmma.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

inline constexpr uint32_t kWmmaInstructionFlags = RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_LO_B |
                                                  RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_NEG_HI_A |
                                                  RJ_ARITH_WMMA_NEG_HI_B | RJ_ARITH_WMMA_ABS_C;

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
