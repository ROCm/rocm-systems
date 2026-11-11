// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/validation.h"
#include "rj_arith/rj_arith.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Floating matrix arithmetic uses its fixed model, ignoring guest rounding,
// denormal and IEEE controls. gfx11 additionally ignores its DX10 control.
// FP16_OVFL is accepted separately and has no effect on FP32 matrix results.
inline constexpr uint64_t kMatrixIgnoredFpFlags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                  RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                  RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_ROUND_MASK;

inline constexpr uint32_t kWmmaInstructionFlags = RJ_ARITH_WMMA_NEG_LO_A | RJ_ARITH_WMMA_NEG_LO_B |
                                                  RJ_ARITH_WMMA_NEG_C | RJ_ARITH_WMMA_NEG_HI_A |
                                                  RJ_ARITH_WMMA_NEG_HI_B | RJ_ARITH_WMMA_ABS_C;

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
