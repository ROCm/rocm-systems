// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// FP controls ignored by floating matrix instructions, shared by MFMA and WMMA.

#pragma once

#include "common/validation.h"
#include "rj_arith/detail/rj_arith_common.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Floating matrix arithmetic uses its fixed model, ignoring guest rounding,
// denormal and IEEE controls. gfx11 additionally ignores its DX10 control.
// FP16_OVFL is accepted separately and handled by the instructions that use it.
inline constexpr uint64_t kMatrixIgnoredFpFlags = RJ_ARITH_FP_FLUSH_INPUT_DENORMALS |
                                                  RJ_ARITH_FP_FLUSH_OUTPUT_DENORMALS |
                                                  RJ_ARITH_FP_IEEE_MODE | RJ_ARITH_FP_ROUND_MASK;

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
