// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Baseline repair routine for directed FP32 interpolation rounding.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith {

// Evaluate the interpolation stage with round-to-zero FP32 intermediate arithmetic.
float interp16_rtz_float(uint32_t a, uint32_t b, uint32_t c);

} // namespace rocjitsu::arith
