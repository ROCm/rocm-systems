// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Typed C-math calls with ISA-local wrappers, avoiding standard-library overload symbols.
#pragma once

#include "common/isa.h"

namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE {

// Scale by a power of two in the operand's precision.
inline float ldexp(float value, int exponent) { return __builtin_ldexpf(value, exponent); }
inline double ldexp(double value, int exponent) { return __builtin_ldexp(value, exponent); }

// Return the magnitude of value with the sign of sign.
inline float copysign(float value, float sign) { return __builtin_copysignf(value, sign); }
inline double copysign(double value, double sign) { return __builtin_copysign(value, sign); }

} // namespace rocjitsu::arith::RJ_ARITH_IMPL_NAMESPACE
