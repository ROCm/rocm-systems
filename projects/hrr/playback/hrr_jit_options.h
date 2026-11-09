/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once


#include <cstdint>

namespace hrr {

// True if the value paired with JIT option `option` is an integer, so the recorded value
// can pass through. False for options whose value is a pointer.
constexpr bool jit_option_value_is_integer(int option) {
  switch (option) {
    case 0:      // hipJitOptionMaxRegisters
    case 1:      // hipJitOptionThreadsPerBlock
    case 4:      // hipJitOptionInfoLogBufferSizeBytes
    case 6:      // hipJitOptionErrorLogBufferSizeBytes
    case 7:      // hipJitOptionOptimizationLevel
    case 8:      // hipJitOptionTargetFromContext
    case 9:      // hipJitOptionTarget
    case 10:     // hipJitOptionFallbackStrategy
    case 11:     // hipJitOptionGenerateDebugInfo
    case 12:     // hipJitOptionLogVerbose
    case 13:     // hipJitOptionGenerateLineInfo
    case 14:     // hipJitOptionCacheMode
    case 15:     // hipJitOptionSm3xOpt
    case 16:     // hipJitOptionFastCompile
    case 19:     // hipJitOptionGlobalSymbolCount
    case 20:     // hipJitOptionLto
    case 21:     // hipJitOptionFtz
    case 22:     // hipJitOptionPrecDiv
    case 23:     // hipJitOptionPrecSqrt
    case 24:     // hipJitOptionFma
    case 25:     // hipJitOptionPositionIndependentCode
    case 26:     // hipJitOptionMinCTAPerSM
    case 27:     // hipJitOptionMaxThreadsPerBlock
    case 28:     // hipJitOptionOverrideDirectiveValues
    case 29:     // hipJitOptionNumOptions
    case 10001:  // hipJitOptionIRtoISAOptCountExt
      return true;
    default:
      // 2 WallTime, 3 InfoLogBuffer, 5 ErrorLogBuffer, 17 GlobalSymbolNames, 18 GlobalSymbolAddresses
      // and 10000 IRtoISAOptExt carry pointers; anything unknown is treated as one.
      return false;
  }
}

// True if any of the first `n` options carries a pointer (or is unknown).
template <typename Option>
bool jit_options_carry_pointer(const Option* options, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i)
    if (!jit_option_value_is_integer(static_cast<int>(options[i]))) return true;
  return false;
}

}  // namespace hrr
