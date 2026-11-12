// Copyright (c) 2020-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

/* Function implementations - see agent_dbgapi_utils.h for documentation.  */

#include "agent_dbgapi_utils.h"

namespace amd::debug_agent
{

std::underlying_type_t<amd_dbgapi_exceptions_t>
map_stop_reason_to_exceptions (
  std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t> stop_reason)
{
  std::underlying_type_t<amd_dbgapi_exceptions_t> resume_exceptions = 0;
  auto stop_reason_bits{ stop_reason };

  do
    {
      auto one_bit
        = stop_reason_bits ^ (stop_reason_bits & (stop_reason_bits - 1));
      stop_reason_bits ^= one_bit;

      switch (one_bit)
        {
        case AMD_DBGAPI_WAVE_STOP_REASON_NONE:
        case AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP:
          /* AMD_DBGAPI_EXCEPTION_NONE is 0, no assignment needed. */
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT:
        case AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT:
        case AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP:
        case AMD_DBGAPI_WAVE_STOP_REASON_TRAP:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_TRAP;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP:
          /* AMD_DBGAPI_EXCEPTION_NONE is 0, no assignment needed. */
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT:
        case AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION:
        case AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ADDRESS_ERROR;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION;
          break;

        case AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR:
        case AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT:
          resume_exceptions |= AMD_DBGAPI_EXCEPTION_WAVE_ABORT;
          break;

#if AMD_DBGAPI_VERSION_MAJOR == 0 && AMD_DBGAPI_VERSION_MINOR < 58
        case AMD_DBGAPI_WAVE_STOP_REASON_RESERVED:
          break;
#endif
        }
    }
  while (stop_reason_bits != 0);

  return resume_exceptions;
}

const char *
stop_reason_to_string (amd_dbgapi_wave_stop_reasons_t reason)
{
  switch (reason)
    {
    case AMD_DBGAPI_WAVE_STOP_REASON_NONE:
      return "NONE";
    case AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT:
      return "BREAKPOINT";
    case AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT:
      return "WATCHPOINT";
    case AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP:
      return "SINGLE_STEP";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL:
      return "FP_INPUT_DENORMAL";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0:
      return "FP_DIVIDE_BY_0";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW:
      return "FP_OVERFLOW";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW:
      return "FP_UNDERFLOW";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT:
      return "FP_INEXACT";
    case AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION:
      return "FP_INVALID_OPERATION";
    case AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0:
      return "INT_DIVIDE_BY_0";
    case AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP:
      return "DEBUG_TRAP";
    case AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP:
      return "ASSERT_TRAP";
    case AMD_DBGAPI_WAVE_STOP_REASON_TRAP:
      return "TRAP";
    case AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION:
      return "MEMORY_VIOLATION";
    case AMD_DBGAPI_WAVE_STOP_REASON_ADDRESS_ERROR:
      return "ADDRESS_ERROR";
    case AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION:
      return "ILLEGAL_INSTRUCTION";
    case AMD_DBGAPI_WAVE_STOP_REASON_ECC_ERROR:
      return "ECC_ERROR";
    case AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT:
      return "FATAL_HALT";
#if AMD_DBGAPI_VERSION_MAJOR == 0 && AMD_DBGAPI_VERSION_MINOR < 58
    case AMD_DBGAPI_WAVE_STOP_REASON_RESERVED:
      return "RESERVED";
#endif
    }

  return "";
}

} /* namespace amd::debug_agent */
