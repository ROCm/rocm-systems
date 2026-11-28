// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "agent_dbgapi_utils.h"

#include <gtest/gtest.h>

#include <limits>
#include <type_traits>

using namespace amd::debug_agent;

using exceptions_t = std::underlying_type_t<amd_dbgapi_exceptions_t>;
using stop_reasons_t = std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t>;

namespace
{

struct stop_reason_entry
{
  amd_dbgapi_wave_stop_reasons_t reason;
  const char *name;
  amd_dbgapi_exceptions_t exception;
};

#define ENTRY(name, exc)                                                      \
  {                                                                           \
    AMD_DBGAPI_WAVE_STOP_REASON_##name, #name, AMD_DBGAPI_EXCEPTION_##exc     \
  }

/* Must be extended when amd-dbgapi adds a stop reason.  */
const stop_reason_entry stop_reason_table[] = {
  ENTRY (NONE, NONE),
  ENTRY (BREAKPOINT, WAVE_TRAP),
  ENTRY (WATCHPOINT, WAVE_TRAP),
  ENTRY (SINGLE_STEP, NONE),
  ENTRY (FP_INPUT_DENORMAL, WAVE_MATH_ERROR),
  ENTRY (FP_DIVIDE_BY_0, WAVE_MATH_ERROR),
  ENTRY (FP_OVERFLOW, WAVE_MATH_ERROR),
  ENTRY (FP_UNDERFLOW, WAVE_MATH_ERROR),
  ENTRY (FP_INEXACT, WAVE_MATH_ERROR),
  ENTRY (FP_INVALID_OPERATION, WAVE_MATH_ERROR),
  ENTRY (INT_DIVIDE_BY_0, WAVE_MATH_ERROR),
  ENTRY (DEBUG_TRAP, NONE),
  ENTRY (ASSERT_TRAP, WAVE_TRAP),
  ENTRY (TRAP, WAVE_TRAP),
  ENTRY (MEMORY_VIOLATION, WAVE_MEMORY_VIOLATION),
  ENTRY (ADDRESS_ERROR, WAVE_ADDRESS_ERROR),
  ENTRY (ILLEGAL_INSTRUCTION, WAVE_ILLEGAL_INSTRUCTION),
  ENTRY (ECC_ERROR, WAVE_ABORT),
  ENTRY (FATAL_HALT, WAVE_ABORT),
#if AMD_DBGAPI_VERSION_MAJOR == 0 && AMD_DBGAPI_VERSION_MINOR < 58
  ENTRY (RESERVED, NONE),
#endif
};

#undef ENTRY

/* Highest value bit of the stop reason type, not a named stop reason.  */
constexpr stop_reasons_t unknown_stop_reason_bit
  = stop_reasons_t{ 1 } << (std::numeric_limits<stop_reasons_t>::digits - 1);

} /* namespace */

/* Each stop reason on its own maps to its expected exception.  */
TEST (AgentDbgapiUtilsTest, SingleStopReasonMapsToException)
{
  for (const auto &entry : stop_reason_table)
    {
      SCOPED_TRACE (entry.name);
      EXPECT_EQ (map_stop_reason_to_exceptions (entry.reason),
                 static_cast<exceptions_t> (entry.exception));
    }
}

/* A zero stop reason is still handled once and reports no exceptions.  */
TEST (AgentDbgapiUtilsTest, ZeroStopReasonMapsToNoExceptions)
{
  EXPECT_EQ (map_stop_reason_to_exceptions (0), exceptions_t{ 0 });
}

/* Bits mapping to different exceptions combine into both exceptions.  */
TEST (AgentDbgapiUtilsTest, MultiBitDistinctExceptions)
{
  EXPECT_EQ (
    map_stop_reason_to_exceptions (AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT
                                   | AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW),
    static_cast<exceptions_t> (AMD_DBGAPI_EXCEPTION_WAVE_TRAP
                               | AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR));
  EXPECT_EQ (
    map_stop_reason_to_exceptions (
      AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION
      | AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION
      | AMD_DBGAPI_WAVE_STOP_REASON_FATAL_HALT),
    static_cast<exceptions_t> (AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION
                               | AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION
                               | AMD_DBGAPI_EXCEPTION_WAVE_ABORT));
}

/* Bits mapping to the same exception report it once.  */
TEST (AgentDbgapiUtilsTest, MultiBitSameException)
{
  EXPECT_EQ (
    map_stop_reason_to_exceptions (AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT
                                   | AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT),
    static_cast<exceptions_t> (AMD_DBGAPI_EXCEPTION_WAVE_TRAP));
  EXPECT_EQ (map_stop_reason_to_exceptions (
               AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0
               | AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0),
             static_cast<exceptions_t> (AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR));
}

/* A bit with no exception does not change the other bits' exceptions.  */
TEST (AgentDbgapiUtilsTest, MultiBitWithNoExceptionBit)
{
  EXPECT_EQ (
    map_stop_reason_to_exceptions (AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT
                                   | AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP),
    static_cast<exceptions_t> (AMD_DBGAPI_EXCEPTION_WAVE_TRAP));
}

/* All stop reasons together give every exception they map to.  */
TEST (AgentDbgapiUtilsTest, AllStopReasonsCombined)
{
  stop_reasons_t all_reasons = 0;
  exceptions_t all_exceptions = 0;
  for (const auto &entry : stop_reason_table)
    {
      all_reasons |= entry.reason;
      all_exceptions |= entry.exception;
    }

  EXPECT_EQ (map_stop_reason_to_exceptions (all_reasons), all_exceptions);
}

/* Bits that are not a known stop reason are ignored.  */
TEST (AgentDbgapiUtilsTest, UnknownBitIgnored)
{
  EXPECT_EQ (map_stop_reason_to_exceptions (unknown_stop_reason_bit),
             exceptions_t{ 0 });
  EXPECT_EQ (
    map_stop_reason_to_exceptions (unknown_stop_reason_bit
                                   | AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT),
    static_cast<exceptions_t> (AMD_DBGAPI_EXCEPTION_WAVE_TRAP));
}

/* stop_reason_to_string gives the name without the
   AMD_DBGAPI_WAVE_STOP_REASON_ prefix.  */
TEST (AgentDbgapiUtilsTest, StopReasonName)
{
  for (const auto &entry : stop_reason_table)
    {
      SCOPED_TRACE (entry.name);
      EXPECT_STREQ (stop_reason_to_string (entry.reason), entry.name);
    }
}

/* A value with more than one stop reason bit has no name.  */
TEST (AgentDbgapiUtilsTest, MultiBitStopReasonHasNoName)
{
  EXPECT_STREQ (
    stop_reason_to_string (static_cast<amd_dbgapi_wave_stop_reasons_t> (
      AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT
      | AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT)),
    "");
}
