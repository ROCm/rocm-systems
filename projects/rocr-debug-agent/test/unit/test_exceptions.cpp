/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimers in
      the documentation and/or other materials provided with the distribution.
    - Neither the names of Advanced Micro Devices, Inc,
      nor the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior written
      permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS WITH THE SOFTWARE.  */

#include "agent_utils.h"

#include <gtest/gtest.h>

using namespace amd::debug_agent;

/* Parameterized test for stop_reason_to_string. */
struct StopReasonStringParam
{
  amd_dbgapi_wave_stop_reasons_t reason;
  const char *expected_string;
};

class StopReasonToStringTest : public ::testing::TestWithParam<StopReasonStringParam>
{
};

TEST_P (StopReasonToStringTest, ReturnsCorrectString)
{
  auto param = GetParam ();
  EXPECT_STREQ (stop_reason_to_string (param.reason), param.expected_string);
}

INSTANTIATE_TEST_SUITE_P (
    AllStopReasons, StopReasonToStringTest,
    ::testing::Values (
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_NONE, "NONE" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT, "BREAKPOINT" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT, "WATCHPOINT" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP, "SINGLE_STEP" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_TRAP, "TRAP" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION,
                               "FP_INVALID_OPERATION" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0,
                               "FP_DIVIDE_BY_0" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW, "FP_OVERFLOW" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW, "FP_UNDERFLOW" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT, "FP_INEXACT" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL,
                               "FP_INPUT_DENORMAL" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0,
                               "INT_DIVIDE_BY_0" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION,
                               "ILLEGAL_INSTRUCTION" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION,
                               "MEMORY_VIOLATION" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP, "ASSERT_TRAP" },
        StopReasonStringParam{ AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP, "DEBUG_TRAP" }));

/* Parameterized test for single-bit stop reason to exception mapping. */
struct MapStopReasonParam
{
  std::underlying_type_t<amd_dbgapi_wave_stop_reasons_t> stop_reason;
  std::underlying_type_t<amd_dbgapi_exceptions_t> expected_exception;
};

class MapStopReasonSingleBitTest : public ::testing::TestWithParam<MapStopReasonParam>
{
};

TEST_P (MapStopReasonSingleBitTest, MapsCorrectly)
{
  auto param = GetParam ();
  auto result = map_stop_reason_to_exceptions (param.stop_reason);
  EXPECT_EQ (result, param.expected_exception);
}

INSTANTIATE_TEST_SUITE_P (
    SingleBitReasons, MapStopReasonSingleBitTest,
    ::testing::Values (
        MapStopReasonParam{ AMD_DBGAPI_WAVE_STOP_REASON_NONE,
                            AMD_DBGAPI_EXCEPTION_NONE },
        MapStopReasonParam{ AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT,
                            AMD_DBGAPI_EXCEPTION_WAVE_TRAP },
        MapStopReasonParam{ AMD_DBGAPI_WAVE_STOP_REASON_TRAP,
                            AMD_DBGAPI_EXCEPTION_WAVE_TRAP },
        MapStopReasonParam{ AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION,
                            AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR },
        MapStopReasonParam{ AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0,
                            AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR },
        MapStopReasonParam{ AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION,
                            AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION },
        MapStopReasonParam{ AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION,
                            AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION }));

TEST (MapStopReasonTest, MultiBit_TrapAndFpError)
{
  /* Multi-bit test: BREAKPOINT | FP_OVERFLOW. */
  auto result = map_stop_reason_to_exceptions (
      AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT | AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW);
  EXPECT_EQ (result,
             AMD_DBGAPI_EXCEPTION_WAVE_TRAP | AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR);
}

TEST (MapStopReasonTest, MultiBit_AllFpErrors)
{
  /* All floating-point exceptions combined. */
  auto result = map_stop_reason_to_exceptions (
      AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION
      | AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0
      | AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW
      | AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW
      | AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT
      | AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL);
  EXPECT_EQ (result, AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR);
}

TEST (MapStopReasonTest, MultiBit_TrapAndMemory)
{
  /* Trap and memory violation combined. */
  auto result = map_stop_reason_to_exceptions (
      AMD_DBGAPI_WAVE_STOP_REASON_TRAP
      | AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION);
  EXPECT_EQ (result,
             AMD_DBGAPI_EXCEPTION_WAVE_TRAP
             | AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION);
}

TEST (MapStopReasonTest, MultiBit_ThreeReasons)
{
  /* Test 3 stop reasons combined: TRAP | FP_OVERFLOW | MEMORY_VIOLATION. */
  auto result = map_stop_reason_to_exceptions (
      AMD_DBGAPI_WAVE_STOP_REASON_TRAP
      | AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW
      | AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION);

  EXPECT_EQ (result,
             AMD_DBGAPI_EXCEPTION_WAVE_TRAP
                 | AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR
                 | AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION);
}

TEST (MapStopReasonTest, MultiBit_AllExceptionTypes)
{
  /* Combine reasons that map to all exception types. */
  auto result = map_stop_reason_to_exceptions (
      AMD_DBGAPI_WAVE_STOP_REASON_TRAP
      | AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW
      | AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION
      | AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION);

  EXPECT_EQ (result,
             AMD_DBGAPI_EXCEPTION_WAVE_TRAP
                 | AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR
                 | AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION
                 | AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION);
}

TEST (MapStopReasonTest, EdgeCase_AllZeroBits)
{
  /* Explicit test for stop_reason = 0. */
  auto result = map_stop_reason_to_exceptions (
      static_cast<amd_dbgapi_wave_stop_reasons_t> (0));
  EXPECT_EQ (result, 0);
}
