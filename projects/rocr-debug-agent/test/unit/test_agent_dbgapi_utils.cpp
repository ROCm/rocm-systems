// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "agent_dbgapi_utils.h"

#include <gtest/gtest.h>

#include <type_traits>

using namespace amd::debug_agent;

using exceptions_t = std::underlying_type_t<amd_dbgapi_exceptions_t>;

/* A wave with no stop reason reports no exceptions.  */
TEST (AgentDbgapiUtilsTest, NoStopReasonMapsToNoExceptions)
{
  EXPECT_EQ (map_stop_reason_to_exceptions (AMD_DBGAPI_WAVE_STOP_REASON_NONE),
             static_cast<exceptions_t> (AMD_DBGAPI_EXCEPTION_NONE));
}

/* A breakpoint stop is reported as a wave trap exception.  */
TEST (AgentDbgapiUtilsTest, BreakpointMapsToWaveTrap)
{
  EXPECT_EQ (
    map_stop_reason_to_exceptions (AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT),
    static_cast<exceptions_t> (AMD_DBGAPI_EXCEPTION_WAVE_TRAP));
}

/* stop_reason_to_string gives the name without the
   AMD_DBGAPI_WAVE_STOP_REASON_ prefix.  */
TEST (AgentDbgapiUtilsTest, BreakpointName)
{
  EXPECT_STREQ (stop_reason_to_string (AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT),
                "BREAKPOINT");
}
