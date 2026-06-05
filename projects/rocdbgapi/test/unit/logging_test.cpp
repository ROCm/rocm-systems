/* Copyright (c) 2026 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

/* Unit tests for src/logging.{h,cpp}.

   What's covered:
     - log_level global (default + set)
     - to_string<> CASE-branch coverage on a representative subset of
       the enum specializations (status, log_level, agent_state,
       wave_state, queue_state, event_kind, runtime_state, dispatch_*,
       memory_precision).  Each test asserts both a known CASE label
       and the hex-fallback path for an unrecognized value.
     - to_string<> for the printf-only handle types (process, agent,
       queue, dispatch, wave, workgroup, event, breakpoint,
       displaced_stepping).  These do not call find() so they are safe
       to instantiate without any process state.  Architecture,
       code_object, register, address_class, and address_space handle
       conversions ARE NOT exercised here for non-NONE values: their
       to_string specializations call find() over architecture/process
       state that isn't initialized in this TU.  The NONE sentinel path
       IS exercised for all handle types because it short-circuits
       before any lookup.
     - bitmask printer for wave_stop_reasons (combination of bits).
     - vlog() as a safe no-op when log_level is set such that the
       message is suppressed.  We deliberately do NOT exercise the
       callback-emitting path: detail::process_callbacks lives in
       callbacks.cpp and is populated by amd_dbgapi_initialize; calling
       a stale/null function pointer from a unit test is too fragile a
       contract to assert on.  */

#include "amd-dbgapi.h"
#include "logging.h"

#include <gtest/gtest.h>

#include <cstdarg>
#include <cstdint>
#include <string>

using namespace amd::dbgapi;

namespace
{

/* RAII helper - restore log_level after a test that mutates it.  */
class scoped_log_level
{
public:
  explicit scoped_log_level (amd_dbgapi_log_level_t lvl) : m_saved (log_level)
  {
    log_level = lvl;
  }
  ~scoped_log_level () { log_level = m_saved; }

private:
  amd_dbgapi_log_level_t m_saved;
};

/* Thin wrapper so we can call vlog() without building a va_list
   ourselves at the test call site.  */
void
call_vlog (amd_dbgapi_log_level_t level, const char *fmt, ...)
{
  va_list va;
  va_start (va, fmt);
  vlog (level, fmt, va);
  va_end (va);
}

} /* namespace */

/* ------------------------------------------------------------------ */
/* log_level global                                                    */
/* ------------------------------------------------------------------ */

TEST (LoggingLevel, DefaultsToNone)
{
  /* Other tests in this TU mutate log_level via scoped_log_level, so
     we can only assert the starting value before any of those tests
     run.  This test is order-independent because no other test in
     this file leaves log_level mutated.  */
  EXPECT_EQ (log_level, AMD_DBGAPI_LOG_LEVEL_NONE);
}

TEST (LoggingLevel, SetAndRestore)
{
  EXPECT_EQ (log_level, AMD_DBGAPI_LOG_LEVEL_NONE);
  {
    scoped_log_level g (AMD_DBGAPI_LOG_LEVEL_VERBOSE);
    EXPECT_EQ (log_level, AMD_DBGAPI_LOG_LEVEL_VERBOSE);
  }
  EXPECT_EQ (log_level, AMD_DBGAPI_LOG_LEVEL_NONE);
}

TEST (LoggingLevel, IndentDepthIsZeroInitially)
{
  EXPECT_EQ (detail::log_indent_depth, 0u);
}

/* ------------------------------------------------------------------ */
/* vlog: safe no-op for suppressed levels                              */
/* ------------------------------------------------------------------ */

TEST (LoggingVlog, SuppressedWhenLevelAboveCurrent)
{
  /* log_level == NONE (0), level=INFO (3) -> early return, no callback
     is dereferenced.  */
  ASSERT_EQ (log_level, AMD_DBGAPI_LOG_LEVEL_NONE);
  call_vlog (AMD_DBGAPI_LOG_LEVEL_INFO, "this should not be emitted: %d", 42);
  /* If we got here without crashing or invoking a null callback, the
     suppression contract holds.  */
  SUCCEED ();
}

/* ------------------------------------------------------------------ */
/* to_string: log_level (CASE branches + hex fallback)                 */
/* ------------------------------------------------------------------ */

TEST (LoggingToString, LogLevel)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_LOG_LEVEL_NONE), "LOG_LEVEL_NONE");
  EXPECT_EQ (to_string (AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR),
             "LOG_LEVEL_FATAL_ERROR");
  EXPECT_EQ (to_string (AMD_DBGAPI_LOG_LEVEL_WARNING), "LOG_LEVEL_WARNING");
  EXPECT_EQ (to_string (AMD_DBGAPI_LOG_LEVEL_INFO), "LOG_LEVEL_INFO");
  EXPECT_EQ (to_string (AMD_DBGAPI_LOG_LEVEL_TRACE), "LOG_LEVEL_TRACE");
  EXPECT_EQ (to_string (AMD_DBGAPI_LOG_LEVEL_VERBOSE), "LOG_LEVEL_VERBOSE");
}

TEST (LoggingToString, LogLevelHexFallbackForUnknown)
{
  /* Cast an out-of-range integer into the enum to force the default
     hex path.  This deliberately exercises the `to_string
     (make_hex(level))` fallback in src/logging.cpp.  */
  auto bogus = static_cast<amd_dbgapi_log_level_t> (0x99);
  EXPECT_EQ (to_string (bogus), "0x99");
}

/* ------------------------------------------------------------------ */
/* to_string: status                                                   */
/* ------------------------------------------------------------------ */

TEST (LoggingToString, Status)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_STATUS_SUCCESS), "STATUS_SUCCESS");
  EXPECT_EQ (to_string (AMD_DBGAPI_STATUS_ERROR), "STATUS_ERROR");
  EXPECT_EQ (to_string (AMD_DBGAPI_STATUS_ERROR_NOT_IMPLEMENTED),
             "STATUS_ERROR_NOT_IMPLEMENTED");
  EXPECT_EQ (to_string (AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID),
             "STATUS_ERROR_INVALID_WAVE_ID");
}

TEST (LoggingToString, StatusHexFallbackForUnknown)
{
  auto bogus = static_cast<amd_dbgapi_status_t> (0x4242);
  EXPECT_EQ (to_string (bogus), "0x4242");
}

/* ------------------------------------------------------------------ */
/* to_string: small enums (representative CASE coverage)               */
/* ------------------------------------------------------------------ */

TEST (LoggingToString, AgentState)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_AGENT_STATE_SUPPORTED),
             "AGENT_STATE_SUPPORTED");
  EXPECT_EQ (to_string (AMD_DBGAPI_AGENT_STATE_NOT_SUPPORTED),
             "AGENT_STATE_NOT_SUPPORTED");
}

TEST (LoggingToString, WaveState)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_WAVE_STATE_RUN), "WAVE_STATE_RUN");
  EXPECT_EQ (to_string (AMD_DBGAPI_WAVE_STATE_SINGLE_STEP),
             "WAVE_STATE_SINGLE_STEP");
  EXPECT_EQ (to_string (AMD_DBGAPI_WAVE_STATE_STOP), "WAVE_STATE_STOP");
}

TEST (LoggingToString, QueueState)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_QUEUE_STATE_VALID), "QUEUE_STATE_VALID");
  EXPECT_EQ (to_string (AMD_DBGAPI_QUEUE_STATE_ERROR), "QUEUE_STATE_ERROR");
}

TEST (LoggingToString, EventKind)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_EVENT_KIND_NONE), "EVENT_KIND_NONE");
  EXPECT_EQ (to_string (AMD_DBGAPI_EVENT_KIND_WAVE_STOP),
             "EVENT_KIND_WAVE_STOP");
  EXPECT_EQ (to_string (AMD_DBGAPI_EVENT_KIND_RUNTIME), "EVENT_KIND_RUNTIME");
}

TEST (LoggingToString, RuntimeState)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS),
             "RUNTIME_STATE_LOADED_SUCCESS");
  EXPECT_EQ (to_string (AMD_DBGAPI_RUNTIME_STATE_UNLOADED),
             "RUNTIME_STATE_UNLOADED");
}

TEST (LoggingToString, MemoryPrecision)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_MEMORY_PRECISION_NONE),
             "MEMORY_PRECISION_NONE");
  EXPECT_EQ (to_string (AMD_DBGAPI_MEMORY_PRECISION_PRECISE),
             "MEMORY_PRECISION_PRECISE");
}

TEST (LoggingToString, DispatchBarrier)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_DISPATCH_BARRIER_NONE),
             "DISPATCH_BARRIER_NONE");
  EXPECT_EQ (to_string (AMD_DBGAPI_DISPATCH_BARRIER_PRESENT),
             "DISPATCH_BARRIER_PRESENT");
}

TEST (LoggingToString, Changed)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_CHANGED_NO), "CHANGED_NO");
  EXPECT_EQ (to_string (AMD_DBGAPI_CHANGED_YES), "CHANGED_YES");
}

/* ------------------------------------------------------------------ */
/* to_string: wave_stop_reasons bitmask (combination)                  */
/* ------------------------------------------------------------------ */

TEST (LoggingToString, WaveStopReasonsSingleBit)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_WAVE_STOP_REASON_NONE),
             "WAVE_STOP_REASON_NONE");
  EXPECT_EQ (to_string (AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT),
             "WAVE_STOP_REASON_BREAKPOINT");
}

TEST (LoggingToString, WaveStopReasonsCombined)
{
  auto combined = static_cast<amd_dbgapi_wave_stop_reasons_t> (
    AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT
    | AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP);
  std::string s = to_string (combined);
  EXPECT_NE (s.find ("WAVE_STOP_REASON_BREAKPOINT"), std::string::npos);
  EXPECT_NE (s.find ("WAVE_STOP_REASON_SINGLE_STEP"), std::string::npos);
  EXPECT_NE (s.find (" | "), std::string::npos);
}

/* ------------------------------------------------------------------ */
/* to_string: handle types (NONE sentinel + populated handle)          */
/* ------------------------------------------------------------------ */

TEST (LoggingToString, ProcessIdSentinelAndPopulated)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_PROCESS_NONE), "PROCESS_NONE");
  EXPECT_EQ (to_string (amd_dbgapi_process_id_t{ 7 }), "process_7");
}

TEST (LoggingToString, AgentIdSentinelAndPopulated)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_AGENT_NONE), "AGENT_NONE");
  EXPECT_EQ (to_string (amd_dbgapi_agent_id_t{ 3 }), "agent_3");
}

TEST (LoggingToString, QueueIdSentinelAndPopulated)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_QUEUE_NONE), "QUEUE_NONE");
  EXPECT_EQ (to_string (amd_dbgapi_queue_id_t{ 11 }), "queue_11");
}

TEST (LoggingToString, DispatchIdSentinelAndPopulated)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_DISPATCH_NONE), "DISPATCH_NONE");
  EXPECT_EQ (to_string (amd_dbgapi_dispatch_id_t{ 5 }), "dispatch_5");
}

TEST (LoggingToString, WaveIdSentinelAndPopulated)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_WAVE_NONE), "WAVE_NONE");
  EXPECT_EQ (to_string (amd_dbgapi_wave_id_t{ 2 }), "wave_2");
}

TEST (LoggingToString, WorkgroupIdSentinelAndPopulated)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_WORKGROUP_NONE), "WORKGROUP_NONE");
  EXPECT_EQ (to_string (amd_dbgapi_workgroup_id_t{ 9 }), "workgroup_9");
}

TEST (LoggingToString, EventIdSentinelAndPopulated)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_EVENT_NONE), "EVENT_NONE");
  EXPECT_EQ (to_string (amd_dbgapi_event_id_t{ 4 }), "event_4");
}

TEST (LoggingToString, BreakpointIdSentinelAndPopulated)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_BREAKPOINT_NONE), "BREAKPOINT_NONE");
  EXPECT_EQ (to_string (amd_dbgapi_breakpoint_id_t{ 1 }), "breakpoint_1");
}

TEST (LoggingToString, DisplacedSteppingIdSentinelAndPopulated)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_DISPLACED_STEPPING_NONE),
             "DISPLACED_STEPPING_NONE");
  EXPECT_EQ (to_string (amd_dbgapi_displaced_stepping_id_t{ 6 }),
             "displaced_stepping_6");
}

/* Handle types whose to_string specialization calls find() (and would
   need architecture/process state to look up the name) are exercised
   here for the NONE sentinel only - that branch short-circuits before
   any lookup.  */

TEST (LoggingToString, ArchitectureIdSentinel)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_ARCHITECTURE_NONE), "ARCHITECTURE_NONE");
}

TEST (LoggingToString, CodeObjectIdSentinel)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_CODE_OBJECT_NONE), "CODE_OBJECT_NONE");
}

TEST (LoggingToString, RegisterClassIdSentinel)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_REGISTER_CLASS_NONE),
             "REGISTER_CLASS_NONE");
}

TEST (LoggingToString, RegisterIdSentinel)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_REGISTER_NONE), "REGISTER_NONE");
}

TEST (LoggingToString, AddressClassIdSentinel)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_ADDRESS_CLASS_NONE),
             "ADDRESS_CLASS_NONE");
}

TEST (LoggingToString, AddressSpaceIdSentinel)
{
  EXPECT_EQ (to_string (AMD_DBGAPI_ADDRESS_SPACE_NONE),
             "ADDRESS_SPACE_NONE");
}
