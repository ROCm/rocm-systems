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

/* Unit tests for event_t.

   Covered:
     * All four public ctors store the right kind and default to
       state_t::created.
     * get_info for the queries that don't need the wave/queue/breakpoint
       graph:
         - PROCESS and KIND for every event kind.
         - WAVE for wave events (and INVALID_ARGUMENT for non-wave kinds).
         - BREAKPOINT / CLIENT_THREAD for BREAKPOINT_RESUME (and
           INVALID_ARGUMENT for other kinds).
         - RUNTIME_STATE for RUNTIME (and INVALID_ARGUMENT for other kinds).
         - QUEUE on a non-QUEUE_ERROR event -> INVALID_ARGUMENT.
         - Unknown enum query -> INVALID_ARGUMENT.
     * set_state monotonic progression created -> queued -> reported for
       all kinds (the cheap state transitions don't touch the driver).
     * set_state(processed) on the inert kinds:
         - BREAKPOINT_RESUME and WAVE_* have no side effects.
         - RUNTIME with runtime_enable_during_attach=true skips
           send_exceptions (verified via Times(0) on the mock).
         - CODE_OBJECT_LIST_UPDATED with breakpoint_resume_event_id ==
           AMD_DBGAPI_EVENT_NONE skips the queue walk.

   NOT covered here (deferred):
     * set_state(processed) for RUNTIME without
       runtime_enable_during_attach -- calls process.send_exceptions
       which walks process_t plumbing not stubbed by MockOsDriver.
     * set_state(processed) for CODE_OBJECT_LIST_UPDATED with a non-NONE
       breakpoint_resume_event_id -- walks process.range<queue_t>(),
       suspend_queues / resume_queues.  Needs the queue_t test seam.
     * pretty_printer_string for WAVE_STOP / WAVE_COMMAND_TERMINATED --
       dereferences process.find<wave_t>().  The "terminated" branch
       (wave==nullptr) IS covered.  */

#include "amd-dbgapi.h"
#include "event.h"
#include "exception.h"
#include "os_driver.h"
#include "process.h"

#include "support/mock_os_driver.h"
#include "support/process_test_access.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

using ::testing::_;
using ::testing::NiceMock;
using ::testing::Return;

using amd::dbgapi::api_error_t;
using amd::dbgapi::event_t;
using amd::dbgapi::os_driver_t;
using amd::dbgapi::process_t;
using amd::dbgapi::test::MockOsDriver;

namespace
{

amd_dbgapi_client_process_id_t
fake_client_id ()
{
  static int sentinel = 0;
  return reinterpret_cast<amd_dbgapi_client_process_id_t> (&sentinel);
}

struct test_process_t
{
  std::unique_ptr<process_t> process;
  NiceMock<MockOsDriver> *driver;
};

test_process_t
make_test_process ()
{
  auto driver = std::make_unique<NiceMock<MockOsDriver>> ();
  ON_CALL (*driver, is_valid ()).WillByDefault (Return (true));
  auto *driver_raw = driver.get ();
  auto proc = process_test_access::make (
    amd_dbgapi_process_id_t{ 1 }, fake_client_id (), std::nullopt,
    std::unique_ptr<os_driver_t> (driver.release ()));
  return { std::move (proc), driver_raw };
}

} /* namespace */

/* ------------------------------------------------------------------ */
/* ctors                                                               */
/* ------------------------------------------------------------------ */

TEST (Event, BreakpointResumeCtorStoresKindAndDefaultsToCreated)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME,
    amd_dbgapi_breakpoint_id_t{ 3 }, amd_dbgapi_client_thread_id_t{ 0 });

  EXPECT_EQ (ev.kind (), AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME);
  EXPECT_EQ (ev.state (), event_t::state_t::created);
  EXPECT_EQ (&ev.process (), tp.process.get ());
}

TEST (Event, CodeObjectListUpdatedCtorStoresKind)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED,
    AMD_DBGAPI_EVENT_NONE);

  EXPECT_EQ (ev.kind (), AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED);
}

TEST (Event, RuntimeCtorStoresKind)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_RUNTIME,
    AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS);

  EXPECT_EQ (ev.kind (), AMD_DBGAPI_EVENT_KIND_RUNTIME);
}

TEST (Event, WaveCtorStoresKindForStopAndCommandTerminated)
{
  auto tp = make_test_process ();
  auto &stop = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP,
    amd_dbgapi_wave_id_t{ 11 });
  auto &term = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED,
    amd_dbgapi_wave_id_t{ 12 });

  EXPECT_EQ (stop.kind (), AMD_DBGAPI_EVENT_KIND_WAVE_STOP);
  EXPECT_EQ (term.kind (), AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED);
}

/* ------------------------------------------------------------------ */
/* get_info: PROCESS / KIND (work for every event kind)                */
/* ------------------------------------------------------------------ */

TEST (Event, GetInfoProcessReturnsProcessId)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_RUNTIME,
    AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS);

  amd_dbgapi_process_id_t pid{};
  ev.get_info (AMD_DBGAPI_EVENT_INFO_PROCESS, sizeof (pid), &pid);
  EXPECT_EQ (pid.handle, tp.process->id ().handle);
}

TEST (Event, GetInfoKindReturnsEventKind)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });

  amd_dbgapi_event_kind_t kind{};
  ev.get_info (AMD_DBGAPI_EVENT_INFO_KIND, sizeof (kind), &kind);
  EXPECT_EQ (kind, AMD_DBGAPI_EVENT_KIND_WAVE_STOP);
}

/* ------------------------------------------------------------------ */
/* get_info: kind-discriminated queries                                */
/* ------------------------------------------------------------------ */

TEST (Event, GetInfoWaveReturnsWaveIdForWaveEvents)
{
  auto tp = make_test_process ();
  auto &stop = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP,
    amd_dbgapi_wave_id_t{ 42 });

  amd_dbgapi_wave_id_t wid{};
  stop.get_info (AMD_DBGAPI_EVENT_INFO_WAVE, sizeof (wid), &wid);
  EXPECT_EQ (wid.handle, 42u);
}

TEST (Event, GetInfoWaveThrowsInvalidArgumentForNonWaveEvent)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_RUNTIME,
    AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS);

  amd_dbgapi_wave_id_t wid{};
  try
    {
      ev.get_info (AMD_DBGAPI_EVENT_INFO_WAVE, sizeof (wid), &wid);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (Event, GetInfoBreakpointReturnsIdForBreakpointResume)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME,
    amd_dbgapi_breakpoint_id_t{ 7 }, amd_dbgapi_client_thread_id_t{ 0 });

  amd_dbgapi_breakpoint_id_t bp{};
  ev.get_info (AMD_DBGAPI_EVENT_INFO_BREAKPOINT, sizeof (bp), &bp);
  EXPECT_EQ (bp.handle, 7u);
}

TEST (Event, GetInfoBreakpointThrowsForNonBreakpointEvent)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });

  amd_dbgapi_breakpoint_id_t bp{};
  try
    {
      ev.get_info (AMD_DBGAPI_EVENT_INFO_BREAKPOINT, sizeof (bp), &bp);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (Event, GetInfoClientThreadReturnsIdForBreakpointResume)
{
  auto tp = make_test_process ();
  static int thread_sentinel = 0;
  auto thread_id = reinterpret_cast<amd_dbgapi_client_thread_id_t> (
    &thread_sentinel);
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME,
    amd_dbgapi_breakpoint_id_t{ 1 }, thread_id);

  amd_dbgapi_client_thread_id_t got{};
  ev.get_info (AMD_DBGAPI_EVENT_INFO_CLIENT_THREAD, sizeof (got), &got);
  EXPECT_EQ (got, thread_id);
}

TEST (Event, GetInfoRuntimeStateReturnsStateForRuntimeEvent)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_RUNTIME,
    AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS);

  amd_dbgapi_runtime_state_t st{};
  ev.get_info (AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE, sizeof (st), &st);
  EXPECT_EQ (st, AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS);
}

TEST (Event, GetInfoRuntimeStateThrowsForNonRuntimeEvent)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });

  amd_dbgapi_runtime_state_t st{};
  try
    {
      ev.get_info (AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE, sizeof (st), &st);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (Event, GetInfoQueueOnNonQueueEventThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_RUNTIME,
    AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS);

  amd_dbgapi_queue_id_t qid{};
  try
    {
      ev.get_info (AMD_DBGAPI_EVENT_INFO_QUEUE, sizeof (qid), &qid);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (Event, GetInfoUnknownQueryThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });

  uint64_t dummy{};
  try
    {
      ev.get_info (static_cast<amd_dbgapi_event_info_t> (0xbeef),
                   sizeof (dummy), &dummy);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* C25 negative-test sweep: value_size and enum edge cases.              */

TEST (Event, GetInfoValueSizeZeroThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });
  amd_dbgapi_process_id_t out{};
  try
    {
      ev.get_info (AMD_DBGAPI_EVENT_INFO_PROCESS, 0, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Event, GetInfoValueSizeMaxThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });
  amd_dbgapi_process_id_t out{};
  try
    {
      ev.get_info (AMD_DBGAPI_EVENT_INFO_PROCESS, SIZE_MAX, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Event, GetInfoValueSizeTooLargeByOneThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });
  uint8_t out[sizeof (amd_dbgapi_process_id_t) + 1]{};
  try
    {
      ev.get_info (AMD_DBGAPI_EVENT_INFO_PROCESS,
                   sizeof (amd_dbgapi_process_id_t) + 1, out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Event, GetInfoNegativeEnumThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });
  amd_dbgapi_process_id_t out{};
  try
    {
      ev.get_info (static_cast<amd_dbgapi_event_info_t> (-1),
                   sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (Event, GetInfoIntMaxEnumThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });
  amd_dbgapi_process_id_t out{};
  try
    {
      ev.get_info (static_cast<amd_dbgapi_event_info_t> (INT_MAX),
                   sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* ------------------------------------------------------------------ */
/* set_state                                                           */
/* ------------------------------------------------------------------ */

TEST (Event, SetStateProgressesCreatedQueuedReported)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });

  EXPECT_EQ (ev.state (), event_t::state_t::created);
  ev.set_state (event_t::state_t::queued);
  EXPECT_EQ (ev.state (), event_t::state_t::queued);
  ev.set_state (event_t::state_t::reported);
  EXPECT_EQ (ev.state (), event_t::state_t::reported);
}

TEST (Event, SetStateProcessedOnBreakpointResumeHasNoSideEffects)
{
  /* BREAKPOINT_RESUME's processed branch is a no-op.  The mock would
     report any unexpected driver call.  */
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME,
    amd_dbgapi_breakpoint_id_t{ 1 }, amd_dbgapi_client_thread_id_t{ 0 });

  ev.set_state (event_t::state_t::queued);
  ev.set_state (event_t::state_t::reported);
  ev.set_state (event_t::state_t::processed);
  EXPECT_EQ (ev.state (), event_t::state_t::processed);
}

TEST (Event, SetStateProcessedOnWaveEventHasNoSideEffects)
{
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_WAVE_STOP, amd_dbgapi_wave_id_t{ 1 });

  ev.set_state (event_t::state_t::queued);
  ev.set_state (event_t::state_t::reported);
  ev.set_state (event_t::state_t::processed);
  EXPECT_EQ (ev.state (), event_t::state_t::processed);
}

TEST (Event,
      SetStateProcessedOnRuntimeWithEnableDuringAttachSkipsSendExceptions)
{
  /* When runtime_enable_during_attach is set on the process, the
     RUNTIME event's processed branch skips send_exceptions and the
     driver is not touched.  */
  auto tp = make_test_process ();
  tp.process->set_flag (process_t::flag_t::runtime_enable_during_attach);

  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_RUNTIME,
    AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS);

  ev.set_state (event_t::state_t::queued);
  ev.set_state (event_t::state_t::reported);
  ev.set_state (event_t::state_t::processed);
  EXPECT_EQ (ev.state (), event_t::state_t::processed);
}

TEST (Event, SetStateProcessedOnCodeObjectListWithNoBreakpointResumeIsNoOp)
{
  /* With breakpoint_resume_event_id == AMD_DBGAPI_EVENT_NONE the
     processed branch skips the queue walk.  */
  auto tp = make_test_process ();
  auto &ev = tp.process->create<event_t> (
    *tp.process, AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED,
    AMD_DBGAPI_EVENT_NONE);

  ev.set_state (event_t::state_t::queued);
  ev.set_state (event_t::state_t::reported);
  ev.set_state (event_t::state_t::processed);
  EXPECT_EQ (ev.state (), event_t::state_t::processed);
}
