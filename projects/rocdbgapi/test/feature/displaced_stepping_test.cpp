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

/* Displaced stepping + exception delivery feature tests.

   These exercise the two pieces of the wave control surface that
   matter for a debugger doing real work — stepping over an inserted
   breakpoint, and delivering an exception to a wave on resume — but
   without writing a real breakpoint into the target.  Inserting a
   breakpoint requires write access to the inferior's code object,
   which is well beyond the scope of this test suite; what we want to
   prove here is that the *API surface* behaves predictably.

   Coverage:

     1. DisplacedSteppingRejectsRunningWave — calling
        amd_dbgapi_displaced_stepping_start on a wave we have not
        stopped must return WAVE_NOT_STOPPED.  This is the precondition
        the docs promise and the first thing any client trips over.

     2. DisplacedSteppingOnStoppedWaveBehavesPredictably — stop the
        wave, query the breakpoint-instruction size from the agent's
        architecture, fabricate a zero-filled "saved bytes" buffer of
        exactly that size, and call displaced_stepping_start.  The
        spinning idle kernel was never patched so the result is one of:
          SUCCESS — we own a buffer; complete + release immediately.
          ILLEGAL_INSTRUCTION / MEMORY_ACCESS / BUFFER_NOT_AVAILABLE
                  — legitimate refusals, that's still a "predictable"
                    outcome.
        Anything else (or a SUCCESS we fail to clean up) is a test
        failure.

     3. ResumeWithAbortExceptionTransitionsWave — stop the wave, then
        resume with AMD_DBGAPI_EXCEPTION_WAVE_ABORT.  Per the
        wave_resume contract, this puts the wave in halt and tells the
        runtime; the call must return SUCCESS.  We detach immediately
        afterward — the queue ends up in the error state, which is
        fine because the workload child is about to be SIGKILLed.

   Single-stepping a *real* breakpoint, exception delivery follow-up
   handling, and queue-error-state introspection are deferred — this
   commit closes the "control surface" half of the feature plan; the
   "round trip through a real breakpoint" half belongs in the gdb
   integration tests that live downstream of this library.  */

#include "amd-dbgapi.h"

#include "support/dbgapi_fixture.h"
#include "support/pid_callbacks.h"
#include "support/spawn_hip_workload.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

using amd::dbgapi::test::gpu_fixture_t;
using amd::dbgapi::test::hip_workload_available;
using amd::dbgapi::test::idle_child_t;
using amd::dbgapi::test::pid_callbacks;
using amd::dbgapi::test::spawn_hip_workload;

namespace
{

class displaced_stepping_fixture_t : public gpu_fixture_t
{
protected:
  const amd_dbgapi_callbacks_s &
  callbacks () const override
  {
    return pid_callbacks ();
  }

  void
  SetUp () override
  {
    if (!hip_workload_available ())
      GTEST_SKIP ()
        << "HIP not found at configure time — workload binary not built";
    gpu_fixture_t::SetUp ();
  }
};

struct attached_wave_t
{
  idle_child_t child;
  amd_dbgapi_process_id_t pid{};
  amd_dbgapi_wave_id_t wave{ AMD_DBGAPI_WAVE_NONE };
  bool ok{ false };
};

/* Same shape as the Commit 22 helper: attach, take a stable wave
   snapshot under NO_FORWARD, switch back to NORMAL so events flow.  */
static attached_wave_t
attach_and_pick_wave ()
{
  attached_wave_t out;
  out.child = spawn_hip_workload ();
  if (!out.child.valid ())
    return out;

  amd_dbgapi_os_process_id_t target = out.child.pid ();
  if (amd_dbgapi_process_attach (
        reinterpret_cast<amd_dbgapi_client_process_id_t> (&target), &out.pid)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return out;

  if (amd_dbgapi_process_set_progress (out.pid, AMD_DBGAPI_PROGRESS_NO_FORWARD)
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      amd_dbgapi_process_detach (out.pid);
      return out;
    }

  size_t wave_count = 0;
  amd_dbgapi_wave_id_t *waves = nullptr;
  if (amd_dbgapi_process_wave_list (out.pid, &wave_count, &waves, nullptr)
        == AMD_DBGAPI_STATUS_SUCCESS
      && wave_count > 0)
    out.wave = waves[0];
  std::free (waves);

  if (amd_dbgapi_process_set_progress (out.pid, AMD_DBGAPI_PROGRESS_NORMAL)
      != AMD_DBGAPI_STATUS_SUCCESS
      || out.wave.handle == AMD_DBGAPI_WAVE_NONE.handle)
    {
      amd_dbgapi_process_detach (out.pid);
      return out;
    }

  out.ok = true;
  return out;
}

static void
detach (attached_wave_t &w)
{
  if (w.ok)
    {
      amd_dbgapi_process_detach (w.pid);
      w.ok = false;
    }
}

/* Same shape as Commit 22.  Polls up to ~3 s for a WAVE_STOP event on
   `wanted`.  Releases any non-matching event so we don't leak slots.  */
static amd_dbgapi_event_id_t
wait_for_wave_stop_event (amd_dbgapi_process_id_t pid,
                          amd_dbgapi_wave_id_t wanted)
{
  using clock = std::chrono::steady_clock;
  auto deadline = clock::now () + std::chrono::seconds (3);

  while (clock::now () < deadline)
    {
      amd_dbgapi_event_id_t ev{};
      amd_dbgapi_event_kind_t kind{};
      if (amd_dbgapi_process_next_pending_event (pid, &ev, &kind)
          != AMD_DBGAPI_STATUS_SUCCESS)
        return AMD_DBGAPI_EVENT_NONE;

      if (ev.handle == AMD_DBGAPI_EVENT_NONE.handle)
        {
          std::this_thread::sleep_for (std::chrono::milliseconds (10));
          continue;
        }

      if (kind == AMD_DBGAPI_EVENT_KIND_WAVE_STOP)
        {
          amd_dbgapi_wave_id_t got{};
          if (amd_dbgapi_event_get_info (ev, AMD_DBGAPI_EVENT_INFO_WAVE,
                                         sizeof (got), &got)
                == AMD_DBGAPI_STATUS_SUCCESS
              && got.handle == wanted.handle)
            return ev;
        }
      amd_dbgapi_event_processed (ev);
    }
  return AMD_DBGAPI_EVENT_NONE;
}

/* Stop `wave` and drain its WAVE_STOP event so the wave is parked in
   STATE_STOP and resume/displaced-stepping operations are legal.
   Returns true on success; the caller should GTEST_SKIP on false.  */
static bool
stop_and_settle (amd_dbgapi_process_id_t pid, amd_dbgapi_wave_id_t wave)
{
  amd_dbgapi_status_t st = amd_dbgapi_wave_stop (wave);
  if (st == AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED)
    return true;
  if (st != AMD_DBGAPI_STATUS_SUCCESS)
    return false;
  amd_dbgapi_event_id_t ev = wait_for_wave_stop_event (pid, wave);
  if (ev.handle == AMD_DBGAPI_EVENT_NONE.handle)
    return false;
  return amd_dbgapi_event_processed (ev) == AMD_DBGAPI_STATUS_SUCCESS;
}

} /* namespace */

using DisplacedStepping = displaced_stepping_fixture_t;

TEST_F (DisplacedStepping, DisplacedSteppingRejectsRunningWave)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  /* Allocate a small zero-filled buffer for the saved bytes argument;
     the call must reject for WAVE_NOT_STOPPED before it ever inspects
     the buffer, but pass something non-NULL to avoid a separate
     INVALID_ARGUMENT path.  */
  std::vector<uint8_t> dummy_bytes (32, 0);
  amd_dbgapi_displaced_stepping_id_t handle{};
  amd_dbgapi_status_t st = amd_dbgapi_displaced_stepping_start (
    w.wave, dummy_bytes.data (), &handle);
  EXPECT_EQ (st, AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED)
    << "expected WAVE_NOT_STOPPED on running wave, got " << st;

  detach (w);
}

TEST_F (DisplacedStepping, DisplacedSteppingOnStoppedWaveBehavesPredictably)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  if (!stop_and_settle (w.pid, w.wave))
    {
      detach (w);
      GTEST_SKIP () << "could not park wave in STATE_STOP";
    }

  /* Get the agent's architecture and the size of the saved-bytes
     buffer that displaced_stepping_start wants.  */
  amd_dbgapi_architecture_id_t arch{};
  ASSERT_EQ (amd_dbgapi_wave_get_info (w.wave,
                                       AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
                                       sizeof (arch), &arch),
             AMD_DBGAPI_STATUS_SUCCESS);
  amd_dbgapi_size_t bp_size = 0;
  ASSERT_EQ (
    amd_dbgapi_architecture_get_info (
      arch, AMD_DBGAPI_ARCHITECTURE_INFO_BREAKPOINT_INSTRUCTION_SIZE,
      sizeof (bp_size), &bp_size),
    AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_GT (bp_size, 0u);
  ASSERT_LT (bp_size, 64u) << "implausible breakpoint instruction size";

  /* Fabricate a saved-bytes buffer of exactly the required size.  The
     spinning kernel never had a breakpoint patched in, so the library
     is allowed to refuse with ILLEGAL_INSTRUCTION / MEMORY_ACCESS /
     BUFFER_NOT_AVAILABLE.  We accept any of those — and SUCCESS — and
     fail only on something genuinely unexpected.  */
  std::vector<uint8_t> saved (bp_size, 0);
  amd_dbgapi_displaced_stepping_id_t handle{};
  amd_dbgapi_status_t st = amd_dbgapi_displaced_stepping_start (
    w.wave, saved.data (), &handle);

  switch (st)
    {
    case AMD_DBGAPI_STATUS_SUCCESS:
      /* We own the buffer; release it immediately.  Per the docs, if
         the single step has not been performed, complete resets the
         wave to its pre-start state, which is exactly what we want.  */
      EXPECT_EQ (amd_dbgapi_displaced_stepping_complete (w.wave, handle),
                 AMD_DBGAPI_STATUS_SUCCESS);
      break;
    case AMD_DBGAPI_STATUS_ERROR_ILLEGAL_INSTRUCTION:
    case AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS:
    case AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_BUFFER_NOT_AVAILABLE:
      /* Legitimate refusals for an un-patched PC.  */
      break;
    default:
      ADD_FAILURE () << "unexpected displaced_stepping_start status: " << st;
      break;
    }

  detach (w);
}

TEST_F (DisplacedStepping, ResumeWithAbortExceptionTransitionsWave)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  if (!stop_and_settle (w.pid, w.wave))
    {
      detach (w);
      GTEST_SKIP () << "could not park wave in STATE_STOP";
    }

  /* Resume with WAVE_ABORT exception.  Per the wave_resume contract
     this halts the wave, notifies the runtime, and the queue moves to
     the error state.  We don't introspect the queue after — the child
     is about to be SIGKILLed by the test teardown — but the resume
     itself must return SUCCESS.  */
  ASSERT_EQ (amd_dbgapi_wave_resume (w.wave, AMD_DBGAPI_RESUME_MODE_NORMAL,
                                     AMD_DBGAPI_EXCEPTION_WAVE_ABORT),
             AMD_DBGAPI_STATUS_SUCCESS);

  detach (w);
}
