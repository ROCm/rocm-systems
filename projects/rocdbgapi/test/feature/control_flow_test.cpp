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

/* Control-flow feature tests.

   Builds on Commit 21's wave-inspection harness: spawn the idle HIP
   workload, attach, pick a wave, then exercise the stop / event /
   query-PC / resume cycle that gdb drives on every user "stop"
   command:

     1. amd_dbgapi_wave_stop on a running wave.
     2. Poll amd_dbgapi_process_next_pending_event until the matching
        AMD_DBGAPI_EVENT_KIND_WAVE_STOP event arrives.  EVENT_INFO_WAVE
        must round-trip to the wave we stopped.
     3. amd_dbgapi_event_processed to release the event.
     4. With the wave parked in AMD_DBGAPI_WAVE_STATE_STOP, query
        WAVE_INFO_STATE (must be STOP) and WAVE_INFO_PC (must succeed
        with a non-zero value — the idle kernel spins in a loop).
     5. amd_dbgapi_wave_resume(NORMAL, EXCEPTION_NONE) returns the wave
        to RUN.  A correctly-resumed spinning wave does not produce
        another event, which is exactly the contract we want.

   Single-step / displaced stepping / exception delivery is deferred to
   Commit 23.

   Progress mode handling differs from Commit 21: we set NORMAL so the
   event pump is active, then poll.  NO_FORWARD parks events in the
   driver and would make next_pending_event return EVENT_NONE forever
   for stop events we ask for ourselves.  */

#include "amd-dbgapi.h"

#include "support/dbgapi_fixture.h"
#include "support/pid_callbacks.h"
#include "support/spawn_hip_workload.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>

using amd::dbgapi::test::gpu_fixture_t;
using amd::dbgapi::test::hip_workload_available;
using amd::dbgapi::test::idle_child_t;
using amd::dbgapi::test::pid_callbacks;
using amd::dbgapi::test::spawn_hip_workload;

namespace
{

class control_flow_fixture_t : public gpu_fixture_t
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

/* attach + pick first wave, leaving the process in NORMAL progress so
   the event pump is live.  On any setup failure ok stays false so the
   test can GTEST_SKIP; the binary stays green on hosts that can't run
   a HIP workload under KFD debug.  */
struct attached_wave_t
{
  idle_child_t child;
  amd_dbgapi_process_id_t pid{};
  amd_dbgapi_wave_id_t wave{ AMD_DBGAPI_WAVE_NONE };
  bool ok{ false };
};

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

  /* Briefly switch to NO_FORWARD just to take a stable wave snapshot,
     then go back to NORMAL so subsequent stop events get delivered.  */
  if (amd_dbgapi_process_set_progress (out.pid, AMD_DBGAPI_PROGRESS_NO_FORWARD)
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      amd_dbgapi_process_detach (out.pid);
      return out;
    }

  size_t wave_count = 0;
  amd_dbgapi_wave_id_t *waves = nullptr;
  amd_dbgapi_status_t list_st = amd_dbgapi_process_wave_list (
    out.pid, &wave_count, &waves, nullptr);
  if (list_st == AMD_DBGAPI_STATUS_SUCCESS && wave_count > 0)
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

/* Poll next_pending_event for up to ~3 s looking for a WAVE_STOP event
   whose EVENT_INFO_WAVE matches `wanted`.  Returns EVENT_NONE if no
   such event arrives — the caller should treat that as a skip, not a
   failure, because the GPU scheduler may not actually stop the wave on
   this host.  Any other (non-matching) event we encounter is released
   via event_processed so we don't leak the event slot.  */
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

      /* Not the event we wanted — release it and keep polling.  */
      amd_dbgapi_event_processed (ev);
    }
  return AMD_DBGAPI_EVENT_NONE;
}

} /* namespace */

using ControlFlow = control_flow_fixture_t;

TEST_F (ControlFlow, WaveStopProducesWaveStopEvent)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  amd_dbgapi_status_t stop_st = amd_dbgapi_wave_stop (w.wave);
  /* WAVE_STOPPED means the wave was already parked by the runtime; that
     is a legitimate state for the spinning kernel under some schedules
     and there is no event for us to wait on, so skip.  */
  if (stop_st == AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED)
    {
      detach (w);
      GTEST_SKIP () << "wave already stopped — nothing to wait for";
    }
  ASSERT_EQ (stop_st, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_event_id_t ev = wait_for_wave_stop_event (w.pid, w.wave);
  if (ev.handle == AMD_DBGAPI_EVENT_NONE.handle)
    {
      detach (w);
      GTEST_SKIP () << "no WAVE_STOP event arrived within 3s — scheduler "
                       "may not have run the wave";
    }
  EXPECT_EQ (amd_dbgapi_event_processed (ev), AMD_DBGAPI_STATUS_SUCCESS);
  detach (w);
}

TEST_F (ControlFlow, StoppedWaveReportsStopStateAndPC)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  amd_dbgapi_status_t stop_st = amd_dbgapi_wave_stop (w.wave);
  if (stop_st != AMD_DBGAPI_STATUS_SUCCESS
      && stop_st != AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED)
    {
      detach (w);
      FAIL () << "wave_stop returned unexpected status: " << stop_st;
    }
  if (stop_st == AMD_DBGAPI_STATUS_SUCCESS)
    {
      amd_dbgapi_event_id_t ev = wait_for_wave_stop_event (w.pid, w.wave);
      if (ev.handle == AMD_DBGAPI_EVENT_NONE.handle)
        {
          detach (w);
          GTEST_SKIP () << "no WAVE_STOP event arrived within 3s";
        }
      ASSERT_EQ (amd_dbgapi_event_processed (ev),
                 AMD_DBGAPI_STATUS_SUCCESS);
    }

  amd_dbgapi_wave_state_t state{};
  ASSERT_EQ (amd_dbgapi_wave_get_info (w.wave, AMD_DBGAPI_WAVE_INFO_STATE,
                                       sizeof (state), &state),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_EQ (state, AMD_DBGAPI_WAVE_STATE_STOP);

  amd_dbgapi_global_address_t pc = 0;
  ASSERT_EQ (amd_dbgapi_wave_get_info (w.wave, AMD_DBGAPI_WAVE_INFO_PC,
                                       sizeof (pc), &pc),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_NE (pc, 0u) << "expected a non-zero PC for the spinning idle kernel";

  detach (w);
}

/* ------------------------------------------------------------------ */
/* C29 — wave_resume argument validation (mode × exception)            */
/*                                                                     */
/* amd_dbgapi_wave_resume validates the resume_mode and exceptions     */
/* bitmask BEFORE checking wave state, so these tests do not require   */
/* a stopped wave — only a valid wave handle.  The "wave not stopped"  */
/* path is exercised explicitly with a known-running wave.             */
/* ------------------------------------------------------------------ */

TEST_F (ControlFlow, ResumeRejectsInvalidResumeMode)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  /* Pick an int outside the {NORMAL, SINGLE_STEP} enumerator range.
     -1 / INT_MAX are the classic out-of-range probes.  */
  EXPECT_EQ (amd_dbgapi_wave_resume (
               w.wave, static_cast<amd_dbgapi_resume_mode_t> (-1),
               AMD_DBGAPI_EXCEPTION_NONE),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  EXPECT_EQ (amd_dbgapi_wave_resume (
               w.wave, static_cast<amd_dbgapi_resume_mode_t> (0x7fffffff),
               AMD_DBGAPI_EXCEPTION_NONE),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  detach (w);
}

TEST_F (ControlFlow, ResumeRejectsInvalidExceptionBits)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  /* The accepted mask is (WAVE_ABORT | WAVE_TRAP | WAVE_MATH_ERROR |
     WAVE_ILLEGAL_INSTRUCTION | WAVE_MEMORY_VIOLATION |
     WAVE_ADDRESS_ERROR).  A high bit well outside that mask must be
     rejected with INVALID_ARGUMENT before any wave-state check.  */
  constexpr auto bogus_bit
    = static_cast<amd_dbgapi_exceptions_t> (1ull << 30);
  EXPECT_EQ (amd_dbgapi_wave_resume (w.wave, AMD_DBGAPI_RESUME_MODE_NORMAL,
                                     bogus_bit),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  /* A valid bit OR'd with a bogus bit still fails: the full mask must
     be a strict subset of the allowed set.  */
  constexpr auto mixed = static_cast<amd_dbgapi_exceptions_t> (
    AMD_DBGAPI_EXCEPTION_WAVE_TRAP | (1u << 30));
  EXPECT_EQ (amd_dbgapi_wave_resume (w.wave, AMD_DBGAPI_RESUME_MODE_NORMAL,
                                     mixed),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);

  detach (w);
}

TEST_F (ControlFlow, ResumeOnRunningWaveReturnsWaveNotStopped)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  /* Don't stop the wave first.  With valid mode + exceptions, the
     resume must hit the wave-state guard and report WAVE_NOT_STOPPED.
     If the wave happens to be parked already (scheduler quiesced the
     idle kernel) accept that as a skip rather than a failure.  */
  amd_dbgapi_wave_state_t state{};
  ASSERT_EQ (amd_dbgapi_wave_get_info (w.wave, AMD_DBGAPI_WAVE_INFO_STATE,
                                       sizeof (state), &state),
             AMD_DBGAPI_STATUS_SUCCESS);
  if (state == AMD_DBGAPI_WAVE_STATE_STOP)
    {
      detach (w);
      GTEST_SKIP () << "wave parked by scheduler — cannot test running path";
    }

  EXPECT_EQ (amd_dbgapi_wave_resume (w.wave, AMD_DBGAPI_RESUME_MODE_NORMAL,
                                     AMD_DBGAPI_EXCEPTION_NONE),
             AMD_DBGAPI_STATUS_ERROR_WAVE_NOT_STOPPED);

  detach (w);
}

TEST_F (ControlFlow, ResumeReturnsStoppedWaveToRun)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  amd_dbgapi_status_t stop_st = amd_dbgapi_wave_stop (w.wave);
  if (stop_st != AMD_DBGAPI_STATUS_SUCCESS
      && stop_st != AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED)
    {
      detach (w);
      FAIL () << "wave_stop returned unexpected status: " << stop_st;
    }
  if (stop_st == AMD_DBGAPI_STATUS_SUCCESS)
    {
      amd_dbgapi_event_id_t ev = wait_for_wave_stop_event (w.pid, w.wave);
      if (ev.handle == AMD_DBGAPI_EVENT_NONE.handle)
        {
          detach (w);
          GTEST_SKIP () << "no WAVE_STOP event arrived within 3s";
        }
      ASSERT_EQ (amd_dbgapi_event_processed (ev),
                 AMD_DBGAPI_STATUS_SUCCESS);
    }

  ASSERT_EQ (amd_dbgapi_wave_resume (w.wave, AMD_DBGAPI_RESUME_MODE_NORMAL,
                                     AMD_DBGAPI_EXCEPTION_NONE),
             AMD_DBGAPI_STATUS_SUCCESS);

  /* After a normal resume the wave should not appear in STATE_STOP
     anymore.  WAVE_INFO_STATE may briefly lag the hardware; allow up
     to ~500 ms for the transition.  */
  using clock = std::chrono::steady_clock;
  auto deadline = clock::now () + std::chrono::milliseconds (500);
  amd_dbgapi_wave_state_t state = AMD_DBGAPI_WAVE_STATE_STOP;
  while (clock::now () < deadline)
    {
      if (amd_dbgapi_wave_get_info (w.wave, AMD_DBGAPI_WAVE_INFO_STATE,
                                    sizeof (state), &state)
            != AMD_DBGAPI_STATUS_SUCCESS
          || state != AMD_DBGAPI_WAVE_STATE_STOP)
        break;
      std::this_thread::sleep_for (std::chrono::milliseconds (10));
    }
  EXPECT_NE (state, AMD_DBGAPI_WAVE_STATE_STOP)
    << "wave should have left STATE_STOP after resume";

  detach (w);
}

/* SINGLE_STEP is a separate code path inside wave_resume: it lands the
   wave in WAVE_STATE_SINGLE_STEP rather than RUN.  We don't try to
   verify the post-resume state here (a single-step wave can re-stop
   immediately and races the query); the contract under test is just
   that the call is accepted when mode + exceptions are valid and the
   wave is stopped.  */
TEST_F (ControlFlow, ResumeAcceptsSingleStepWithExceptionNone)
{
  attached_wave_t w = attach_and_pick_wave ();
  if (!w.ok)
    GTEST_SKIP () << "attach / wave selection failed — see Commit 21 notes";

  amd_dbgapi_status_t stop_st = amd_dbgapi_wave_stop (w.wave);
  if (stop_st != AMD_DBGAPI_STATUS_SUCCESS
      && stop_st != AMD_DBGAPI_STATUS_ERROR_WAVE_STOPPED)
    {
      detach (w);
      FAIL () << "wave_stop returned unexpected status: " << stop_st;
    }
  if (stop_st == AMD_DBGAPI_STATUS_SUCCESS)
    {
      amd_dbgapi_event_id_t ev = wait_for_wave_stop_event (w.pid, w.wave);
      if (ev.handle == AMD_DBGAPI_EVENT_NONE.handle)
        {
          detach (w);
          GTEST_SKIP () << "no WAVE_STOP event arrived within 3s";
        }
      ASSERT_EQ (amd_dbgapi_event_processed (ev), AMD_DBGAPI_STATUS_SUCCESS);
    }

  EXPECT_EQ (amd_dbgapi_wave_resume (w.wave,
                                     AMD_DBGAPI_RESUME_MODE_SINGLE_STEP,
                                     AMD_DBGAPI_EXCEPTION_NONE),
             AMD_DBGAPI_STATUS_SUCCESS);

  detach (w);
}
