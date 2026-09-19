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

/* Wave inspection feature tests.

   Builds on the Commit 20 HIP idle workload: spawn the idle_kernel
   child (single block, single wave, spinning), attach, switch to
   NO_FORWARD progress so wave snapshots are stable, then walk
   amd_dbgapi_process_wave_list and exercise the read-only
   AMD_DBGAPI_WAVE_INFO_* queries that do not require the wave to
   be stopped (AGENT, PROCESS, ARCHITECTURE, QUEUE, LANE_COUNT).

   wave_stop / PC / register inspection lands in Commit 22; this
   commit covers the "list-and-classify" half of the wave surface.  */

#include "amd-dbgapi.h"

#include "support/dbgapi_fixture.h"
#include "support/pid_callbacks.h"
#include "support/spawn_hip_workload.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

using amd::dbgapi::test::gpu_fixture_t;
using amd::dbgapi::test::hip_workload_available;
using amd::dbgapi::test::idle_child_t;
using amd::dbgapi::test::pid_callbacks;
using amd::dbgapi::test::spawn_hip_workload;

namespace
{

class wave_inspection_fixture_t : public gpu_fixture_t
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

/* Spawn the idle workload, attach, and switch to NO_FORWARD so wave
   snapshots are stable for the test body.  On any setup failure we
   GTEST_SKIP rather than fail so the binary stays green on hosts
   that lack the prerequisites (missing HIP, KFD attach denied,
   etc.).  Caller owns the returned process id; pass it back through
   teardown() before returning.  */
struct attached_workload_t
{
  idle_child_t child;
  amd_dbgapi_process_id_t pid{};
  bool ok{ false };
};

static attached_workload_t
attach_to_idle_workload ()
{
  attached_workload_t out;
  out.child = spawn_hip_workload ();
  if (!out.child.valid ())
    return out;

  amd_dbgapi_os_process_id_t target = out.child.pid ();
  amd_dbgapi_status_t st = amd_dbgapi_process_attach (
    reinterpret_cast<amd_dbgapi_client_process_id_t> (&target), &out.pid);
  if (st != AMD_DBGAPI_STATUS_SUCCESS)
    return out;

  if (amd_dbgapi_process_set_progress (out.pid, AMD_DBGAPI_PROGRESS_NO_FORWARD)
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      amd_dbgapi_process_detach (out.pid);
      return out;
    }

  out.ok = true;
  return out;
}

static void
detach_workload (attached_workload_t &w)
{
  if (w.ok)
    {
      amd_dbgapi_process_set_progress (w.pid, AMD_DBGAPI_PROGRESS_NORMAL);
      amd_dbgapi_process_detach (w.pid);
      w.ok = false;
    }
}

} /* namespace */

using WaveInspection = wave_inspection_fixture_t;

TEST_F (WaveInspection, WaveListIsNonEmpty)
{
  attached_workload_t w = attach_to_idle_workload ();
  if (!w.ok)
    GTEST_SKIP () << "attach/set_progress failed — see Commit 20 notes";

  size_t wave_count = 0;
  amd_dbgapi_wave_id_t *waves = nullptr;
  amd_dbgapi_changed_t changed{};
  ASSERT_EQ (amd_dbgapi_process_wave_list (w.pid, &wave_count, &waves,
                                           &changed),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_GT (wave_count, 0u)
    << "expected at least one wave from the spinning idle kernel";
  if (wave_count > 0)
    {
      ASSERT_NE (waves, nullptr);
      for (size_t i = 0; i < wave_count; ++i)
        EXPECT_NE (waves[i].handle, AMD_DBGAPI_WAVE_NONE.handle)
          << "wave index " << i;
    }
  std::free (waves);
  detach_workload (w);
}

TEST_F (WaveInspection, WaveArchitectureMatchesAgentArchitecture)
{
  attached_workload_t w = attach_to_idle_workload ();
  if (!w.ok)
    GTEST_SKIP () << "attach/set_progress failed — see Commit 20 notes";

  size_t wave_count = 0;
  amd_dbgapi_wave_id_t *waves = nullptr;
  ASSERT_EQ (amd_dbgapi_process_wave_list (w.pid, &wave_count, &waves,
                                           nullptr),
             AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_GT (wave_count, 0u);

  for (size_t i = 0; i < wave_count; ++i)
    {
      amd_dbgapi_architecture_id_t wave_arch{};
      ASSERT_EQ (amd_dbgapi_wave_get_info (waves[i],
                                           AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
                                           sizeof (wave_arch), &wave_arch),
                 AMD_DBGAPI_STATUS_SUCCESS)
        << "wave index " << i;
      EXPECT_NE (wave_arch.handle, AMD_DBGAPI_ARCHITECTURE_NONE.handle);

      amd_dbgapi_agent_id_t agent{};
      ASSERT_EQ (amd_dbgapi_wave_get_info (waves[i], AMD_DBGAPI_WAVE_INFO_AGENT,
                                           sizeof (agent), &agent),
                 AMD_DBGAPI_STATUS_SUCCESS)
        << "wave index " << i;
      EXPECT_NE (agent.handle, AMD_DBGAPI_AGENT_NONE.handle);

      amd_dbgapi_architecture_id_t agent_arch{};
      ASSERT_EQ (amd_dbgapi_agent_get_info (agent,
                                            AMD_DBGAPI_AGENT_INFO_ARCHITECTURE,
                                            sizeof (agent_arch), &agent_arch),
                 AMD_DBGAPI_STATUS_SUCCESS)
        << "wave index " << i;
      EXPECT_EQ (wave_arch.handle, agent_arch.handle)
        << "wave " << i << " architecture must match its owning agent";
    }

  std::free (waves);
  detach_workload (w);
}

TEST_F (WaveInspection, WaveQueueAndProcessHandlesAreValid)
{
  attached_workload_t w = attach_to_idle_workload ();
  if (!w.ok)
    GTEST_SKIP () << "attach/set_progress failed — see Commit 20 notes";

  size_t wave_count = 0;
  amd_dbgapi_wave_id_t *waves = nullptr;
  ASSERT_EQ (amd_dbgapi_process_wave_list (w.pid, &wave_count, &waves,
                                           nullptr),
             AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_GT (wave_count, 0u);

  for (size_t i = 0; i < wave_count; ++i)
    {
      amd_dbgapi_queue_id_t queue{};
      ASSERT_EQ (amd_dbgapi_wave_get_info (waves[i], AMD_DBGAPI_WAVE_INFO_QUEUE,
                                           sizeof (queue), &queue),
                 AMD_DBGAPI_STATUS_SUCCESS)
        << "wave index " << i;
      EXPECT_NE (queue.handle, AMD_DBGAPI_QUEUE_NONE.handle);

      amd_dbgapi_process_id_t proc{};
      ASSERT_EQ (amd_dbgapi_wave_get_info (waves[i],
                                           AMD_DBGAPI_WAVE_INFO_PROCESS,
                                           sizeof (proc), &proc),
                 AMD_DBGAPI_STATUS_SUCCESS)
        << "wave index " << i;
      EXPECT_EQ (proc.handle, w.pid.handle)
        << "wave " << i << " must report our attached process";
    }

  std::free (waves);
  detach_workload (w);
}

TEST_F (WaveInspection, WaveLaneCountIsSane)
{
  attached_workload_t w = attach_to_idle_workload ();
  if (!w.ok)
    GTEST_SKIP () << "attach/set_progress failed — see Commit 20 notes";

  size_t wave_count = 0;
  amd_dbgapi_wave_id_t *waves = nullptr;
  ASSERT_EQ (amd_dbgapi_process_wave_list (w.pid, &wave_count, &waves,
                                           nullptr),
             AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_GT (wave_count, 0u);

  for (size_t i = 0; i < wave_count; ++i)
    {
      size_t lane_count = 0;
      ASSERT_EQ (amd_dbgapi_wave_get_info (waves[i],
                                           AMD_DBGAPI_WAVE_INFO_LANE_COUNT,
                                           sizeof (lane_count), &lane_count),
                 AMD_DBGAPI_STATUS_SUCCESS)
        << "wave index " << i;
      EXPECT_TRUE (lane_count == 32 || lane_count == 64)
        << "wave " << i << " reported unexpected lane count: " << lane_count;
    }

  std::free (waves);
  detach_workload (w);
}
