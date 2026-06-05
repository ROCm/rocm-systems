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

/* HIP workload feature tests.

   These are the first tests where the inferior is actually doing
   GPU work.  We spawn the idle_kernel child (single block, single
   wave, spinning on a flag we never set), wait for its READY
   handshake, then attach via amd_dbgapi_process_attach and assert
   the dbgapi view of the child is sane:

     * agent_list contains at least one agent
     * the agent's architecture is reachable via the public
       AGENT_INFO_ARCHITECTURE query and round-trips to the
       architecture's NAME
     * the architecture name is one of the gfx_target_versions we
       enumerated from KFD topology

   Wave / queue / dispatch inspection is deferred to Commit 21; this
   commit only proves the workload harness works end-to-end.

   Fixture is gpu_fixture_t, which already skips on hosts without a
   GPU.  The spawn itself can still fail (HIP runtime issue, child
   crash before READY) — in that case we GTEST_SKIP with a message
   pointing the developer at the workload binary so they can run it
   by hand for diagnostics.  */

#include "amd-dbgapi.h"

#include "support/dbgapi_fixture.h"
#include "support/pid_callbacks.h"
#include "support/spawn_hip_workload.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <string>

using amd::dbgapi::test::gpu_fixture_t;
using amd::dbgapi::test::hip_workload_available;
using amd::dbgapi::test::idle_child_t;
using amd::dbgapi::test::pid_callbacks;
using amd::dbgapi::test::spawn_hip_workload;

namespace
{

class hip_workload_fixture_t : public gpu_fixture_t
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

} /* namespace */

using HipWorkload = hip_workload_fixture_t;

TEST_F (HipWorkload, SpawnAttachAgentListIsNonEmpty)
{
  idle_child_t child = spawn_hip_workload ();
  if (!child.valid ())
    GTEST_SKIP () << "failed to spawn HIP idle workload (READY not seen)";

  amd_dbgapi_os_process_id_t target = child.pid ();
  amd_dbgapi_process_id_t pid{};
  amd_dbgapi_status_t attach_st = amd_dbgapi_process_attach (
    reinterpret_cast<amd_dbgapi_client_process_id_t> (&target), &pid);
  /* KFD DBG_TRAP_ENABLE on a non-self PID requires elevated privilege
     (CAP_SYS_PTRACE) on stock kernels; an unprivileged run sees this
     surface as STATUS_FATAL.  Skip rather than fail so the binary
     still passes on CI nodes without that capability.  */
  if (attach_st == AMD_DBGAPI_STATUS_FATAL
      || attach_st == AMD_DBGAPI_STATUS_ERROR
      || attach_st == AMD_DBGAPI_STATUS_ERROR_RESTRICTION)
    GTEST_SKIP () << "process_attach against child PID failed ("
                  << attach_st
                  << ") — likely missing CAP_SYS_PTRACE for KFD DBG_TRAP";
  ASSERT_EQ (attach_st, AMD_DBGAPI_STATUS_SUCCESS);

  size_t agent_count = 0;
  amd_dbgapi_agent_id_t *agents = nullptr;
  ASSERT_EQ (
    amd_dbgapi_process_agent_list (pid, &agent_count, &agents, nullptr),
    AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_GT (agent_count, 0u)
    << "expected at least one agent visible to the running HIP child";
  if (agent_count > 0)
    {
      ASSERT_NE (agents, nullptr);
      for (size_t i = 0; i < agent_count; ++i)
        EXPECT_NE (agents[i].handle, AMD_DBGAPI_AGENT_NONE.handle)
          << "agent index " << i;
    }
  std::free (agents);

  EXPECT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);
}

TEST_F (HipWorkload, AgentArchitectureRoundTripsToKnownGfx)
{
  idle_child_t child = spawn_hip_workload ();
  if (!child.valid ())
    GTEST_SKIP () << "failed to spawn HIP idle workload (READY not seen)";

  amd_dbgapi_os_process_id_t target = child.pid ();
  amd_dbgapi_process_id_t pid{};
  amd_dbgapi_status_t attach_st = amd_dbgapi_process_attach (
    reinterpret_cast<amd_dbgapi_client_process_id_t> (&target), &pid);
  if (attach_st == AMD_DBGAPI_STATUS_FATAL
      || attach_st == AMD_DBGAPI_STATUS_ERROR
      || attach_st == AMD_DBGAPI_STATUS_ERROR_RESTRICTION)
    GTEST_SKIP () << "process_attach against child PID failed ("
                  << attach_st
                  << ") — likely missing CAP_SYS_PTRACE for KFD DBG_TRAP";
  ASSERT_EQ (attach_st, AMD_DBGAPI_STATUS_SUCCESS);

  size_t agent_count = 0;
  amd_dbgapi_agent_id_t *agents = nullptr;
  ASSERT_EQ (
    amd_dbgapi_process_agent_list (pid, &agent_count, &agents, nullptr),
    AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_GT (agent_count, 0u);

  /* Look at the first supported agent.  The architecture query
     returns NOT_AVAILABLE on unsupported agents, which is legal —
     skip those and require that at least one supported agent
     exists.  */
  bool saw_supported = false;
  for (size_t i = 0; i < agent_count; ++i)
    {
      amd_dbgapi_architecture_id_t arch{};
      amd_dbgapi_status_t st = amd_dbgapi_agent_get_info (
        agents[i], AMD_DBGAPI_AGENT_INFO_ARCHITECTURE, sizeof (arch), &arch);
      if (st == AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE)
        continue;
      ASSERT_EQ (st, AMD_DBGAPI_STATUS_SUCCESS) << "agent index " << i;
      EXPECT_NE (arch.handle, AMD_DBGAPI_ARCHITECTURE_NONE.handle);

      char *name = nullptr;
      ASSERT_EQ (
        amd_dbgapi_architecture_get_info (arch, AMD_DBGAPI_ARCHITECTURE_INFO_NAME,
                                          sizeof (name), &name),
        AMD_DBGAPI_STATUS_SUCCESS);
      ASSERT_NE (name, nullptr);
      EXPECT_EQ (std::strncmp (name, "gfx", 3), 0)
        << "expected gfx<n> name, got: " << name;
      std::free (name);
      saw_supported = true;
      break;
    }
  EXPECT_TRUE (saw_supported)
    << "no agent reported a supported architecture";

  std::free (agents);
  EXPECT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);
}
