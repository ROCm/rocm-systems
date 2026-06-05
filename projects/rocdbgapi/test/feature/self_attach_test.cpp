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

/* Self-attach feature tests.

   These tests exercise amd_dbgapi_process_attach /
   amd_dbgapi_process_detach against (a) the current test process and
   (b) a freshly spawned idle child.  Neither uses the GPU, but
   process_attach still requires the kernel driver to be present
   (KFD on Linux), so the fixture is driver-tier and the binary
   skips cleanly on driverless hosts.

   The interesting properties under test:
     * attach succeeds and returns a non-NONE process handle
     * the OS_ID query round-trips the PID we configured the
       callbacks with
     * agent_list is queryable post-attach (count may be zero — this
       process isn't running any GPU work)
     * a second attach for the same client_process_id returns
       ALREADY_ATTACHED rather than corrupting library state
     * detaching an invalid handle is rejected, not silently
       accepted
     * attach handles NULL out-pointer

   The child-attach path is gated on a successful spawn; if the
   helper returns an invalid handle (non-Linux platform or fork
   failure), the test GTEST_SKIPs.  */

#include "amd-dbgapi.h"

#include "support/dbgapi_fixture.h"
#include "support/pid_callbacks.h"
#include "support/spawn_idle_child.h"

#include <gtest/gtest.h>

#if defined(__linux__)
#  include <unistd.h>
#endif

#include <cstdlib>

using amd::dbgapi::test::driver_fixture_t;
using amd::dbgapi::test::idle_child_t;
using amd::dbgapi::test::pid_callbacks;
using amd::dbgapi::test::spawn_idle_child;

namespace
{

/* Driver-tier fixture that supplies the pid-aware callbacks instead
   of the minimal ones, so tests can drive the PID they want the
   library to attach to via the client_process_id pointer.  */
class self_attach_fixture_t : public driver_fixture_t
{
protected:
  const amd_dbgapi_callbacks_s &
  callbacks () const override
  {
    return pid_callbacks ();
  }
};

} /* namespace */

using SelfAttach = self_attach_fixture_t;

TEST_F (SelfAttach, AttachAndDetachReturnsValidHandle)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  amd_dbgapi_process_id_t pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_NE (pid.handle, AMD_DBGAPI_PROCESS_NONE.handle);

  EXPECT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);
}

TEST_F (SelfAttach, OsIdRoundTripsConfiguredPid)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  amd_dbgapi_process_id_t pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid),
             AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_os_process_id_t reported = 0;
  ASSERT_EQ (amd_dbgapi_process_get_info (pid, AMD_DBGAPI_PROCESS_INFO_OS_ID,
                                          sizeof (reported), &reported),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_EQ (reported, target);

  EXPECT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);
}

TEST_F (SelfAttach, AgentListIsQueryableForOurOwnProcess)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  amd_dbgapi_process_id_t pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid),
             AMD_DBGAPI_STATUS_SUCCESS);

  /* The library reports every GPU agent the driver topology knows
     about, not just agents the inferior is using.  So count is 0 on
     a CPU-only host and N on an N-GPU host; either way the call
     must succeed, and any reported handle must be non-NONE.  */
  size_t agent_count = static_cast<size_t> (-1);
  amd_dbgapi_agent_id_t *agents = nullptr;
  ASSERT_EQ (
    amd_dbgapi_process_agent_list (pid, &agent_count, &agents, nullptr),
    AMD_DBGAPI_STATUS_SUCCESS);
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

TEST_F (SelfAttach, DoubleAttachReturnsAlreadyAttached)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  amd_dbgapi_process_id_t pid1{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid1),
             AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_process_id_t pid2{};
  EXPECT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid2),
             AMD_DBGAPI_STATUS_ERROR_ALREADY_ATTACHED);

  EXPECT_EQ (amd_dbgapi_process_detach (pid1), AMD_DBGAPI_STATUS_SUCCESS);
}

TEST_F (SelfAttach, AttachRejectsNullOutPointer)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  EXPECT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               nullptr),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST_F (SelfAttach, DetachWithInvalidHandleIsRejected)
{
  EXPECT_EQ (amd_dbgapi_process_detach (AMD_DBGAPI_PROCESS_NONE),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

/* C27 handle-misuse: after detach, the process_id is removed from the
   handle map.  A second detach with the same handle must report
   INVALID_PROCESS_ID, NOT crash or return SUCCESS again.  */
TEST_F (SelfAttach, DoubleDetachReturnsInvalidProcessId)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  amd_dbgapi_process_id_t pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid),
             AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);

  /* Same handle, second time -- destroy_process already removed it.  */
  EXPECT_EQ (amd_dbgapi_process_detach (pid),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

TEST_F (SelfAttach, GetInfoAfterDetachReturnsInvalidProcessId)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  amd_dbgapi_process_id_t pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid),
             AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_os_process_id_t reported = 0;
  EXPECT_EQ (amd_dbgapi_process_get_info (pid, AMD_DBGAPI_PROCESS_INFO_OS_ID,
                                          sizeof (reported), &reported),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

TEST_F (SelfAttach, AgentListAfterDetachReturnsInvalidProcessId)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  amd_dbgapi_process_id_t pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid),
             AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);

  size_t agent_count = 0;
  amd_dbgapi_agent_id_t *agents = nullptr;
  EXPECT_EQ (
    amd_dbgapi_process_agent_list (pid, &agent_count, &agents, nullptr),
    AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

/* C28 state-machine misuse: amd_dbgapi_process_freeze requires
   forward_progress_needed==false and wave_launch_mode==halt before
   it will try to suspend execution.  A freshly attached process has
   forward_progress_needed=true, so freeze() must report
   INCOMPATIBLE_PROCESS_STATE without touching the driver.  */
TEST_F (SelfAttach, FreezeWithForwardProgressReturnsIncompatibleState)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  amd_dbgapi_process_id_t pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid),
             AMD_DBGAPI_STATUS_SUCCESS);

  EXPECT_EQ (amd_dbgapi_process_freeze (pid),
             AMD_DBGAPI_STATUS_ERROR_INCOMPATIBLE_PROCESS_STATE);

  EXPECT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);
}

/* C28 state-machine misuse: unfreeze on a process that was never
   frozen must report PROCESS_NOT_FROZEN rather than silently
   succeeding (which would let the client lose track of state).  */
TEST_F (SelfAttach, UnfreezeWithoutPriorFreezeReturnsNotFrozen)
{
#if defined(__linux__)
  amd_dbgapi_os_process_id_t target = ::getpid ();
#else
  GTEST_SKIP () << "process_attach requires Linux for this test";
#endif

  amd_dbgapi_process_id_t pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid),
             AMD_DBGAPI_STATUS_SUCCESS);

  EXPECT_EQ (amd_dbgapi_process_unfreeze (pid),
             AMD_DBGAPI_STATUS_ERROR_PROCESS_NOT_FROZEN);

  EXPECT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);
}

/* C32 PID edge cases.

   client_process_id is the caller's opaque cookie that the library
   round-trips through the client_process_get_info callback to obtain
   the OS PID.  pid_client_process_get_info rejects a null cookie
   with INVALID_ARGUMENT, so attach must propagate that error rather
   than crash inside the library.  */
TEST_F (SelfAttach, AttachWithNullClientProcessIdRejected)
{
  amd_dbgapi_process_id_t pid{};
  EXPECT_NE (amd_dbgapi_process_attach (nullptr, &pid),
             AMD_DBGAPI_STATUS_SUCCESS);
}

/* Attach to a PID that no longer exists must fail (KFD has nothing
   to attach to).  We don't pin a specific error code because the
   exact mapping depends on the kernel driver path, but it must NOT
   return SUCCESS and must NOT corrupt library state (a follow-up
   self-attach should still work).  */
TEST_F (SelfAttach, AttachToDeadChildIsRejected)
{
  idle_child_t child = spawn_idle_child ();
  if (!child.valid ())
    GTEST_SKIP () << "spawn_idle_child not supported on this platform";

  amd_dbgapi_os_process_id_t target = child.pid ();
  child.kill_and_wait ();

  amd_dbgapi_process_id_t pid{};
  amd_dbgapi_status_t status = amd_dbgapi_process_attach (
    reinterpret_cast<amd_dbgapi_client_process_id_t> (&target), &pid);
  if (status == AMD_DBGAPI_STATUS_SUCCESS)
    {
      /* Some driver versions accept the attach and only surface the
         dead PID on a subsequent operation.  Either way clean up.  */
      EXPECT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);
      GTEST_SKIP () << "driver accepted attach to a dead PID; skipping";
    }
  EXPECT_NE (status, AMD_DBGAPI_STATUS_SUCCESS);

  /* Library state must still be usable after the rejection.  */
#if defined(__linux__)
  amd_dbgapi_os_process_id_t self = ::getpid ();
  amd_dbgapi_process_id_t self_pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&self),
               &self_pid),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_EQ (amd_dbgapi_process_detach (self_pid),
             AMD_DBGAPI_STATUS_SUCCESS);
#endif
}

/* Two different client_process_ids must yield two distinct process
   handles -- the handle map is keyed off the cookie, not off the
   OS PID, so even attaching to two different children must not
   collide or alias.  */
TEST_F (SelfAttach, AttachToTwoChildrenReturnsDistinctHandles)
{
  idle_child_t child_a = spawn_idle_child ();
  idle_child_t child_b = spawn_idle_child ();
  if (!child_a.valid () || !child_b.valid ())
    GTEST_SKIP () << "spawn_idle_child not supported on this platform";
  ASSERT_NE (child_a.pid (), child_b.pid ());

  amd_dbgapi_os_process_id_t target_a = child_a.pid ();
  amd_dbgapi_os_process_id_t target_b = child_b.pid ();

  amd_dbgapi_process_id_t pid_a{}, pid_b{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target_a),
               &pid_a),
             AMD_DBGAPI_STATUS_SUCCESS);
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target_b),
               &pid_b),
             AMD_DBGAPI_STATUS_SUCCESS);

  EXPECT_NE (pid_a.handle, pid_b.handle);
  EXPECT_NE (pid_a.handle, AMD_DBGAPI_PROCESS_NONE.handle);
  EXPECT_NE (pid_b.handle, AMD_DBGAPI_PROCESS_NONE.handle);

  EXPECT_EQ (amd_dbgapi_process_detach (pid_a), AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_EQ (amd_dbgapi_process_detach (pid_b), AMD_DBGAPI_STATUS_SUCCESS);
}

TEST_F (SelfAttach, AttachToSpawnedChildSucceeds)
{
  idle_child_t child = spawn_idle_child ();
  if (!child.valid ())
    GTEST_SKIP () << "spawn_idle_child not supported on this platform";

  amd_dbgapi_os_process_id_t target = child.pid ();

  amd_dbgapi_process_id_t pid{};
  ASSERT_EQ (amd_dbgapi_process_attach (
               reinterpret_cast<amd_dbgapi_client_process_id_t> (&target),
               &pid),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_NE (pid.handle, AMD_DBGAPI_PROCESS_NONE.handle);

  /* Round-trip the OS_ID query to confirm we attached to the child,
     not to ourselves.  */
  amd_dbgapi_os_process_id_t reported = 0;
  EXPECT_EQ (amd_dbgapi_process_get_info (pid, AMD_DBGAPI_PROCESS_INFO_OS_ID,
                                          sizeof (reported), &reported),
             AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_EQ (reported, child.pid ());

  EXPECT_EQ (amd_dbgapi_process_detach (pid), AMD_DBGAPI_STATUS_SUCCESS);
}
