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

/* Unit tests for breakpoint_t (callbacks.cpp).

   The ctor delegates insertion to process_t::insert_breakpoint, which in
   turn calls the global insert_breakpoint client callback installed by
   amd_dbgapi_initialize.  Tests install per-test stub callbacks to drive
   either the success or failure branch:

     * Success branch: ctor records m_inserted=true; is_inserted()
       returns true; ~breakpoint_t calls remove_breakpoint.
     * Failure branch: ctor records m_inserted=false; is_inserted()
       returns false; ~breakpoint_t skips remove_breakpoint.
     * get_info BREAKPOINT_INFO_PROCESS returns the bound process id.
     * get_info with an unknown enum throws INVALID_ARGUMENT.  */

#include "amd-dbgapi.h"
#include "callbacks.h"
#include "exception.h"
#include "os_driver.h"
#include "process.h"

#include "support/mock_os_driver.h"
#include "support/process_test_access.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

using ::testing::NiceMock;
using ::testing::Return;

using amd::dbgapi::api_error_t;
using amd::dbgapi::breakpoint_t;
using amd::dbgapi::host_address_t;
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

/* Per-test stub state.  The library invokes insert_breakpoint /
   remove_breakpoint through global function pointers installed by
   amd_dbgapi_initialize; these globals capture the last call so tests
   can inspect what the library asked for and decide what to return.  */
struct stub_state_t
{
  int insert_calls{ 0 };
  int remove_calls{ 0 };
  amd_dbgapi_global_address_t last_insert_address{ 0 };
  amd_dbgapi_breakpoint_id_t last_insert_id{ AMD_DBGAPI_BREAKPOINT_NONE };
  amd_dbgapi_breakpoint_id_t last_remove_id{ AMD_DBGAPI_BREAKPOINT_NONE };
  amd_dbgapi_status_t insert_result{ AMD_DBGAPI_STATUS_SUCCESS };
};

stub_state_t *g_stub = nullptr;

amd_dbgapi_status_t
stub_insert_breakpoint (amd_dbgapi_client_process_id_t,
                        amd_dbgapi_global_address_t address,
                        amd_dbgapi_breakpoint_id_t id)
{
  if (g_stub != nullptr)
    {
      ++g_stub->insert_calls;
      g_stub->last_insert_address = address;
      g_stub->last_insert_id = id;
      return g_stub->insert_result;
    }
  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
stub_remove_breakpoint (amd_dbgapi_client_process_id_t,
                        amd_dbgapi_breakpoint_id_t id)
{
  if (g_stub != nullptr)
    {
      ++g_stub->remove_calls;
      g_stub->last_remove_id = id;
    }
  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
stub_client_process_get_info (amd_dbgapi_client_process_id_t,
                              amd_dbgapi_client_process_info_t, size_t, void *)
{
  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
stub_xfer_global_memory (amd_dbgapi_client_process_id_t,
                         amd_dbgapi_global_address_t, amd_dbgapi_size_t *,
                         void *, const void *)
{
  return AMD_DBGAPI_STATUS_SUCCESS;
}

void *
stub_allocate_memory (size_t size)
{
  return std::malloc (size);
}

void
stub_deallocate_memory (void *p)
{
  std::free (p);
}

void
stub_log_message (amd_dbgapi_log_level_t, const char *)
{
}

class BreakpointFixture : public ::testing::Test
{
protected:
  void SetUp () override
  {
    g_stub = &stub_state;

    amd_dbgapi_callbacks_s cb{};
    cb.allocate_memory = stub_allocate_memory;
    cb.deallocate_memory = stub_deallocate_memory;
    cb.client_process_get_info = stub_client_process_get_info;
    cb.insert_breakpoint = stub_insert_breakpoint;
    cb.remove_breakpoint = stub_remove_breakpoint;
    cb.xfer_global_memory = stub_xfer_global_memory;
    cb.log_message = stub_log_message;
    ASSERT_EQ (amd_dbgapi_initialize (&cb), AMD_DBGAPI_STATUS_SUCCESS);

    auto driver = std::make_unique<NiceMock<MockOsDriver>> ();
    ON_CALL (*driver, is_valid ()).WillByDefault (Return (true));
    process = process_test_access::make (
      amd_dbgapi_process_id_t{ 1 }, fake_client_id (), std::nullopt,
      std::unique_ptr<os_driver_t> (driver.release ()));
  }

  void TearDown () override
  {
    process.reset ();
    amd_dbgapi_finalize ();
    g_stub = nullptr;
  }

  static void noop_action (breakpoint_t &, amd_dbgapi_client_thread_id_t,
                           amd_dbgapi_breakpoint_action_t *)
  {
  }

  stub_state_t stub_state{};
  std::unique_ptr<process_t> process;
};

} /* namespace */

TEST_F (BreakpointFixture, CtorOnSuccessRecordsInsertedAndPassesAddressAndId)
{
  auto &bp = process->create<breakpoint_t> (
    *process, host_address_t{ 0x4000ull }, &BreakpointFixture::noop_action);

  EXPECT_TRUE (bp.is_inserted ());
  EXPECT_EQ (stub_state.insert_calls, 1);
  EXPECT_EQ (stub_state.last_insert_address, 0x4000u);
  EXPECT_EQ (stub_state.last_insert_id.handle, bp.id ().handle);

  /* Destroying the breakpoint runs ~breakpoint_t which must call
     remove_breakpoint with the id used at insert time.  */
  amd_dbgapi_breakpoint_id_t bp_id = bp.id ();
  process->destroy (&bp);
  EXPECT_EQ (stub_state.remove_calls, 1);
  EXPECT_EQ (stub_state.last_remove_id.handle, bp_id.handle);
}

TEST_F (BreakpointFixture, CtorOnFailureLeavesInsertedFalseAndDtorSkipsRemove)
{
  stub_state.insert_result = AMD_DBGAPI_STATUS_ERROR;

  auto &bp = process->create<breakpoint_t> (
    *process, host_address_t{ 0x8000ull }, &BreakpointFixture::noop_action);

  EXPECT_FALSE (bp.is_inserted ());
  EXPECT_EQ (stub_state.insert_calls, 1);

  process->destroy (&bp);
  /* When insert failed, dtor must NOT call remove_breakpoint.  */
  EXPECT_EQ (stub_state.remove_calls, 0);
}

TEST_F (BreakpointFixture, GetInfoProcessReturnsBoundProcessId)
{
  auto &bp = process->create<breakpoint_t> (
    *process, host_address_t{ 0x1000ull }, &BreakpointFixture::noop_action);

  amd_dbgapi_process_id_t pid{};
  bp.get_info (AMD_DBGAPI_BREAKPOINT_INFO_PROCESS, sizeof (pid), &pid);
  EXPECT_EQ (pid.handle, process->id ().handle);
}

TEST_F (BreakpointFixture, GetInfoUnknownQueryThrowsInvalidArgument)
{
  auto &bp = process->create<breakpoint_t> (
    *process, host_address_t{ 0x1000ull }, &BreakpointFixture::noop_action);

  uint64_t dummy{};
  try
    {
      bp.get_info (static_cast<amd_dbgapi_breakpoint_info_t> (0xbeef),
                   sizeof (dummy), &dummy);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}
