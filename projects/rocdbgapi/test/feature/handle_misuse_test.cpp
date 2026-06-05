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

/* Public-API handle-misuse tests.

   Every "*_get_info" / "*_list" entry point in amd-dbgapi.h begins
   with `T *obj = T::find(id); if (obj == nullptr) THROW(INVALID_*_ID)`.
   This file exercises that path for a representative slice of object
   types: process, agent, queue, wave, dispatch, code_object,
   displaced_stepping, watchpoint, breakpoint, register, event,
   address_space, address_class, register_class, and architecture.

   No driver is required -- the library is initialized via the base
   fixture but no process is attached, so every lookup must miss and
   return the INVALID_*_ID status.  These tests catch regressions where
   a handle lookup is forgotten, miscategorized, or silently treated
   as a success.  Complements the unit tests by exercising the public
   C symbol exports and the TRY/CATCH plumbing.  */

#include "amd-dbgapi.h"

#include "support/dbgapi_fixture.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

using amd::dbgapi::test::dbgapi_fixture_t;

using HandleMisuse = dbgapi_fixture_t;

namespace
{

/* A handle value that no live object map will ever contain.  Using
   0xdead... keeps it well clear of the small monotonic IDs the library
   hands out so we don't accidentally clash with a live handle.  */
constexpr uint64_t bogus_handle = 0xdeadbeefcafef00dull;

amd_dbgapi_process_id_t
bogus_process ()
{
  return amd_dbgapi_process_id_t{ bogus_handle };
}

amd_dbgapi_agent_id_t
bogus_agent ()
{
  return amd_dbgapi_agent_id_t{ bogus_handle };
}

amd_dbgapi_queue_id_t
bogus_queue ()
{
  return amd_dbgapi_queue_id_t{ bogus_handle };
}

amd_dbgapi_wave_id_t
bogus_wave ()
{
  return amd_dbgapi_wave_id_t{ bogus_handle };
}

amd_dbgapi_dispatch_id_t
bogus_dispatch ()
{
  return amd_dbgapi_dispatch_id_t{ bogus_handle };
}

amd_dbgapi_code_object_id_t
bogus_code_object ()
{
  return amd_dbgapi_code_object_id_t{ bogus_handle };
}

amd_dbgapi_displaced_stepping_id_t
bogus_displaced_stepping ()
{
  return amd_dbgapi_displaced_stepping_id_t{ bogus_handle };
}

amd_dbgapi_watchpoint_id_t
bogus_watchpoint ()
{
  return amd_dbgapi_watchpoint_id_t{ bogus_handle };
}

amd_dbgapi_breakpoint_id_t
bogus_breakpoint ()
{
  return amd_dbgapi_breakpoint_id_t{ bogus_handle };
}

amd_dbgapi_register_id_t
bogus_register ()
{
  return amd_dbgapi_register_id_t{ bogus_handle };
}

amd_dbgapi_event_id_t
bogus_event ()
{
  return amd_dbgapi_event_id_t{ bogus_handle };
}

amd_dbgapi_address_space_id_t
bogus_address_space ()
{
  return amd_dbgapi_address_space_id_t{ bogus_handle };
}

amd_dbgapi_address_class_id_t
bogus_address_class ()
{
  return amd_dbgapi_address_class_id_t{ bogus_handle };
}

amd_dbgapi_register_class_id_t
bogus_register_class ()
{
  return amd_dbgapi_register_class_id_t{ bogus_handle };
}

amd_dbgapi_architecture_id_t
bogus_architecture ()
{
  return amd_dbgapi_architecture_id_t{ bogus_handle };
}

} /* namespace */

/* ------------------------------------------------------------------ */
/* Process handle                                                      */
/* ------------------------------------------------------------------ */

TEST_F (HandleMisuse, ProcessDetachRejectsBogusHandle)
{
  EXPECT_EQ (amd_dbgapi_process_detach (bogus_process ()),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

TEST_F (HandleMisuse, ProcessGetInfoRejectsBogusHandle)
{
  amd_dbgapi_os_process_id_t out = 0;
  EXPECT_EQ (
    amd_dbgapi_process_get_info (
      bogus_process (), AMD_DBGAPI_PROCESS_INFO_OS_ID, sizeof (out), &out),
    AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

TEST_F (HandleMisuse, ProcessAgentListRejectsBogusHandle)
{
  size_t count = 0;
  amd_dbgapi_agent_id_t *agents = nullptr;
  EXPECT_EQ (amd_dbgapi_process_agent_list (bogus_process (), &count, &agents,
                                            nullptr),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

TEST_F (HandleMisuse, ProcessSetProgressRejectsBogusHandle)
{
  EXPECT_EQ (amd_dbgapi_process_set_progress (
               bogus_process (), AMD_DBGAPI_PROGRESS_NORMAL),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

/* ------------------------------------------------------------------ */
/* Per-object get_info rejects bogus handles                           */
/* ------------------------------------------------------------------ */

TEST_F (HandleMisuse, AgentGetInfoRejectsBogusHandle)
{
  amd_dbgapi_os_agent_id_t out = 0;
  EXPECT_EQ (amd_dbgapi_agent_get_info (bogus_agent (),
                                        AMD_DBGAPI_AGENT_INFO_OS_ID,
                                        sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID);
}

TEST_F (HandleMisuse, QueueGetInfoRejectsBogusHandle)
{
  amd_dbgapi_os_queue_id_t out = 0;
  EXPECT_EQ (amd_dbgapi_queue_get_info (bogus_queue (),
                                        AMD_DBGAPI_QUEUE_INFO_OS_ID,
                                        sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_QUEUE_ID);
}

TEST_F (HandleMisuse, WaveGetInfoRejectsBogusHandle)
{
  amd_dbgapi_wave_state_t out{};
  EXPECT_EQ (amd_dbgapi_wave_get_info (bogus_wave (),
                                       AMD_DBGAPI_WAVE_INFO_STATE,
                                       sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

TEST_F (HandleMisuse, DispatchGetInfoRejectsBogusHandle)
{
  amd_dbgapi_os_queue_packet_id_t out = 0;
  EXPECT_EQ (amd_dbgapi_dispatch_get_info (
               bogus_dispatch (),
               AMD_DBGAPI_DISPATCH_INFO_OS_QUEUE_PACKET_ID, sizeof (out),
               &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_DISPATCH_ID);
}

TEST_F (HandleMisuse, CodeObjectGetInfoRejectsBogusHandle)
{
  amd_dbgapi_global_address_t out = 0;
  EXPECT_EQ (amd_dbgapi_code_object_get_info (
               bogus_code_object (), AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS,
               sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_CODE_OBJECT_ID);
}

TEST_F (HandleMisuse, WatchpointGetInfoRejectsBogusHandle)
{
  amd_dbgapi_global_address_t out = 0;
  EXPECT_EQ (amd_dbgapi_watchpoint_get_info (
               bogus_watchpoint (), AMD_DBGAPI_WATCHPOINT_INFO_ADDRESS,
               sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WATCHPOINT_ID);
}

TEST_F (HandleMisuse, EventGetInfoRejectsBogusHandle)
{
  amd_dbgapi_event_kind_t out{};
  EXPECT_EQ (amd_dbgapi_event_get_info (bogus_event (),
                                        AMD_DBGAPI_EVENT_INFO_KIND,
                                        sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_EVENT_ID);
}

TEST_F (HandleMisuse, ArchitectureGetInfoRejectsBogusHandle)
{
  uint32_t out = 0;
  EXPECT_EQ (amd_dbgapi_architecture_get_info (
               bogus_architecture (),
               AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE,
               sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID);
}

TEST_F (HandleMisuse, AddressSpaceGetInfoRejectsBogusHandle)
{
  amd_dbgapi_size_t out = 0;
  EXPECT_EQ (amd_dbgapi_address_space_get_info (
               bogus_address_space (),
               AMD_DBGAPI_ADDRESS_SPACE_INFO_ADDRESS_SIZE, sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_SPACE_ID);
}

TEST_F (HandleMisuse, AddressClassGetInfoRejectsBogusHandle)
{
  uint64_t out = 0;
  EXPECT_EQ (amd_dbgapi_address_class_get_info (
               bogus_address_class (), AMD_DBGAPI_ADDRESS_CLASS_INFO_DWARF,
               sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ADDRESS_CLASS_ID);
}

TEST_F (HandleMisuse, RegisterGetInfoRejectsBogusHandle)
{
  amd_dbgapi_size_t out = 0;
  EXPECT_EQ (amd_dbgapi_register_get_info (bogus_register (),
                                           AMD_DBGAPI_REGISTER_INFO_SIZE,
                                           sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID);
}

TEST_F (HandleMisuse, RegisterClassGetInfoRejectsBogusHandle)
{
  amd_dbgapi_architecture_id_t out{};
  EXPECT_EQ (amd_dbgapi_architecture_register_class_get_info (
               bogus_register_class (),
               AMD_DBGAPI_REGISTER_CLASS_INFO_ARCHITECTURE, sizeof (out),
               &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_CLASS_ID);
}

TEST_F (HandleMisuse, DisplacedSteppingGetInfoRejectsBogusHandle)
{
  amd_dbgapi_global_address_t out = 0;
  EXPECT_EQ (amd_dbgapi_displaced_stepping_get_info (
               bogus_displaced_stepping (),
               AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS, sizeof (out),
               &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID);
}

/* ------------------------------------------------------------------ */
/* Wave-state mutations reject bogus handles                           */
/* ------------------------------------------------------------------ */

TEST_F (HandleMisuse, WaveStopRejectsBogusHandle)
{
  EXPECT_EQ (amd_dbgapi_wave_stop (bogus_wave ()),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

TEST_F (HandleMisuse, WaveResumeRejectsBogusHandle)
{
  EXPECT_EQ (amd_dbgapi_wave_resume (bogus_wave (),
                                     AMD_DBGAPI_RESUME_MODE_NORMAL,
                                     AMD_DBGAPI_EXCEPTION_NONE),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

/* ------------------------------------------------------------------ */
/* Breakpoint surfaces reject bogus handles                            */
/* ------------------------------------------------------------------ */

TEST_F (HandleMisuse, BreakpointGetInfoRejectsBogusHandle)
{
  amd_dbgapi_process_id_t out{};
  EXPECT_EQ (amd_dbgapi_breakpoint_get_info (
               bogus_breakpoint (), AMD_DBGAPI_BREAKPOINT_INFO_PROCESS,
               sizeof (out), &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID);
}
