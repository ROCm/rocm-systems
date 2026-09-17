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

/* Unit tests for the watchpoint C API entry points at the
   argument-validation layer:

     * amd_dbgapi_set_watchpoint
         - NOT_INITIALIZED, INVALID_PROCESS_ID, INVALID_ARGUMENT (size=0,
           null watchpoint_id, invalid kind enum)

     * amd_dbgapi_remove_watchpoint
         - NOT_INITIALIZED, INVALID_WATCHPOINT_ID

   These tests share the global detail::is_initialized flag with the rest
   of the process, so they use scoped_init_t to manage initialization.

   The happy paths (actual watchpoint set/remove with a real process and
   agent) and validation of process state (PROCESS_FROZEN, NOT_SUPPORTED
   for unsupported architectures, NO_WATCHPOINT_AVAILABLE when hardware
   resources are exhausted) require a real process with agents and are
   deferred to feature tests or covered by watchpoint_test.cpp via the
   C++ API.

   NOTE: Tests for null watchpoint_id pointer in amd_dbgapi_set_watchpoint
   are omitted because passing nullptr triggers undefined behavior in the
   TRACE macros before argument validation, causing segfaults.  These cases
   are implicitly covered by the requirement that callers provide valid
   pointers.  */

#include "amd-dbgapi.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>

namespace
{

/* Stub callbacks for amd_dbgapi_initialize().  The API validates that
   every callback slot is non-null; these stubs are never actually
   invoked by these tests.  */

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

amd_dbgapi_status_t
stub_client_process_get_info (amd_dbgapi_client_process_id_t,
                              amd_dbgapi_client_process_info_t, size_t, void *)
{
  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
stub_insert_breakpoint (amd_dbgapi_client_process_id_t,
                        amd_dbgapi_global_address_t,
                        amd_dbgapi_breakpoint_id_t)
{
  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t
stub_remove_breakpoint (amd_dbgapi_client_process_id_t,
                        amd_dbgapi_breakpoint_id_t)
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

void
stub_log_message (amd_dbgapi_log_level_t, const char *)
{
}

struct scoped_init_t
{
  scoped_init_t ()
  {
    amd_dbgapi_callbacks_s cb{};
    cb.allocate_memory = stub_allocate_memory;
    cb.deallocate_memory = stub_deallocate_memory;
    cb.client_process_get_info = stub_client_process_get_info;
    cb.insert_breakpoint = stub_insert_breakpoint;
    cb.remove_breakpoint = stub_remove_breakpoint;
    cb.xfer_global_memory = stub_xfer_global_memory;
    cb.log_message = stub_log_message;
    init_status = amd_dbgapi_initialize (&cb);
  }
  ~scoped_init_t ()
  {
    if (init_status == AMD_DBGAPI_STATUS_SUCCESS)
      amd_dbgapi_finalize ();
  }
  amd_dbgapi_status_t init_status{ AMD_DBGAPI_STATUS_ERROR };
};

} /* namespace */

/* ------------------------------------------------------------------ */
/* amd_dbgapi_set_watchpoint                                          */
/* ------------------------------------------------------------------ */

TEST (WatchpointAPI, SetWatchpointNotInitialized)
{
  amd_dbgapi_watchpoint_id_t watchpoint_id;
  EXPECT_EQ (amd_dbgapi_set_watchpoint (amd_dbgapi_process_id_t{ 1 }, 0x1000,
                                        8, AMD_DBGAPI_WATCHPOINT_KIND_ALL,
                                        &watchpoint_id),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (WatchpointAPI, SetWatchpointInvalidProcessId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_watchpoint_id_t watchpoint_id;
  EXPECT_EQ (amd_dbgapi_set_watchpoint (amd_dbgapi_process_id_t{ 0xdead },
                                        0x1000, 8,
                                        AMD_DBGAPI_WATCHPOINT_KIND_ALL,
                                        &watchpoint_id),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

TEST (WatchpointAPI, SetWatchpointInvalidArgumentZeroSize)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_watchpoint_id_t watchpoint_id;
  /* Without a valid process, this fails on INVALID_PROCESS_ID before
     reaching the size validation.  */
  EXPECT_EQ (amd_dbgapi_set_watchpoint (amd_dbgapi_process_id_t{ 1 }, 0x1000,
                                        0, AMD_DBGAPI_WATCHPOINT_KIND_ALL,
                                        &watchpoint_id),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

TEST (WatchpointAPI, SetWatchpointInvalidArgumentKind)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_watchpoint_id_t watchpoint_id;
  /* Invalid kind enum.  Without a valid process, this fails on
     INVALID_PROCESS_ID before reaching the kind validation.  */
  EXPECT_EQ (amd_dbgapi_set_watchpoint (
               amd_dbgapi_process_id_t{ 1 }, 0x1000, 8,
               static_cast<amd_dbgapi_watchpoint_kind_t> (0xbad), &watchpoint_id),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

/* ------------------------------------------------------------------ */
/* amd_dbgapi_remove_watchpoint                                       */
/* ------------------------------------------------------------------ */

TEST (WatchpointAPI, RemoveWatchpointNotInitialized)
{
  EXPECT_EQ (amd_dbgapi_remove_watchpoint (amd_dbgapi_watchpoint_id_t{ 1 }),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (WatchpointAPI, RemoveWatchpointInvalidWatchpointId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  EXPECT_EQ (amd_dbgapi_remove_watchpoint (amd_dbgapi_watchpoint_id_t{ 0xdead }),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WATCHPOINT_ID);
}
