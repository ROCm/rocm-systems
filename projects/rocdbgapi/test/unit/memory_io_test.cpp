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

/* Unit tests for the memory I/O C API entry points
   (amd_dbgapi_read_memory and amd_dbgapi_write_memory) at the
   argument-validation layer that does not need a real process or wave:

     * amd_dbgapi_read_memory
         - NOT_INITIALIZED when called before amd_dbgapi_initialize
         - INVALID_PROCESS_ID for an arbitrary handle once initialized

     * amd_dbgapi_write_memory
         - NOT_INITIALIZED when called before amd_dbgapi_initialize
         - INVALID_PROCESS_ID for an arbitrary handle once initialized

   These tests share the global detail::is_initialized flag with the rest
   of the process, so they use scoped_init_t to manage initialization.

   The happy paths (actual memory reads/writes with a real process) and
   validation of address_space/wave/lane arguments are covered by
   process_lifecycle_test.cpp via the C++ API (process_t::read_global_memory_partial
   / xfer_host_memory_partial) which can construct mock processes.

   NOTE: Tests for null value_size or value pointers are omitted because
   passing nullptr for these arguments triggers undefined behavior in the
   TRACE macros before argument validation, causing segfaults.  These
   cases are implicitly covered by the requirement that callers provide
   valid pointers.

   Additional argument validation (INVALID_ADDRESS_SPACE_ID, INVALID_WAVE_ID,
   INVALID_LANE_ID) cannot be tested in a pure unit-test environment because
   the validation order checks process validity first, and without a real
   process in the global process map, all calls return INVALID_PROCESS_ID
   before reaching the other validation checks.  */

#include "amd-dbgapi.h"
#include "memory.h"

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
/* amd_dbgapi_read_memory                                             */
/* ------------------------------------------------------------------ */

TEST (MemoryIO, ReadMemoryNotInitialized)
{
  amd_dbgapi_size_t size = 8;
  char buf[8];
  EXPECT_EQ (amd_dbgapi_read_memory (amd_dbgapi_process_id_t{ 1 },
                                     AMD_DBGAPI_WAVE_NONE,
                                     AMD_DBGAPI_LANE_NONE,
                                     amd::dbgapi::address_space_t::global ().id (),
                                     0x1000, &size, buf),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (MemoryIO, ReadMemoryInvalidProcessId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_size_t size = 8;
  char buf[8];
  EXPECT_EQ (amd_dbgapi_read_memory (amd_dbgapi_process_id_t{ 0xdead },
                                     AMD_DBGAPI_WAVE_NONE,
                                     AMD_DBGAPI_LANE_NONE,
                                     amd::dbgapi::address_space_t::global ().id (),
                                     0x1000, &size, buf),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}

/* ------------------------------------------------------------------ */
/* amd_dbgapi_write_memory                                            */
/* ------------------------------------------------------------------ */

TEST (MemoryIO, WriteMemoryNotInitialized)
{
  amd_dbgapi_size_t size = 8;
  const char data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
  EXPECT_EQ (amd_dbgapi_write_memory (amd_dbgapi_process_id_t{ 1 },
                                      AMD_DBGAPI_WAVE_NONE,
                                      AMD_DBGAPI_LANE_NONE,
                                      amd::dbgapi::address_space_t::global ().id (),
                                      0x1000, &size, data),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (MemoryIO, WriteMemoryInvalidProcessId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_size_t size = 8;
  const char data[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
  EXPECT_EQ (amd_dbgapi_write_memory (amd_dbgapi_process_id_t{ 0xdead },
                                      AMD_DBGAPI_WAVE_NONE,
                                      AMD_DBGAPI_LANE_NONE,
                                      amd::dbgapi::address_space_t::global ().id (),
                                      0x1000, &size, data),
             AMD_DBGAPI_STATUS_ERROR_INVALID_PROCESS_ID);
}
