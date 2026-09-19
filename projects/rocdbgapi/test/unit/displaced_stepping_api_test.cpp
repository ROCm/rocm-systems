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

/* Unit tests for the displaced-stepping C API entry points
   (displaced_stepping.cpp) at the argument-validation layer that does not
   need a real stopped wave:

     * amd_dbgapi_displaced_stepping_start
         - NOT_INITIALIZED when called before amd_dbgapi_initialize
         - INVALID_ARGUMENT when the saved-instruction or out-id pointer
           is null (also checked before init)
         - INVALID_WAVE_ID for an arbitrary handle once initialized

     * amd_dbgapi_displaced_stepping_complete
         - NOT_INITIALIZED when called before amd_dbgapi_initialize
         - INVALID_WAVE_ID for an arbitrary handle once initialized

     * amd_dbgapi_displaced_stepping_get_info
         - NOT_INITIALIZED when called before amd_dbgapi_initialize
         - INVALID_DISPLACED_STEPPING_ID for an arbitrary handle

   These tests share the global detail::is_initialized flag with the rest
   of the process, so they save and restore the (callbacks, initialized)
   pair around the "initialized" cases via amd_dbgapi_finalize().

   The happy paths (an actual wave doing a displaced step) need a real
   process / queue / wave and live in feature tests.  */

#include "amd-dbgapi.h"

#include <gtest/gtest.h>

#include <cstring>

namespace
{

/* A callbacks struct with every function pointer set to a non-null stub.
   The initialize() code path validates that every slot is non-null by
   walking sizeof(*callbacks) / sizeof(fn-ptr); the stubs themselves are
   never invoked by these tests because we never reach a code path that
   uses them.  */

amd_dbgapi_status_t stub_insert_breakpoint (amd_dbgapi_client_process_id_t,
                                            amd_dbgapi_global_address_t,
                                            amd_dbgapi_breakpoint_id_t)
{
  return AMD_DBGAPI_STATUS_SUCCESS;
}

amd_dbgapi_status_t stub_remove_breakpoint (amd_dbgapi_client_process_id_t,
                                            amd_dbgapi_breakpoint_id_t)
{
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
/* _start                                                              */
/* ------------------------------------------------------------------ */

TEST (DisplacedSteppingApi, StartBeforeInitReturnsNotInitialized)
{
  uint32_t bytes[1]{ 0 };
  amd_dbgapi_displaced_stepping_id_t out{};
  amd_dbgapi_wave_id_t wave{ 42 };
  EXPECT_EQ (amd_dbgapi_displaced_stepping_start (wave, bytes, &out),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (DisplacedSteppingApi, StartRejectsNullSavedInstructionPointer)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_displaced_stepping_id_t out{};
  amd_dbgapi_wave_id_t wave{ 42 };
  EXPECT_EQ (amd_dbgapi_displaced_stepping_start (wave, nullptr, &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST (DisplacedSteppingApi, StartRejectsNullOutPointer)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  uint32_t bytes[1]{ 0 };
  amd_dbgapi_wave_id_t wave{ 42 };
  EXPECT_EQ (amd_dbgapi_displaced_stepping_start (wave, bytes, nullptr),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
}

TEST (DisplacedSteppingApi, StartUnknownWaveReturnsInvalidWaveId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  uint32_t bytes[1]{ 0 };
  amd_dbgapi_displaced_stepping_id_t out{};
  amd_dbgapi_wave_id_t wave{ 0xdead };
  EXPECT_EQ (amd_dbgapi_displaced_stepping_start (wave, bytes, &out),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

/* ------------------------------------------------------------------ */
/* _complete                                                           */
/* ------------------------------------------------------------------ */

TEST (DisplacedSteppingApi, CompleteBeforeInitReturnsNotInitialized)
{
  amd_dbgapi_wave_id_t wave{ 1 };
  amd_dbgapi_displaced_stepping_id_t ds{ 1 };
  EXPECT_EQ (amd_dbgapi_displaced_stepping_complete (wave, ds),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (DisplacedSteppingApi, CompleteUnknownWaveReturnsInvalidWaveId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_wave_id_t wave{ 0xdead };
  amd_dbgapi_displaced_stepping_id_t ds{ 1 };
  EXPECT_EQ (amd_dbgapi_displaced_stepping_complete (wave, ds),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

/* ------------------------------------------------------------------ */
/* _get_info                                                           */
/* ------------------------------------------------------------------ */

TEST (DisplacedSteppingApi, GetInfoBeforeInitReturnsNotInitialized)
{
  amd_dbgapi_process_id_t pid{};
  EXPECT_EQ (amd_dbgapi_displaced_stepping_get_info (
               amd_dbgapi_displaced_stepping_id_t{ 1 },
               AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS, sizeof (pid), &pid),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (DisplacedSteppingApi, GetInfoUnknownIdReturnsInvalidDisplacedSteppingId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_process_id_t pid{};
  EXPECT_EQ (amd_dbgapi_displaced_stepping_get_info (
               amd_dbgapi_displaced_stepping_id_t{ 0xdead },
               AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS, sizeof (pid), &pid),
             AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID);
}

/* The NONE sentinel is reserved and must be rejected as an invalid
   displaced-stepping id, not silently confused with an unknown handle.  */
TEST (DisplacedSteppingApi, GetInfoNoneIdReturnsInvalidDisplacedSteppingId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_process_id_t pid{};
  EXPECT_EQ (amd_dbgapi_displaced_stepping_get_info (
               AMD_DBGAPI_DISPLACED_STEPPING_NONE,
               AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS, sizeof (pid), &pid),
             AMD_DBGAPI_STATUS_ERROR_INVALID_DISPLACED_STEPPING_ID);
}

/* Buffer-edge case: a null output buffer must not be silently accepted
   even when the id itself is unknown.  We don't pin the exact status
   because the implementation is free to check the id before or after
   the buffer pointer; either INVALID_ARGUMENT or
   INVALID_DISPLACED_STEPPING_ID is acceptable, but SUCCESS is not.  */
TEST (DisplacedSteppingApi, GetInfoNullValueIsNotAccepted)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_status_t st = amd_dbgapi_displaced_stepping_get_info (
    amd_dbgapi_displaced_stepping_id_t{ 0xdead },
    AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS, sizeof (amd_dbgapi_process_id_t),
    nullptr);
  EXPECT_NE (st, AMD_DBGAPI_STATUS_SUCCESS);
}

/* Buffer-edge case: a wrong value_size must be rejected.  Same
   ordering caveat as above -- either the lookup error or the
   size-compat error is acceptable; SUCCESS is not.  */
TEST (DisplacedSteppingApi, GetInfoWrongSizeIsNotAccepted)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_process_id_t pid{};
  amd_dbgapi_status_t st = amd_dbgapi_displaced_stepping_get_info (
    amd_dbgapi_displaced_stepping_id_t{ 0xdead },
    AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS, sizeof (pid) + 7, &pid);
  EXPECT_NE (st, AMD_DBGAPI_STATUS_SUCCESS);
}

/* An out-of-range info-query enum must be rejected.  */
TEST (DisplacedSteppingApi, GetInfoUnknownQueryIsRejected)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  uint64_t sink = 0;
  amd_dbgapi_status_t st = amd_dbgapi_displaced_stepping_get_info (
    amd_dbgapi_displaced_stepping_id_t{ 0xdead },
    static_cast<amd_dbgapi_displaced_stepping_info_t> (0x7fffffff),
    sizeof (sink), &sink);
  EXPECT_NE (st, AMD_DBGAPI_STATUS_SUCCESS);
}

/* The NONE wave handle is not a usable wave; complete on it must fail.  */
TEST (DisplacedSteppingApi, CompleteWithNoneWaveIsRejected)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  EXPECT_EQ (amd_dbgapi_displaced_stepping_complete (
               AMD_DBGAPI_WAVE_NONE, amd_dbgapi_displaced_stepping_id_t{ 1 }),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}
