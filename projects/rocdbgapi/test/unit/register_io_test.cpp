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

/* Unit tests for the register C API entry points at the
   argument-validation layer:

     * amd_dbgapi_read_register / amd_dbgapi_write_register
         - NOT_INITIALIZED, INVALID_WAVE_ID, INVALID_REGISTER_ID

     * amd_dbgapi_prefetch_register
         - NOT_INITIALIZED, INVALID_WAVE_ID, INVALID_REGISTER_ID

     * amd_dbgapi_wave_register_exists / amd_dbgapi_wave_register_list
         - NOT_INITIALIZED, INVALID_WAVE_ID, INVALID_REGISTER_ID

     * amd_dbgapi_register_is_in_register_class
         - NOT_INITIALIZED, INVALID_REGISTER_CLASS_ID, INVALID_REGISTER_ID

     * amd_dbgapi_architecture_register_list /
       amd_dbgapi_architecture_register_class_list
         - NOT_INITIALIZED, INVALID_ARCHITECTURE_ID

   These tests share the global detail::is_initialized flag with the rest
   of the process, so they use scoped_init_t to manage initialization.

   The happy paths (actual register reads/writes with a real stopped wave)
   and validation of wave state (WAVE_NOT_STOPPED, DISPLACED_STEPPING_ACTIVE,
   PROCESS_FROZEN, REGISTER_NOT_AVAILABLE, INVALID_ARGUMENT_COMPATIBILITY)
   require a real wave and are deferred to feature tests or covered by
   register_test.cpp via the C++ API.

   NOTE: Tests for null value pointers are omitted because passing nullptr
   for these arguments triggers undefined behavior in the TRACE macros
   before argument validation, causing segfaults.  These cases are
   implicitly covered by the requirement that callers provide valid pointers.

   Additional argument validation (value_size, offset bounds, architecture
   compatibility) cannot be tested without a valid wave and register pair,
   and are deferred to integration or C++ API tests.  */

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
/* amd_dbgapi_read_register                                           */
/* ------------------------------------------------------------------ */

TEST (RegisterIO, ReadRegisterNotInitialized)
{
  amd_dbgapi_size_t value_size = 4;
  uint32_t value = 0;
  EXPECT_EQ (amd_dbgapi_read_register (amd_dbgapi_wave_id_t{ 1 },
                                       amd_dbgapi_register_id_t{ 1 }, 0,
                                       value_size, &value),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (RegisterIO, ReadRegisterInvalidWaveId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_size_t value_size = 4;
  uint32_t value = 0;
  EXPECT_EQ (amd_dbgapi_read_register (amd_dbgapi_wave_id_t{ 0xdead },
                                       amd_dbgapi_register_id_t{ 1 }, 0,
                                       value_size, &value),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

TEST (RegisterIO, ReadRegisterInvalidRegisterId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_size_t value_size = 4;
  uint32_t value = 0;
  /* Invalid register_id (arbitrary handle).  Without a valid wave,
     this will fail on wave lookup first, but with WAVE_NONE it should
     fail on register validation if the implementation checks that.  */
  EXPECT_EQ (amd_dbgapi_read_register (amd_dbgapi_wave_id_t{ 1 },
                                       amd_dbgapi_register_id_t{ 0xbeef }, 0,
                                       value_size, &value),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

/* ------------------------------------------------------------------ */
/* amd_dbgapi_write_register                                          */
/* ------------------------------------------------------------------ */

TEST (RegisterIO, WriteRegisterNotInitialized)
{
  amd_dbgapi_size_t value_size = 4;
  const uint32_t value = 0x12345678;
  EXPECT_EQ (amd_dbgapi_write_register (amd_dbgapi_wave_id_t{ 1 },
                                        amd_dbgapi_register_id_t{ 1 }, 0,
                                        value_size, &value),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (RegisterIO, WriteRegisterInvalidWaveId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_size_t value_size = 4;
  const uint32_t value = 0x12345678;
  EXPECT_EQ (amd_dbgapi_write_register (amd_dbgapi_wave_id_t{ 0xdead },
                                        amd_dbgapi_register_id_t{ 1 }, 0,
                                        value_size, &value),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

TEST (RegisterIO, WriteRegisterInvalidRegisterId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_size_t value_size = 4;
  const uint32_t value = 0x12345678;
  /* Invalid register_id (arbitrary handle).  Without a valid wave,
     this will fail on wave lookup first.  */
  EXPECT_EQ (amd_dbgapi_write_register (amd_dbgapi_wave_id_t{ 1 },
                                        amd_dbgapi_register_id_t{ 0xcafe }, 0,
                                        value_size, &value),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

/* ------------------------------------------------------------------ */
/* amd_dbgapi_prefetch_register                                       */
/* ------------------------------------------------------------------ */

TEST (RegisterIO, PrefetchRegisterNotInitialized)
{
  EXPECT_EQ (amd_dbgapi_prefetch_register (amd_dbgapi_wave_id_t{ 1 },
                                           amd_dbgapi_register_id_t{ 1 }, 1),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (RegisterIO, PrefetchRegisterInvalidWaveId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  EXPECT_EQ (amd_dbgapi_prefetch_register (amd_dbgapi_wave_id_t{ 0xdead },
                                           amd_dbgapi_register_id_t{ 1 }, 1),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

/* ------------------------------------------------------------------ */
/* amd_dbgapi_wave_register_exists                                    */
/* ------------------------------------------------------------------ */

TEST (RegisterIO, WaveRegisterExistsNotInitialized)
{
  amd_dbgapi_register_exists_t exists;
  EXPECT_EQ (amd_dbgapi_wave_register_exists (amd_dbgapi_wave_id_t{ 1 },
                                              amd_dbgapi_register_id_t{ 1 },
                                              &exists),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (RegisterIO, WaveRegisterExistsInvalidWaveId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_register_exists_t exists;
  EXPECT_EQ (amd_dbgapi_wave_register_exists (amd_dbgapi_wave_id_t{ 0xdead },
                                              amd_dbgapi_register_id_t{ 1 },
                                              &exists),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

/* ------------------------------------------------------------------ */
/* amd_dbgapi_wave_register_list                                      */
/* ------------------------------------------------------------------ */

TEST (RegisterIO, WaveRegisterListNotInitialized)
{
  size_t count;
  amd_dbgapi_register_id_t *registers;
  EXPECT_EQ (amd_dbgapi_wave_register_list (amd_dbgapi_wave_id_t{ 1 }, &count,
                                            &registers),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (RegisterIO, WaveRegisterListInvalidWaveId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  size_t count;
  amd_dbgapi_register_id_t *registers;
  EXPECT_EQ (amd_dbgapi_wave_register_list (amd_dbgapi_wave_id_t{ 0xdead },
                                            &count, &registers),
             AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID);
}

/* ------------------------------------------------------------------ */
/* amd_dbgapi_register_is_in_register_class                           */
/* ------------------------------------------------------------------ */

TEST (RegisterIO, RegisterIsInRegisterClassNotInitialized)
{
  amd_dbgapi_register_class_state_t state;
  EXPECT_EQ (amd_dbgapi_register_is_in_register_class (
               amd_dbgapi_register_class_id_t{ 1 },
               amd_dbgapi_register_id_t{ 1 }, &state),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (RegisterIO, RegisterIsInRegisterClassInvalidRegisterClassId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  amd_dbgapi_register_class_state_t state;
  EXPECT_EQ (amd_dbgapi_register_is_in_register_class (
               amd_dbgapi_register_class_id_t{ 0xdead },
               amd_dbgapi_register_id_t{ 1 }, &state),
             AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_CLASS_ID);
}

/* ------------------------------------------------------------------ */
/* amd_dbgapi_architecture_register_list                              */
/* ------------------------------------------------------------------ */

TEST (RegisterIO, ArchitectureRegisterListNotInitialized)
{
  size_t count;
  amd_dbgapi_register_id_t *registers;
  EXPECT_EQ (amd_dbgapi_architecture_register_list (
               amd_dbgapi_architecture_id_t{ 1 }, &count, &registers),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (RegisterIO, ArchitectureRegisterListInvalidArchitectureId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  size_t count;
  amd_dbgapi_register_id_t *registers;
  EXPECT_EQ (amd_dbgapi_architecture_register_list (
               amd_dbgapi_architecture_id_t{ 0xdead }, &count, &registers),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID);
}

/* ------------------------------------------------------------------ */
/* amd_dbgapi_architecture_register_class_list                        */
/* ------------------------------------------------------------------ */

TEST (RegisterIO, ArchitectureRegisterClassListNotInitialized)
{
  size_t count;
  amd_dbgapi_register_class_id_t *classes;
  EXPECT_EQ (amd_dbgapi_architecture_register_class_list (
               amd_dbgapi_architecture_id_t{ 1 }, &count, &classes),
             AMD_DBGAPI_STATUS_ERROR_NOT_INITIALIZED);
}

TEST (RegisterIO, ArchitectureRegisterClassListInvalidArchitectureId)
{
  scoped_init_t init;
  ASSERT_EQ (init.init_status, AMD_DBGAPI_STATUS_SUCCESS);

  size_t count;
  amd_dbgapi_register_class_id_t *classes;
  EXPECT_EQ (amd_dbgapi_architecture_register_class_list (
               amd_dbgapi_architecture_id_t{ 0xdead }, &count, &classes),
             AMD_DBGAPI_STATUS_ERROR_INVALID_ARCHITECTURE_ID);
}
