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

/* Unit tests for process_t surfaces that do NOT require agent_t / queue_t
   construction.  Anything that walks process.range<agent_t>() or queue_t> is
   deferred to Commits 9/14 once those types have a friend-key seam.

   Covered here:
     * flag_t set/clear/is_flag_set bookkeeping.
     * from_core(): true iff no os_process_id was supplied.
     * Default-state accessors (forward_progress_needed, is_frozen,
       wave_launch_mode).
     * set_wave_launch_mode / set_wave_launch_trap_override "record only"
       behavior when the runtime is UNLOADED — verified by asserting the
       driver method is NEVER called on the mock.
     * set_precise_memory / set_precise_alu_exceptions throw NOT_SUPPORTED
       when the process didn't observe a "supports..." runtime_info.
     * xfer_global_memory / xfer_host_memory: delegate to the os_driver and
       translate driver status codes to the documented exception types.  */

#include "amd-dbgapi.h"
#include "exception.h"
#include "os_driver.h"
#include "process.h"

#include "support/mock_os_driver.h"
#include "support/process_test_access.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

using ::testing::_;
using ::testing::DoAll;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::SetArgPointee;

using amd::dbgapi::api_error_t;
using amd::dbgapi::memory_access_error_t;
using amd::dbgapi::memory_unavailable_error_t;
using amd::dbgapi::os_driver_t;
using amd::dbgapi::os_wave_launch_mode_t;
using amd::dbgapi::os_wave_launch_trap_mask_t;
using amd::dbgapi::process_exited_exception_t;
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

/* Build a process_t bound to a NiceMock<MockOsDriver>.  Returns the
   process and a raw pointer to the mock so callers can program
   expectations after construction.  Programs is_valid() to default
   true so the test ctor's validity check passes.  */
struct test_process_t
{
  std::unique_ptr<process_t> process;
  NiceMock<MockOsDriver> *driver;
};

test_process_t
make_test_process (amd_dbgapi_process_id_t id = amd_dbgapi_process_id_t{ 1 },
                   std::optional<amd_dbgapi_os_process_id_t> os_pid
                   = std::nullopt)
{
  auto driver = std::make_unique<NiceMock<MockOsDriver>> ();
  ON_CALL (*driver, is_valid ()).WillByDefault (Return (true));
  auto *driver_raw = driver.get ();
  auto proc = process_test_access::make (
    id, fake_client_id (), os_pid,
    std::unique_ptr<os_driver_t> (driver.release ()));
  return { std::move (proc), driver_raw };
}

} /* namespace */

TEST (ProcessLifecycle, FromCoreTrueWhenNoOsPid)
{
  auto tp = make_test_process ();
  EXPECT_TRUE (tp.process->from_core ());
  EXPECT_FALSE (tp.process->os_id ().has_value ());
}

TEST (ProcessLifecycle, FromCoreFalseWhenOsPidProvided)
{
  auto tp = make_test_process (amd_dbgapi_process_id_t{ 7 },
                               amd_dbgapi_os_process_id_t{ 4242 });
  EXPECT_FALSE (tp.process->from_core ());
  ASSERT_TRUE (tp.process->os_id ().has_value ());
  EXPECT_EQ (*tp.process->os_id (), 4242);
  EXPECT_EQ (tp.process->id ().handle, 7u);
}

TEST (ProcessLifecycle, DefaultStateIsSane)
{
  auto tp = make_test_process ();
  EXPECT_TRUE (tp.process->forward_progress_needed ());
  EXPECT_FALSE (tp.process->is_frozen ());
  EXPECT_EQ (tp.process->wave_launch_mode (), os_wave_launch_mode_t::normal);
  EXPECT_EQ (tp.process->client_id (), fake_client_id ());
}

TEST (ProcessLifecycle, FlagSetClearIsSet)
{
  auto tp = make_test_process ();
  using flag_t = process_t::flag_t;

  EXPECT_FALSE (tp.process->is_flag_set (flag_t::runtime_enable_during_attach));
  EXPECT_FALSE (tp.process->is_flag_set (flag_t::spi_ttmps_setup_enabled));

  tp.process->set_flag (flag_t::runtime_enable_during_attach);
  EXPECT_TRUE (tp.process->is_flag_set (flag_t::runtime_enable_during_attach));
  EXPECT_FALSE (tp.process->is_flag_set (flag_t::spi_ttmps_setup_enabled));

  tp.process->set_flag (flag_t::spi_ttmps_setup_enabled);
  EXPECT_TRUE (tp.process->is_flag_set (flag_t::runtime_enable_during_attach));
  EXPECT_TRUE (tp.process->is_flag_set (flag_t::spi_ttmps_setup_enabled));

  tp.process->clear_flag (flag_t::runtime_enable_during_attach);
  EXPECT_FALSE (tp.process->is_flag_set (flag_t::runtime_enable_during_attach));
  EXPECT_TRUE (tp.process->is_flag_set (flag_t::spi_ttmps_setup_enabled));
}

TEST (ProcessLifecycle, UnfreezeKeepsFrozenClear)
{
  /* freeze() itself walks queues and agents, so it can't run in a
     unit-test TU without queue/agent seams.  unfreeze() just clears
     the flag and is exercised here.  */
  auto tp = make_test_process ();
  EXPECT_FALSE (tp.process->is_frozen ());
  tp.process->unfreeze ();
  EXPECT_FALSE (tp.process->is_frozen ());
}

TEST (ProcessLifecycle, SetWaveLaunchModeBeforeRuntimeLoadedRecordsOnly)
{
  /* Runtime state defaults to UNLOADED, so set_wave_launch_mode must
     NOT call the driver — it only records the value.  */
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, set_wave_launch_mode (_)).Times (0);

  tp.process->set_wave_launch_mode (os_wave_launch_mode_t::single_step);
  EXPECT_EQ (tp.process->wave_launch_mode (),
             os_wave_launch_mode_t::single_step);

  /* Setting the same value again is a no-op — also no driver call.  */
  tp.process->set_wave_launch_mode (os_wave_launch_mode_t::single_step);
  EXPECT_EQ (tp.process->wave_launch_mode (),
             os_wave_launch_mode_t::single_step);
}

TEST (ProcessLifecycle, SetWaveLaunchTrapOverrideBeforeRuntimeLoadedRecordsOnly)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, set_wave_launch_trap_override (_, _, _, _, _))
    .Times (0);

  /* No-op until runtime loaded.  Just verify no driver call escapes
     the "record only" branch.  */
  tp.process->set_wave_launch_trap_override (os_wave_launch_trap_mask_t::none,
                                             os_wave_launch_trap_mask_t::none);
}

TEST (ProcessLifecycle, SetPreciseMemoryThrowsNotSupportedByDefault)
{
  /* m_supports_precise_memory defaults to false; toggling from false
     to true must throw NOT_SUPPORTED before any driver call.  */
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, set_process_flags (_)).Times (0);

  try
    {
      tp.process->set_precise_memory (true);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED);
    }
}

TEST (ProcessLifecycle, SetPreciseAluExceptionsThrowsNotSupportedByDefault)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, set_process_flags (_)).Times (0);

  try
    {
      tp.process->set_precise_alu_exceptions (true);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED);
    }
}

/* xfer_global_memory: success path returns the (possibly partial) size
   the driver reported back via the in/out *size pointer.  */
TEST (ProcessLifecycle, XferGlobalMemoryDelegatesAndReturnsSize)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_global_memory_partial (_, _, _, _))
    .WillOnce (DoAll (SetArgPointee<3> (32),
                      Return (AMD_DBGAPI_STATUS_SUCCESS)));

  char buf[64];
  size_t got = tp.process->read_global_memory_partial (0x1000, buf, 64);
  EXPECT_EQ (got, 32u);
}

TEST (ProcessLifecycle, XferGlobalMemoryTranslatesProcessExited)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_global_memory_partial (_, _, _, _))
    .WillOnce (Return (AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED));

  char buf[8];
  EXPECT_THROW (
    { (void) tp.process->read_global_memory_partial (0x2000, buf, 8); },
    process_exited_exception_t);
}

TEST (ProcessLifecycle, XferGlobalMemoryTranslatesMemoryAccess)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_global_memory_partial (_, _, _, _))
    .WillOnce (Return (AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS));

  char buf[8];
  EXPECT_THROW (
    { (void) tp.process->read_global_memory_partial (0x3000, buf, 8); },
    memory_access_error_t);
}

TEST (ProcessLifecycle, XferGlobalMemoryTranslatesMemoryUnavailable)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_global_memory_partial (_, _, _, _))
    .WillOnce (Return (AMD_DBGAPI_STATUS_ERROR_MEMORY_UNAVAILABLE));

  char buf[8];
  EXPECT_THROW (
    { (void) tp.process->read_global_memory_partial (0x4000, buf, 8); },
    memory_unavailable_error_t);
}

TEST (ProcessLifecycle, XferHostMemoryDelegatesSuccess)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_host_memory_partial (_, _, _, _))
    .WillOnce (DoAll (SetArgPointee<3> (16),
                      Return (AMD_DBGAPI_STATUS_SUCCESS)));

  char buf[16];
  size_t got = tp.process->read_host_memory_partial (0x5000, buf, 16);
  EXPECT_EQ (got, 16u);
}

TEST (ProcessLifecycle, XferHostMemoryTranslatesProcessExited)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_host_memory_partial (_, _, _, _))
    .WillOnce (Return (AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED));

  char buf[8];
  EXPECT_THROW (
    { (void) tp.process->read_host_memory_partial (0x6000, buf, 8); },
    process_exited_exception_t);
}

/* C26 boundary cases: xfer write path + driver-reports-zero + host
   variant of memory-access translation.  The C25 read-path coverage
   above already validates the global read translations.  */

TEST (ProcessLifecycle, XferGlobalMemoryReturnsZeroWhenDriverReportsZero)
{
  /* Driver succeeds but reports 0 bytes transferred (e.g. the address
     fell in an unmapped page that the driver chose to short-circuit).
     The wrapper must return that 0 verbatim, NOT throw.  */
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_global_memory_partial (_, _, _, _))
    .WillOnce (DoAll (SetArgPointee<3> (0),
                      Return (AMD_DBGAPI_STATUS_SUCCESS)));

  char buf[8];
  size_t got = tp.process->read_global_memory_partial (0x7000, buf, 8);
  EXPECT_EQ (got, 0u);
}

TEST (ProcessLifecycle, WriteGlobalMemoryDelegatesAndReturnsSize)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_global_memory_partial (_, _, _, _))
    .WillOnce (DoAll (SetArgPointee<3> (24),
                      Return (AMD_DBGAPI_STATUS_SUCCESS)));

  char buf[32]{};
  size_t got = tp.process->write_global_memory_partial (0x8000, buf, 32);
  EXPECT_EQ (got, 24u);
}

TEST (ProcessLifecycle, WriteGlobalMemoryTranslatesProcessExited)
{
  /* The write path shares the xfer_global_memory wrapper with reads, so
     it must perform the same status->exception translation.  */
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_global_memory_partial (_, _, _, _))
    .WillOnce (Return (AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED));

  char buf[8]{};
  EXPECT_THROW (
    { (void) tp.process->write_global_memory_partial (0x9000, buf, 8); },
    process_exited_exception_t);
}

TEST (ProcessLifecycle, WriteHostMemoryDelegatesAndReturnsSize)
{
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_host_memory_partial (_, _, _, _))
    .WillOnce (DoAll (SetArgPointee<3> (8),
                      Return (AMD_DBGAPI_STATUS_SUCCESS)));

  char buf[8]{};
  size_t got = tp.process->write_host_memory_partial (0xA000, buf, 8);
  EXPECT_EQ (got, 8u);
}

TEST (ProcessLifecycle, XferHostMemoryTranslatesMemoryAccess)
{
  /* The host xfer wrapper only translates PROCESS_EXITED and
     MEMORY_ACCESS (MEMORY_UNAVAILABLE goes through fatal_error -- not
     tested because it would terminate the test binary).  Verify the
     ACCESS translation here; PROCESS_EXITED is covered above.  */
  auto tp = make_test_process ();
  EXPECT_CALL (*tp.driver, xfer_host_memory_partial (_, _, _, _))
    .WillOnce (Return (AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS));

  char buf[8];
  EXPECT_THROW (
    { (void) tp.process->read_host_memory_partial (0xB000, buf, 8); },
    memory_access_error_t);
}

/* C25 negative-test sweep: value_size and enum edge cases on
   process_t::get_info.  Use OS_PROCESS_ID since the test ctor populates
   m_os_process_id when an os_pid is supplied.  */

TEST (ProcessLifecycle, GetInfoValueSizeZeroThrowsCompatibility)
{
  auto tp = make_test_process (amd_dbgapi_process_id_t{ 1 },
                               amd_dbgapi_os_process_id_t{ 4242 });
  amd_dbgapi_os_process_id_t out{};
  try
    {
      tp.process->get_info (AMD_DBGAPI_PROCESS_INFO_OS_ID, 0, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (ProcessLifecycle, GetInfoValueSizeMaxThrowsCompatibility)
{
  auto tp = make_test_process (amd_dbgapi_process_id_t{ 1 },
                               amd_dbgapi_os_process_id_t{ 4242 });
  amd_dbgapi_os_process_id_t out{};
  try
    {
      tp.process->get_info (AMD_DBGAPI_PROCESS_INFO_OS_ID, SIZE_MAX,
                            &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (ProcessLifecycle, GetInfoValueSizeTooLargeByOneThrowsCompatibility)
{
  auto tp = make_test_process (amd_dbgapi_process_id_t{ 1 },
                               amd_dbgapi_os_process_id_t{ 4242 });
  uint8_t out[sizeof (amd_dbgapi_os_process_id_t) + 1]{};
  try
    {
      tp.process->get_info (AMD_DBGAPI_PROCESS_INFO_OS_ID,
                            sizeof (amd_dbgapi_os_process_id_t) + 1, out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (ProcessLifecycle, GetInfoNegativeEnumThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  amd_dbgapi_os_process_id_t out{};
  try
    {
      tp.process->get_info (static_cast<amd_dbgapi_process_info_t> (-1),
                            sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (ProcessLifecycle, GetInfoIntMaxEnumThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  amd_dbgapi_os_process_id_t out{};
  try
    {
      tp.process->get_info (static_cast<amd_dbgapi_process_info_t> (INT_MAX),
                            sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* ------------------------------------------------------------------ */
/* C28 — state-machine misuse on get_info / unfreeze                  */
/* ------------------------------------------------------------------ */

/* from_core() processes have no os_pid; OS_ID query must report
   NOT_AVAILABLE, not return a garbage handle.  */
TEST (ProcessLifecycle, GetInfoOsIdReturnsNotAvailableForCoreProcess)
{
  auto tp = make_test_process ();
  ASSERT_TRUE (tp.process->from_core ());
  amd_dbgapi_os_process_id_t out{};
  try
    {
      tp.process->get_info (AMD_DBGAPI_PROCESS_INFO_OS_ID, sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
    }
}

/* CORE_STATE on a from_core process: no os_pid → NOT_AVAILABLE
   takes precedence over PROCESS_NOT_FROZEN.  */
TEST (ProcessLifecycle, GetInfoCoreStateReturnsNotAvailableForCoreProcess)
{
  auto tp = make_test_process ();
  ASSERT_TRUE (tp.process->from_core ());
  /* CORE_STATE returns a blob pointer; the actual size/type doesn't
     matter because the call throws before touching the buffer.  */
  void *out = nullptr;
  try
    {
      tp.process->get_info (AMD_DBGAPI_PROCESS_INFO_CORE_STATE, sizeof (out),
                            &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
    }
}

/* Attached (os_pid present) but not frozen: CORE_STATE must report
   PROCESS_NOT_FROZEN.  */
TEST (ProcessLifecycle, GetInfoCoreStateThrowsNotFrozenWhenAttachedButNotFrozen)
{
  auto tp = make_test_process (amd_dbgapi_process_id_t{ 11 },
                               amd_dbgapi_os_process_id_t{ 9999 });
  ASSERT_FALSE (tp.process->from_core ());
  ASSERT_FALSE (tp.process->is_frozen ());
  void *out = nullptr;
  try
    {
      tp.process->get_info (AMD_DBGAPI_PROCESS_INFO_CORE_STATE, sizeof (out),
                            &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_PROCESS_NOT_FROZEN);
    }
}

/* process_t::unfreeze() (internal seam, not the public C API) is a
   pure flag clear with no validation: calling it when never frozen
   must be a quiet no-op, not a throw or crash.  The public-API
   surface that requires is_frozen() is exercised in self_attach.  */
TEST (ProcessLifecycle, UnfreezeOnNeverFrozenIsNoOp)
{
  auto tp = make_test_process ();
  ASSERT_FALSE (tp.process->is_frozen ());
  EXPECT_NO_THROW (tp.process->unfreeze ());
  EXPECT_FALSE (tp.process->is_frozen ());
  /* Idempotent.  */
  EXPECT_NO_THROW (tp.process->unfreeze ());
  EXPECT_FALSE (tp.process->is_frozen ());
}
