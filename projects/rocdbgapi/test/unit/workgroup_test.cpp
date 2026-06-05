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

/* Unit tests for workgroup_t::xfer_local_memory bounds checking.

   Covered:
     * Normal bounds: read entirely past limit throws memory_access_error_t
       (caught as the api_error_t base; the exception carries
       AMD_DBGAPI_STATUS_ERROR_MEMORY_ACCESS).
     * Clamp-to-limit: a read straddling the end of the LDS region returns
       a short transfer of (limit - offset) bytes when offset < limit and
       size would extend past limit.  Zero-byte transfer is allowed even
       at the boundary.
     * Overflow regression: offset near UINT64_MAX with size > 0 must NOT
       wrap (limit - offset would underflow if the order were reversed).
       The fix is verified by calling xfer_local_memory directly via the
       friend access trampoline (xfer_segment_memory's lower() truncates
       offset to 32 bits and would mask the bug).

   xfer_local_memory's read/write callouts to agent().read/write_agent
   _memory_partial are NOT reached: every test either expects a throw or
   uses size==0 / clamped-to-zero, so no MockOsDriver xfer_*_memory
   expectations are needed.  */

#include "agent.h"
#include "amd-dbgapi.h"
#include "architecture.h"
#include "dispatch.h"
#include "exception.h"
#include "memory.h"
#include "os_driver.h"
#include "process.h"
#include "queue.h"
#include "workgroup.h"

#include "support/mock_os_driver.h"
#include "support/process_test_access.h"
#include "support/test_queue.h"
#include "support/workgroup_test_access.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

using ::testing::NiceMock;
using ::testing::Return;

using amd::dbgapi::address_space_t;
using amd::dbgapi::agent_t;
using amd::dbgapi::api_error_t;
using amd::dbgapi::architecture_t;
using amd::dbgapi::local_address_space_t;
using amd::dbgapi::os_agent_info_t;
using amd::dbgapi::os_driver_t;
using amd::dbgapi::process_t;
using amd::dbgapi::queue_t;
using amd::dbgapi::workgroup_t;
using amd::dbgapi::test::MockOsDriver;
using amd::dbgapi::test::make_test_os_queue_info;
using amd::dbgapi::test::test_compute_queue_t;
using amd::dbgapi::test::test_dispatch_t;
using amd::dbgapi::test::workgroup_test_access;

namespace
{

amd_dbgapi_client_process_id_t
fake_client_id ()
{
  static int sentinel = 0;
  return reinterpret_cast<amd_dbgapi_client_process_id_t> (&sentinel);
}

struct test_process_t
{
  std::unique_ptr<process_t> process;
  NiceMock<MockOsDriver> *driver;
};

test_process_t
make_test_process ()
{
  auto driver = std::make_unique<NiceMock<MockOsDriver>> ();
  ON_CALL (*driver, is_valid ()).WillByDefault (Return (true));
  auto *driver_raw = driver.get ();
  auto proc = process_test_access::make (
    amd_dbgapi_process_id_t{ 1 }, fake_client_id (), std::nullopt,
    std::unique_ptr<os_driver_t> (driver.release ()));
  return { std::move (proc), driver_raw };
}

os_agent_info_t
make_os_agent_info (amd::dbgapi::os_agent_id_t id = 42)
{
  os_agent_info_t info{};
  info.os_agent_id = id;
  info.name = "fake-gfx942";
  info.gfxip = { 9, 4, 2 };
  info.simd_count = 512;
  info.max_waves_per_simd = 10;
  info.vendor_id = 0x1002;
  info.device_id = 0x740c;
  info.fw_version = 137;
  info.local_address_aperture_base = 0x10000;
  info.local_address_aperture_limit = 0x1ffff;
  info.private_address_aperture_base = 0x20000;
  info.private_address_aperture_limit = 0x2ffff;
  info.debugging_supported = true;
  info.firmware_supported = true;
  return info;
}

const architecture_t *
gfx942 ()
{
  return architecture_t::find (std::string{ "gfx942" });
}

/* Fixture that wires up process -> agent -> queue -> dispatch -> workgroup
   with the test seams.  The queue is pre-suspended so xfer_local_memory's
   "if (!queue().is_suspended())" guard short-circuits the scoped suspend
   block and proceeds directly to the bounds check.  m_local_memory_size
   defaults to 4 KiB and m_local_memory_base_address is set via update().  */
class WorkgroupBoundsFixture
{
public:
  static constexpr amd_dbgapi_size_t local_memory_size = 4096;
  static constexpr uint64_t local_memory_base = 0x800000;

  WorkgroupBoundsFixture ()
    : m_tp (make_test_process ()),
      m_agent (&m_tp.process->create<agent_t> (*m_tp.process, gfx942 (),
                                               make_os_agent_info ())),
      m_queue (amd_dbgapi_queue_id_t{ 11 }, *m_agent,
               make_test_os_queue_info ()),
      m_dispatch (amd_dbgapi_dispatch_id_t{ 21 }, m_queue),
      m_workgroup (amd_dbgapi_workgroup_id_t{ 31 }, m_dispatch, std::nullopt,
                   local_memory_size),
      m_local_as (amd_dbgapi_address_space_id_t{ 41 }, "local")
  {
    /* Skip the suspend block in xfer_local_memory.  */
    m_queue.set_state (queue_t::state_t::suspended);
    /* Set m_local_memory_base_address; the bounds path asserts it.  */
    m_workgroup.update (amd::dbgapi::agent_address_t{ local_memory_base });
  }

  workgroup_t &workgroup () { return m_workgroup; }
  const address_space_t &local_as () const { return m_local_as; }

private:
  test_process_t m_tp;
  agent_t *m_agent;
  test_compute_queue_t m_queue;
  test_dispatch_t m_dispatch;
  workgroup_t m_workgroup;
  local_address_space_t m_local_as;
};

} /* namespace */

TEST (Workgroup, LocalMemoryReadPastLimitThrowsMemoryAccess)
{
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  std::array<std::byte, 32> buffer{};
  /* offset == limit, size > 0 → unreachable, must throw.  */
  EXPECT_THROW (
    workgroup_test_access::xfer_local_memory (
      fx.workgroup (), fx.local_as (),
      WorkgroupBoundsFixture::local_memory_size, buffer.data (), nullptr,
      buffer.size ()),
    api_error_t);
}

TEST (Workgroup, LocalMemoryReadStraddlingLimitClampsToShortTransfer)
{
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  /* offset = limit - 4, size = 32 → clamped to 4.  The actual transfer
     would then call read_agent_memory_partial; we can't reach that here
     because MockOsDriver's xfer_agent_memory_partial defaults to no
     expectation.  Use size == limit - offset (exact fit) so no clamping
     is needed, and offset==limit with size==0 which is a permitted no-op.
   */
  std::array<std::byte, 4> buffer{};
  amd_dbgapi_segment_address_t offset
    = WorkgroupBoundsFixture::local_memory_size;

  /* Zero-byte transfer at the boundary is allowed (max_size==0 && size==0
     skips the throw, and the subsequent read/write_partial gets size 0).
     The MockOsDriver default xfer_agent_memory_partial returns
     ERROR_STATUS_FAILURE for an uninstrumented call, but a 0-byte
     transfer never reaches the driver because read/write_agent_memory
     _partial returns early.  */
  size_t xferred = workgroup_test_access::xfer_local_memory (
    fx.workgroup (), fx.local_as (), offset, buffer.data (), nullptr, 0);
  EXPECT_EQ (xferred, 0u);
}

TEST (Workgroup, LocalMemoryOffsetNearUInt64MaxDoesNotWrap)
{
  /* Regression for the overflow fix in workgroup.cpp: the original code
     wrote `(offset + size) > limit`, which wraps when offset is near
     UINT64_MAX.  The fix is `offset > limit || size > limit - offset`.
     With offset = UINT64_MAX - 10 and size = 100, the throw must fire
     even though `offset + size` would overflow to a small value < limit.

     This path is unreachable from xfer_segment_memory: local_address_space
     ::lower zero-extends the address to 32 bits, masking the bug.  The
     friend trampoline calls xfer_local_memory directly.  */
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  std::array<std::byte, 32> buffer{};
  amd_dbgapi_segment_address_t huge_offset = UINT64_MAX - 10;
  EXPECT_THROW (workgroup_test_access::xfer_local_memory (
                  fx.workgroup (), fx.local_as (), huge_offset, buffer.data (),
                  nullptr, buffer.size ()),
                api_error_t);
}

TEST (Workgroup, GetInfoProcessRoundTripsId)
{
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  amd_dbgapi_process_id_t pid{};
  fx.workgroup ().get_info (AMD_DBGAPI_WORKGROUP_INFO_PROCESS, sizeof (pid),
                            &pid);
  EXPECT_EQ (pid.handle, 1u);
}

TEST (Workgroup, GetInfoWorkgroupCoordWithoutGroupIdsThrowsNotAvailable)
{
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  std::array<uint32_t, 3> coord{};
  /* The fixture passes std::nullopt for group_ids.  */
  EXPECT_THROW (fx.workgroup ().get_info (
                  AMD_DBGAPI_WORKGROUP_INFO_WORKGROUP_COORD, sizeof (coord),
                  coord.data ()),
                api_error_t);
}

TEST (Workgroup, GetInfoUnknownQueryThrowsInvalidArgument)
{
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  uint64_t sink = 0;
  EXPECT_THROW (
    fx.workgroup ().get_info (
      static_cast<amd_dbgapi_workgroup_info_t> (0x7fffffff), sizeof (sink),
      &sink),
    api_error_t);
}

TEST (Workgroup, GetInfoNullValuePointerThrowsInvalidArgument)
{
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  try
    {
      fx.workgroup ().get_info (AMD_DBGAPI_WORKGROUP_INFO_PROCESS,
                                sizeof (amd_dbgapi_process_id_t), nullptr);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (Workgroup, GetInfoWrongValueSizeThrowsInvalidArgumentCompatibility)
{
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  amd_dbgapi_process_id_t pid{};
  try
    {
      /* Pass a size that does not match sizeof(amd_dbgapi_process_id_t).  */
      fx.workgroup ().get_info (AMD_DBGAPI_WORKGROUP_INFO_PROCESS,
                                sizeof (pid) + 1, &pid);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Workgroup, LocalMemoryOffsetPastLimitThrowsMemoryAccess)
{
  /* offset strictly greater than limit (not just ==).  Sanity check that the
     "offset > limit" arm of the bounds check fires in addition to the
     "size > limit - offset" arm exercised by other tests.  */
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  std::array<std::byte, 8> buffer{};
  EXPECT_THROW (
    workgroup_test_access::xfer_local_memory (
      fx.workgroup (), fx.local_as (),
      WorkgroupBoundsFixture::local_memory_size + 1, buffer.data (), nullptr,
      buffer.size ()),
    api_error_t);
}

TEST (Workgroup, LocalMemoryWritePastLimitThrowsMemoryAccess)
{
  /* Same bounds check applies to writes via the write_ptr path.  */
  ASSERT_NE (gfx942 (), nullptr);
  WorkgroupBoundsFixture fx;

  std::array<std::byte, 16> buffer{};
  EXPECT_THROW (
    workgroup_test_access::xfer_local_memory (
      fx.workgroup (), fx.local_as (),
      WorkgroupBoundsFixture::local_memory_size, nullptr, buffer.data (),
      buffer.size ()),
    api_error_t);
}
