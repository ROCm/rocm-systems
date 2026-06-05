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

/* Unit tests for the AQL dispatch packet decoding in
   detail::aql_queue_t::aql_dispatch_t::get_info.  The class is private to
   queue.cpp; access is via test::make_aql_queue_and_dispatch_for_test
   (see support/aql_dispatch_test_access.h), which constructs the queue +
   dispatch pair via process->create<> so destruction goes back through
   process.destroy().

   Backing-store setup: MockOsDriver serves an hsa_kernel_dispatch_packet_t
   at the queue's ring base for xfer_host_memory_partial, and a 64-byte
   kernel_descriptor blob at packet.kernel_object for
   xfer_global_memory_partial.  Both mocks honor partial reads (returning
   *size unchanged, which the caller treats as a full transfer for the
   requested chunk).  */

#include "agent.h"
#include "amd-dbgapi.h"
#include "architecture.h"
#include "dispatch.h"
#include "exception.h"
#include "hsa/hsa.h"
#include "os_driver.h"
#include "process.h"
#include "queue.h"

#include "support/aql_dispatch_test_access.h"
#include "support/mock_os_driver.h"
#include "support/process_test_access.h"
#include "support/test_queue.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

using ::testing::_;
using ::testing::DoAll;
using ::testing::Invoke;
using ::testing::NiceMock;
using ::testing::Return;

using amd::dbgapi::agent_t;
using amd::dbgapi::api_error_t;
using amd::dbgapi::architecture_t;
using amd::dbgapi::dispatch_t;
using amd::dbgapi::global_address_t;
using amd::dbgapi::host_address_t;
using amd::dbgapi::os_agent_info_t;
using amd::dbgapi::os_driver_t;
using amd::dbgapi::process_t;
using amd::dbgapi::queue_t;
using amd::dbgapi::test::make_aql_queue_and_dispatch_for_test;
using amd::dbgapi::test::make_test_os_queue_info;
using amd::dbgapi::test::MockOsDriver;

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

/* 64-byte hsa_kernel_dispatch_packet_t with field values the tests assert.
   header: TYPE=KERNEL_DISPATCH(2)@bit0w8, BARRIER=1@bit8,
           SCACQUIRE=SYSTEM(2)@bit9w2, SCRELEASE=AGENT(1)@bit11w2.
   setup: DIMENSIONS=3@bit0w2.  */
constexpr uint64_t kKernelObject = 0x80000ull;
constexpr uint64_t kKernargAddress = 0x90000ull;
constexpr uint64_t kCompletionSignal = 0xa0000ull;
constexpr uint16_t kWgX = 8, kWgY = 4, kWgZ = 2;
constexpr uint32_t kGridX = 64, kGridY = 16, kGridZ = 4;
constexpr uint32_t kPrivSize = 128, kGroupSize = 256;

hsa_kernel_dispatch_packet_t
make_dispatch_packet ()
{
  hsa_kernel_dispatch_packet_t p{};
  uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE)
                    | (1u << HSA_PACKET_HEADER_BARRIER)
                    | (HSA_FENCE_SCOPE_SYSTEM
                       << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
                    | (HSA_FENCE_SCOPE_AGENT
                       << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
  uint16_t setup = (3u << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS);
  p.header = header;
  p.setup = setup;
  p.workgroup_size_x = kWgX;
  p.workgroup_size_y = kWgY;
  p.workgroup_size_z = kWgZ;
  p.grid_size_x = kGridX;
  p.grid_size_y = kGridY;
  p.grid_size_z = kGridZ;
  p.private_segment_size = kPrivSize;
  p.group_segment_size = kGroupSize;
  p.kernel_object = kKernelObject;
  p.kernarg_address = reinterpret_cast<void *> (kKernargAddress);
  p.completion_signal = hsa_signal_t{ kCompletionSignal };
  return p;
}

/* 64-byte amdgcn_architecture_t::kernel_descriptor_t backing layout.
   kernel_code_entry_byte_offset is at offset 16 (after the two u32 +
   reserved0[8]) and we set it to 0x100 so entry_address () returns
   kKernelObject + 0x100.  */
constexpr int64_t kEntryByteOffset = 0x100;

std::array<std::byte, 64>
make_kernel_descriptor_bytes ()
{
  std::array<std::byte, 64> bytes{};
  std::memcpy (bytes.data () + 16, &kEntryByteOffset,
               sizeof (kEntryByteOffset));
  return bytes;
}

/* Install mock expectations: any host-memory read at the queue's ring base
   serves packet bytes; any global-memory read at kKernelObject serves
   kernel-descriptor bytes.  Both honor partial transfers (memcpy *size
   bytes from the appropriate offset).  */
void
install_aql_backing_store (MockOsDriver &drv,
                           const hsa_kernel_dispatch_packet_t &packet,
                           const std::array<std::byte, 64> &kd_bytes,
                           host_address_t ring_base)
{
  ON_CALL (drv, xfer_host_memory_partial (_, _, _, _))
    .WillByDefault (
      Invoke ([packet, ring_base] (host_address_t address, void *read,
                                   const void * /*write*/, size_t *size)
              {
                if (read == nullptr || address < ring_base
                    || address + *size > ring_base + sizeof (packet))
                  return AMD_DBGAPI_STATUS_ERROR;
                std::memcpy (read,
                             reinterpret_cast<const std::byte *> (&packet)
                               + (address - ring_base),
                             *size);
                return AMD_DBGAPI_STATUS_SUCCESS;
              }));

  ON_CALL (drv, xfer_global_memory_partial (_, _, _, _))
    .WillByDefault (
      Invoke ([kd_bytes] (global_address_t address, void *read,
                          const void * /*write*/, size_t *size)
              {
                if (read == nullptr || address < kKernelObject
                    || address + *size > kKernelObject + kd_bytes.size ())
                  return AMD_DBGAPI_STATUS_ERROR;
                std::memcpy (read,
                             kd_bytes.data () + (address - kKernelObject),
                             *size);
                return AMD_DBGAPI_STATUS_SUCCESS;
              }));
}

/* Fixture stands up process -> agent -> aql_queue_t -> aql_dispatch_t with
   the backing-store mocks pre-installed.  The pair is auto-id'd by
   handle_object_set_t; tests address it via fx.dispatch ().  Destruction
   order in the dtor is dispatch first, then queue, matching
   ~aql_queue_t's sweep-on-destroy invariant.  */
class AqlDispatchFixture
{
public:
  static constexpr amd_dbgapi_os_queue_packet_id_t kPacketId = 0;

  AqlDispatchFixture ()
    : m_tp (make_test_process ()),
      m_agent (&m_tp.process->create<agent_t> (*m_tp.process, gfx942 (),
                                               make_os_agent_info ())),
      m_packet (make_dispatch_packet ()),
      m_kd_bytes (make_kernel_descriptor_bytes ()),
      m_os_queue_info (make_test_os_queue_info ())
  {
    install_aql_backing_store (
      *m_tp.driver, m_packet, m_kd_bytes,
      std::get<host_address_t> (m_os_queue_info.ring_base_address));

    auto [q, d] = make_aql_queue_and_dispatch_for_test (
      *m_tp.process, *m_agent, m_os_queue_info, kPacketId);
    m_queue = q;
    m_dispatch = d;
  }

  ~AqlDispatchFixture ()
  {
    m_tp.process->destroy (m_dispatch);
    m_tp.process->destroy (m_queue);
  }

  process_t &process () { return *m_tp.process; }
  agent_t &agent () { return *m_agent; }
  queue_t &queue () { return *m_queue; }
  dispatch_t &dispatch () { return *m_dispatch; }

private:
  test_process_t m_tp;
  agent_t *m_agent;
  hsa_kernel_dispatch_packet_t m_packet;
  std::array<std::byte, 64> m_kd_bytes;
  amd::dbgapi::os_queue_snapshot_entry_t m_os_queue_info;
  queue_t *m_queue{};
  dispatch_t *m_dispatch{};
};

} /* namespace */

TEST (AqlDispatch, ConstructsAndExposesBacklinks)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  amd_dbgapi_queue_id_t qid{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_QUEUE, sizeof (qid), &qid);
  EXPECT_EQ (qid.handle, fx.queue ().id ().handle);

  amd_dbgapi_agent_id_t aid{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_AGENT, sizeof (aid), &aid);
  EXPECT_EQ (aid.handle, fx.agent ().id ().handle);

  amd_dbgapi_process_id_t pid{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_PROCESS, sizeof (pid),
                           &pid);
  EXPECT_EQ (pid.handle, 1u);

  amd_dbgapi_architecture_id_t arch{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_ARCHITECTURE,
                           sizeof (arch), &arch);
  EXPECT_EQ (arch.handle, gfx942 ()->id ().handle);

  amd_dbgapi_os_queue_packet_id_t pkid{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_OS_QUEUE_PACKET_ID,
                           sizeof (pkid), &pkid);
  EXPECT_EQ (pkid, AqlDispatchFixture::kPacketId);
}

TEST (AqlDispatch, GetInfoBarrierBitDecodes)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  amd_dbgapi_dispatch_barrier_t barrier{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_BARRIER, sizeof (barrier),
                           &barrier);
  EXPECT_EQ (barrier, AMD_DBGAPI_DISPATCH_BARRIER_PRESENT);
}

TEST (AqlDispatch, GetInfoAcquireAndReleaseFenceScopesDecode)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  amd_dbgapi_dispatch_fence_scope_t acq{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_ACQUIRE_FENCE, sizeof (acq),
                           &acq);
  EXPECT_EQ (acq, AMD_DBGAPI_DISPATCH_FENCE_SCOPE_SYSTEM);

  amd_dbgapi_dispatch_fence_scope_t rel{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_RELEASE_FENCE, sizeof (rel),
                           &rel);
  EXPECT_EQ (rel, AMD_DBGAPI_DISPATCH_FENCE_SCOPE_AGENT);
}

TEST (AqlDispatch, GetInfoGridDimensionsDecodesFromSetup)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  uint32_t dims = 0;
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_GRID_DIMENSIONS,
                           sizeof (dims), &dims);
  EXPECT_EQ (dims, 3u);
}

TEST (AqlDispatch, GetInfoWorkgroupAndGridSizesDecode)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  std::array<uint16_t, 3> wg{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_WORKGROUP_SIZES,
                           sizeof (wg), wg.data ());
  EXPECT_EQ (wg[0], kWgX);
  EXPECT_EQ (wg[1], kWgY);
  EXPECT_EQ (wg[2], kWgZ);

  std::array<uint32_t, 3> grid{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_GRID_SIZES, sizeof (grid),
                           grid.data ());
  EXPECT_EQ (grid[0], kGridX);
  EXPECT_EQ (grid[1], kGridY);
  EXPECT_EQ (grid[2], kGridZ);
}

TEST (AqlDispatch, GetInfoSegmentSizesAndAddressesDecode)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  amd_dbgapi_size_t priv = 0;
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_PRIVATE_SEGMENT_SIZE,
                           sizeof (priv), &priv);
  EXPECT_EQ (priv, kPrivSize);

  amd_dbgapi_size_t grp = 0;
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_GROUP_SEGMENT_SIZE,
                           sizeof (grp), &grp);
  EXPECT_EQ (grp, kGroupSize);

  void *karg = nullptr;
  fx.dispatch ().get_info (
    AMD_DBGAPI_DISPATCH_INFO_KERNEL_ARGUMENT_SEGMENT_ADDRESS, sizeof (karg),
    &karg);
  EXPECT_EQ (reinterpret_cast<uint64_t> (karg), kKernargAddress);

  hsa_signal_t completion{};
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_KERNEL_COMPLETION_ADDRESS,
                           sizeof (completion), &completion);
  EXPECT_EQ (completion.handle, kCompletionSignal);
}

TEST (AqlDispatch, GetInfoKernelDescriptorAndEntryAddressesDecode)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  global_address_t kd_addr = 0;
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_KERNEL_DESCRIPTOR_ADDRESS,
                           sizeof (kd_addr), &kd_addr);
  EXPECT_EQ (kd_addr, kKernelObject);

  global_address_t entry_addr = 0;
  fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_KERNEL_CODE_ENTRY_ADDRESS,
                           sizeof (entry_addr), &entry_addr);
  EXPECT_EQ (entry_addr, kKernelObject + kEntryByteOffset);
}

TEST (AqlDispatch, GetInfoUnknownQueryThrowsInvalidArgument)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  uint64_t sink = 0;
  EXPECT_THROW (
    fx.dispatch ().get_info (
      static_cast<amd_dbgapi_dispatch_info_t> (0x7fffffff), sizeof (sink),
      &sink),
    api_error_t);
}

TEST (AqlDispatch, GetInfoNullValuePointerThrowsInvalidArgument)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  try
    {
      fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_PROCESS,
                               sizeof (amd_dbgapi_process_id_t), nullptr);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (AqlDispatch, GetInfoWrongValueSizeThrowsInvalidArgumentCompatibility)
{
  ASSERT_NE (gfx942 (), nullptr);
  AqlDispatchFixture fx;

  amd_dbgapi_process_id_t pid{};
  try
    {
      /* Pass a size that does not match sizeof(amd_dbgapi_process_id_t).  */
      fx.dispatch ().get_info (AMD_DBGAPI_DISPATCH_INFO_PROCESS,
                               sizeof (pid) + 1, &pid);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

/* NOT TESTED (would terminate the test process):
   - Malformed packet header (e.g. HSA_PACKET_TYPE_INVALID or any value other
     than KERNEL_DISPATCH / VENDOR_SPECIFIC) reaches the "Unsupported AQL
     packet type" branch in queue.cpp which calls fatal_error (noreturn).
   - MockOsDriver returning ERROR from xfer_host_memory_partial or
     xfer_global_memory_partial during dispatch construction.  process_t::
     read_host_memory / read_global_memory catch the memory_access_error_t
     and call fatal_error.  */
