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

/* Unit tests for agent_t surfaces that do NOT require queue_t / wave_t or
   amd_dbgapi_initialize() callbacks.

   Covered:
     * Construction with a static-registered architecture (gfx942) and with
       a nullptr "dummy" architecture (the AGENT_NONE path).
     * os_agent_id() / os_info() round-trip.
     * supports_debugging(): true only when architecture != nullptr AND
       debugging_supported AND firmware_supported.
     * spi_ttmps_setup_enabled(): true if the process flag is set OR
       ttmps_always_initialized is set on the agent.
     * watchpoint_share_kind(): UNSUPPORTED / UNSHARED / SHARED selection.
     * set_exceptions / clear_exceptions / exceptions bookkeeping.
     * mark() / set_mark() round-trip.
     * apertures(): populated for a real architecture (3 apertures on gfx9),
       empty for the dummy agent.
     * get_info() for POD queries (OS_ID, PCI_DOMAIN, STATE, PROCESS,
       ARCHITECTURE) and the NOT_AVAILABLE branch when the dummy agent is
       queried for ARCHITECTURE.

   NOT covered here (deferred):
     * insert_watchpoint / remove_watchpoint — exercised in the watchpoint
       commit, where watchpoint_t can be constructed.
     * memory_cache xfer paths — exercised via xfer_agent_memory in a later
       memory-API commit.
     * get_info(NAME) — the std::string specialization routes through the
       dbgapi allocator callback, which is only wired during
       amd_dbgapi_initialize().  */

#include "amd-dbgapi.h"
#include "agent.h"
#include "architecture.h"
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

using ::testing::NiceMock;
using ::testing::Return;

using amd::dbgapi::agent_t;
using amd::dbgapi::api_error_t;
using amd::dbgapi::architecture_t;
using amd::dbgapi::os_agent_info_t;
using amd::dbgapi::os_driver_t;
using amd::dbgapi::os_exception_mask_t;
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

/* A representative populated os_agent_info_t.  Field values are chosen to
   be distinguishable so accessors can round-trip them.  */
os_agent_info_t
make_os_agent_info (amd::dbgapi::os_agent_id_t id = 42)
{
  os_agent_info_t info{};
  info.os_agent_id = id;
  info.name = "fake-gfx942";
  info.gfxip = { 9, 4, 2 };
  info.domain = 0x1234;
  info.location_id = 0x5678;
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
  info.address_watch_supported = true;
  info.address_watch_register_count = 4;
  info.watchpoint_exclusive = false;
  info.ttmps_always_initialized = false;
  return info;
}

const architecture_t *
gfx942 ()
{
  return architecture_t::find (std::string{ "gfx942" });
}

} /* namespace */

TEST (Agent, ConstructsWithRealArchitectureAndExposesOsInfo)
{
  auto tp = make_test_process ();
  ASSERT_NE (gfx942 (), nullptr);

  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  EXPECT_EQ (agent.os_agent_id (), 42u);
  EXPECT_EQ (agent.architecture (), gfx942 ());
  EXPECT_EQ (&agent.process (), tp.process.get ());

  /* os_info() should round-trip the supplied fields.  */
  EXPECT_EQ (agent.os_info ().domain, 0x1234);
  EXPECT_EQ (agent.os_info ().location_id, 0x5678);
  EXPECT_EQ (agent.os_info ().simd_count, 512u);
  EXPECT_EQ (agent.os_info ().fw_version, 137u);
}

TEST (Agent, DummyAgentHasNoArchitectureAndNoApertures)
{
  /* The AGENT_NONE constructor path skips firmware/architecture setup.
     m_apertures is empty and m_watchpoints is not resized.  process_t
     stores its dummy as a direct member, not in the handle set, so we
     construct on the stack to mirror that contract.  */
  auto tp = make_test_process ();
  os_agent_info_t info{};
  agent_t agent (AMD_DBGAPI_AGENT_NONE, *tp.process, nullptr, info);

  EXPECT_EQ (agent.architecture (), nullptr);
  EXPECT_TRUE (agent.apertures ().empty ());
  EXPECT_FALSE (agent.supports_debugging ());
}

TEST (Agent, AperturesPopulatedForGfx9)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  /* gfx9 architectures publish three apertures: local, private_lane,
     global (see architecture.cpp gfx9_architecture_t::get_apertures).  */
  EXPECT_EQ (agent.apertures ().size (), 3u);

  /* The first two apertures' bases mirror the supplied os_agent_info.  */
  EXPECT_EQ (agent.apertures ()[0].base, 0x10000u);
  EXPECT_EQ (agent.apertures ()[0].limit, 0x1ffffu);
  EXPECT_EQ (agent.apertures ()[1].base, 0x20000u);
  EXPECT_EQ (agent.apertures ()[1].limit, 0x2ffffu);
}

TEST (Agent, SupportsDebuggingRequiresAllThree)
{
  auto tp = make_test_process ();

  auto good = make_os_agent_info (1);
  auto &agent_good
    = tp.process->create<agent_t> (*tp.process, gfx942 (), good);
  EXPECT_TRUE (agent_good.supports_debugging ());

  auto no_fw = make_os_agent_info (2);
  no_fw.firmware_supported = false;
  auto &agent_no_fw
    = tp.process->create<agent_t> (*tp.process, gfx942 (), no_fw);
  EXPECT_FALSE (agent_no_fw.supports_debugging ());

  auto no_dbg = make_os_agent_info (3);
  no_dbg.debugging_supported = false;
  auto &agent_no_dbg
    = tp.process->create<agent_t> (*tp.process, gfx942 (), no_dbg);
  EXPECT_FALSE (agent_no_dbg.supports_debugging ());

  auto no_arch = make_os_agent_info (4);
  agent_t agent_no_arch (AMD_DBGAPI_AGENT_NONE, *tp.process, nullptr, no_arch);
  EXPECT_FALSE (agent_no_arch.supports_debugging ());
}

TEST (Agent, SpiTtmpsSetupEnabledTracksProcessFlagOrAgentFlag)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  /* ttmps_always_initialized=false (default in make_os_agent_info).  */
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  EXPECT_FALSE (agent.spi_ttmps_setup_enabled ());

  tp.process->set_flag (process_t::flag_t::spi_ttmps_setup_enabled);
  EXPECT_TRUE (agent.spi_ttmps_setup_enabled ());

  tp.process->clear_flag (process_t::flag_t::spi_ttmps_setup_enabled);
  EXPECT_FALSE (agent.spi_ttmps_setup_enabled ());

  /* A second agent whose firmware reports ttmps as always initialized
     should report true regardless of the process flag.  */
  auto info2 = make_os_agent_info (99);
  info2.ttmps_always_initialized = true;
  auto &agent2 = tp.process->create<agent_t> (*tp.process, gfx942 (), info2);
  EXPECT_TRUE (agent2.spi_ttmps_setup_enabled ());
}

TEST (Agent, WatchpointShareKindReflectsOsInfo)
{
  auto tp = make_test_process ();

  auto info_unsupported = make_os_agent_info (1);
  info_unsupported.address_watch_supported = false;
  auto &agent_unsupported = tp.process->create<agent_t> (
    *tp.process, gfx942 (), info_unsupported);
  EXPECT_EQ (agent_unsupported.watchpoint_share_kind (),
             AMD_DBGAPI_WATCHPOINT_SHARE_KIND_UNSUPPORTED);

  auto info_unshared = make_os_agent_info (2);
  info_unshared.address_watch_supported = true;
  info_unshared.watchpoint_exclusive = true;
  auto &agent_unshared
    = tp.process->create<agent_t> (*tp.process, gfx942 (), info_unshared);
  EXPECT_EQ (agent_unshared.watchpoint_share_kind (),
             AMD_DBGAPI_WATCHPOINT_SHARE_KIND_UNSHARED);

  auto info_shared = make_os_agent_info (3);
  info_shared.address_watch_supported = true;
  info_shared.watchpoint_exclusive = false;
  auto &agent_shared
    = tp.process->create<agent_t> (*tp.process, gfx942 (), info_shared);
  EXPECT_EQ (agent_shared.watchpoint_share_kind (),
             AMD_DBGAPI_WATCHPOINT_SHARE_KIND_SHARED);
}

TEST (Agent, ExceptionsAccumulateAndClear)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  EXPECT_EQ (agent.exceptions (), os_exception_mask_t::none);

  agent.set_exceptions (os_exception_mask_t::device_memory_violation);
  EXPECT_EQ (agent.exceptions (), os_exception_mask_t::device_memory_violation);

  /* set_exceptions ORs in additional bits.  */
  agent.set_exceptions (os_exception_mask_t::device_ras_error);
  EXPECT_EQ (agent.exceptions (), (os_exception_mask_t::device_memory_violation
                                   | os_exception_mask_t::device_ras_error));

  /* clear_exceptions removes only the named bits.  */
  agent.clear_exceptions (os_exception_mask_t::device_memory_violation);
  EXPECT_EQ (agent.exceptions (), os_exception_mask_t::device_ras_error);

  agent.clear_exceptions (os_exception_mask_t::device_ras_error);
  EXPECT_EQ (agent.exceptions (), os_exception_mask_t::none);
}

TEST (Agent, MarkRoundTrips)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  EXPECT_EQ (agent.mark (), 0u);
  agent.set_mark (123);
  EXPECT_EQ (agent.mark (), 123u);

  /* next_mark() yields strictly increasing values.  */
  auto a = agent_t::next_mark ();
  auto b = agent_t::next_mark ();
  EXPECT_LT (a, b);
}

TEST (Agent, GetInfoOsIdReturnsOsAgentId)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info (77);
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  amd_dbgapi_os_agent_id_t os_id{};
  agent.get_info (AMD_DBGAPI_AGENT_INFO_OS_ID, sizeof (os_id), &os_id);
  EXPECT_EQ (os_id, 77u);
}

TEST (Agent, GetInfoPciDomainReturnsDomain)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  uint16_t domain{};
  agent.get_info (AMD_DBGAPI_AGENT_INFO_PCI_DOMAIN, sizeof (domain), &domain);
  EXPECT_EQ (domain, 0x1234);
}

TEST (Agent, GetInfoStateReflectsSupportsDebugging)
{
  auto tp = make_test_process ();
  auto good = make_os_agent_info (1);
  auto &agent_good
    = tp.process->create<agent_t> (*tp.process, gfx942 (), good);

  amd_dbgapi_agent_state_t state{};
  agent_good.get_info (AMD_DBGAPI_AGENT_INFO_STATE, sizeof (state), &state);
  EXPECT_EQ (state, AMD_DBGAPI_AGENT_STATE_SUPPORTED);

  auto bad = make_os_agent_info (2);
  bad.firmware_supported = false;
  auto &agent_bad = tp.process->create<agent_t> (*tp.process, gfx942 (), bad);
  agent_bad.get_info (AMD_DBGAPI_AGENT_INFO_STATE, sizeof (state), &state);
  EXPECT_EQ (state, AMD_DBGAPI_AGENT_STATE_NOT_SUPPORTED);
}

TEST (Agent, GetInfoProcessReturnsProcessId)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  amd_dbgapi_process_id_t pid{};
  agent.get_info (AMD_DBGAPI_AGENT_INFO_PROCESS, sizeof (pid), &pid);
  EXPECT_EQ (pid.handle, tp.process->id ().handle);
}

TEST (Agent, GetInfoArchitectureReturnsArchitectureId)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  amd_dbgapi_architecture_id_t arch_id{};
  agent.get_info (AMD_DBGAPI_AGENT_INFO_ARCHITECTURE, sizeof (arch_id),
                  &arch_id);
  EXPECT_EQ (arch_id.handle, gfx942 ()->id ().handle);
}

TEST (Agent, GetInfoArchitectureThrowsNotAvailableForDummyAgent)
{
  auto tp = make_test_process ();
  os_agent_info_t info{};
  agent_t agent (AMD_DBGAPI_AGENT_NONE, *tp.process, nullptr, info);

  amd_dbgapi_architecture_id_t arch_id{};
  try
    {
      agent.get_info (AMD_DBGAPI_AGENT_INFO_ARCHITECTURE, sizeof (arch_id),
                      &arch_id);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE);
    }
}

TEST (Agent, GetInfoWrongSizeThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  uint8_t too_small{};
  try
    {
      agent.get_info (AMD_DBGAPI_AGENT_INFO_OS_ID, sizeof (too_small),
                      &too_small);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Agent, GetInfoNullValueThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  try
    {
      agent.get_info (AMD_DBGAPI_AGENT_INFO_OS_ID,
                      sizeof (amd_dbgapi_os_agent_id_t), nullptr);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (Agent, GetInfoUnknownQueryThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  uint64_t dummy{};
  try
    {
      agent.get_info (static_cast<amd_dbgapi_agent_info_t> (0xdead),
                      sizeof (dummy), &dummy);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* C25 negative-test sweep: value_size and enum edge cases.                */

TEST (Agent, GetInfoValueSizeZeroThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  amd_dbgapi_os_agent_id_t out{};
  try
    {
      agent.get_info (AMD_DBGAPI_AGENT_INFO_OS_ID, 0, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Agent, GetInfoValueSizeMaxThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  amd_dbgapi_os_agent_id_t out{};
  try
    {
      agent.get_info (AMD_DBGAPI_AGENT_INFO_OS_ID, SIZE_MAX, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Agent, GetInfoValueSizeTooLargeByOneThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  /* sizeof(amd_dbgapi_os_agent_id_t) + 1 buffer; pass +1 size.  */
  uint8_t out[sizeof (amd_dbgapi_os_agent_id_t) + 1]{};
  try
    {
      agent.get_info (AMD_DBGAPI_AGENT_INFO_OS_ID,
                      sizeof (amd_dbgapi_os_agent_id_t) + 1, out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Agent, GetInfoNegativeEnumThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  uint64_t out{};
  try
    {
      agent.get_info (static_cast<amd_dbgapi_agent_info_t> (-1),
                      sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (Agent, GetInfoIntMaxEnumThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto info = make_os_agent_info ();
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  uint64_t out{};
  try
    {
      agent.get_info (static_cast<amd_dbgapi_agent_info_t> (INT_MAX),
                      sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}
