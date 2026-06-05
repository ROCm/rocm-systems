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

/* Unit tests for watchpoint_t construction and the agent_t
   insert/remove_watchpoint bookkeeping that depends on it.

   Covered:
     * watchpoint_t ctor's mask-bits algorithm:
         - With no debugging-capable agents the mask is permissive
           (programmable_mask_bits stays at MAX), so requested address
           and size round-trip directly when already aligned.
         - When the requested address is unaligned, the stored address
           is aligned down to the size.
         - When the requested size isn't a power of two, the stored
           size is the next power of two.
     * watchpoint_t::get_info for the three POD queries.
     * agent_t::insert_watchpoint:
         - Translates the watchpoint kind to the matching os_watch_mode_t.
         - Calls os_driver_t::set_address_watch with (os_agent_id,
           address, alignment_to_mask(size), watch_mode, &os_watch_id).
         - Stores the watchpoint pointer at m_watchpoints[os_watch_id]
           so get_watchpoint() returns it.
         - Throws NO_WATCHPOINT_AVAILABLE when the driver reports it.
     * agent_t::remove_watchpoint:
         - Calls os_driver_t::clear_address_watch with the os_watch_id
           the agent recorded at insert time.
         - Returns silently when the watchpoint was never inserted on
           this agent (the find_if() == end() path).

   NOT covered here (deferred):
     * displaced_stepping_t — holds a queue_t& and needs a queue_t test
       seam that doesn't exist yet (queue_t is abstract and its concrete
       subclasses pull in dispatch/wave construction).  Will land once
       that seam exists.  */

#include "amd-dbgapi.h"
#include "agent.h"
#include "architecture.h"
#include "exception.h"
#include "os_driver.h"
#include "process.h"
#include "watchpoint.h"

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

using amd::dbgapi::agent_address_t;
using amd::dbgapi::agent_t;
using amd::dbgapi::api_error_t;
using amd::dbgapi::architecture_t;
using amd::dbgapi::os_agent_info_t;
using amd::dbgapi::os_driver_t;
using amd::dbgapi::os_watch_mode_t;
using amd::dbgapi::process_t;
using amd::dbgapi::watchpoint_t;
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

const architecture_t *
gfx942 ()
{
  return architecture_t::find (std::string{ "gfx942" });
}

/* Populated os_agent_info_t that supports debugging + address watch.
   Mask bits cover bits [6..39], i.e. the user-programmable mask range
   in the diagram in watchpoint.cpp.  4 hardware watch registers.  */
os_agent_info_t
make_debugging_agent_info (amd::dbgapi::os_agent_id_t id = 1)
{
  os_agent_info_t info{};
  info.os_agent_id = id;
  info.debugging_supported = true;
  info.firmware_supported = true;
  info.address_watch_supported = true;
  info.address_watch_register_count = 4;
  info.address_watch_mask_bits = 0xFFFFFFFFC0ull;
  return info;
}

} /* namespace */

/* ------------------------------------------------------------------ */
/* watchpoint_t ctor: mask-bits algorithm                              */
/* ------------------------------------------------------------------ */

TEST (Watchpoint, CtorRoundTripsAlignedAddressAndPowerOfTwoSize)
{
  /* No agents -> programmable_mask_bits stays at MAX, which lets the
     stable_bits mask pass through unchanged.  An aligned address and a
     power-of-two size round-trip to themselves.  */
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, /* requested_address */ 0x1000ull,
    /* requested_size */ 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);

  EXPECT_EQ (wp.requested_address (), 0x1000ull);
  EXPECT_EQ (wp.requested_size (), 0x40ull);
  EXPECT_EQ (wp.address (), 0x1000ull);
  EXPECT_EQ (wp.size (), 0x40ull);
  EXPECT_EQ (wp.kind (), AMD_DBGAPI_WATCHPOINT_KIND_LOAD);
}

TEST (Watchpoint, CtorAlignsAddressDownAndGrowsSizeToCoverSpan)
{
  /* Request a misaligned address: the stored range must cover
     [0x1005, 0x1045).  Aligning down to 0x1000 with size 0x40 would
     only cover [0x1000, 0x1040), missing the tail.  The ctor therefore
     grows size to the next power of two (0x80) so [0x1000, 0x1080)
     covers the requested span.  */
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, /* requested_address */ 0x1005ull,
    /* requested_size */ 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_RMW);

  EXPECT_EQ (wp.address (), 0x1000ull);
  EXPECT_EQ (wp.size (), 0x80ull);
}

TEST (Watchpoint, CtorRoundsSizeUpToNextPowerOfTwo)
{
  /* A non-power-of-two requested_size grows to the next power of two
     (the smallest range that covers the requested span).  */
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, /* requested_address */ 0x2000ull,
    /* requested_size */ 0x30ull, AMD_DBGAPI_WATCHPOINT_KIND_STORE_AND_RMW);

  EXPECT_EQ (wp.address (), 0x2000ull);
  EXPECT_EQ (wp.size (), 0x40ull);
}

TEST (Watchpoint, CtorWithDebuggingAgentRespectsAgentMaskBits)
{
  /* When at least one debugging-capable agent is present, the ctor ANDs
     in that agent's address_watch_mask_bits.  The bits we chose
     (0xFFFFFFFFC0) still permit the [0x1000, 0x103F] range exactly,
     so the result must match the no-agent case.  This verifies that
     the agent loop runs without disturbing a request that fits within
     the agent's capabilities.  */
  auto tp = make_test_process ();
  tp.process->create<agent_t> (*tp.process, gfx942 (),
                               make_debugging_agent_info ());

  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x1000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_ALL);

  EXPECT_EQ (wp.address (), 0x1000ull);
  EXPECT_EQ (wp.size (), 0x40ull);
}

/* ------------------------------------------------------------------ */
/* watchpoint_t::get_info                                              */
/* ------------------------------------------------------------------ */

TEST (Watchpoint, GetInfoReturnsAddressSizeAndProcess)
{
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x4000ull, 0x80ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);

  amd_dbgapi_global_address_t addr{};
  wp.get_info (AMD_DBGAPI_WATCHPOINT_INFO_ADDRESS, sizeof (addr), &addr);
  EXPECT_EQ (addr, 0x4000ull);

  amd_dbgapi_size_t size{};
  wp.get_info (AMD_DBGAPI_WATCHPOINT_INFO_SIZE, sizeof (size), &size);
  EXPECT_EQ (size, 0x80ull);

  amd_dbgapi_process_id_t pid{};
  wp.get_info (AMD_DBGAPI_WATCHPOINT_INFO_PROCESS, sizeof (pid), &pid);
  EXPECT_EQ (pid.handle, tp.process->id ().handle);
}

TEST (Watchpoint, GetInfoUnknownQueryThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x4000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);

  uint64_t dummy{};
  try
    {
      wp.get_info (static_cast<amd_dbgapi_watchpoint_info_t> (0xbeef),
                   sizeof (dummy), &dummy);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* C25 negative-test sweep: value_size and enum edge cases.              */

TEST (Watchpoint, GetInfoValueSizeZeroThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x4000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);
  amd_dbgapi_global_address_t out{};
  try
    {
      wp.get_info (AMD_DBGAPI_WATCHPOINT_INFO_ADDRESS, 0, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Watchpoint, GetInfoValueSizeMaxThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x4000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);
  amd_dbgapi_global_address_t out{};
  try
    {
      wp.get_info (AMD_DBGAPI_WATCHPOINT_INFO_ADDRESS, SIZE_MAX, &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Watchpoint, GetInfoValueSizeTooLargeByOneThrowsCompatibility)
{
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x4000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);
  uint8_t out[sizeof (amd_dbgapi_global_address_t) + 1]{};
  try
    {
      wp.get_info (AMD_DBGAPI_WATCHPOINT_INFO_ADDRESS,
                   sizeof (amd_dbgapi_global_address_t) + 1, out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }
}

TEST (Watchpoint, GetInfoNegativeEnumThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x4000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);
  amd_dbgapi_global_address_t out{};
  try
    {
      wp.get_info (static_cast<amd_dbgapi_watchpoint_info_t> (-1),
                   sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

TEST (Watchpoint, GetInfoIntMaxEnumThrowsInvalidArgument)
{
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x4000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);
  amd_dbgapi_global_address_t out{};
  try
    {
      wp.get_info (static_cast<amd_dbgapi_watchpoint_info_t> (INT_MAX),
                   sizeof (out), &out);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }
}

/* ------------------------------------------------------------------ */
/* C26 boundary cases: watchpoint_t ctor at the edges of size/address.  */
/* ------------------------------------------------------------------ */

TEST (Watchpoint, CtorRequestedSizeOneRoundsToOne)
{
  /* Smallest legal size.  next_power_of_two(1) is 1 (stable_bits == ~0),
     so the stored size must also be 1 and the address must round-trip.  */
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, /* requested_address */ 0x2000ull,
    /* requested_size */ 1ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);
  EXPECT_EQ (wp.address (), 0x2000ull);
  EXPECT_EQ (wp.size (), 1ull);
}

TEST (Watchpoint, CtorAtAddressZeroRoundTrips)
{
  /* address 0 is a legal kernel-side global address; the alignment math
     must not blow up on it.  */
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, /* requested_address */ 0ull,
    /* requested_size */ 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_STORE_AND_RMW);
  EXPECT_EQ (wp.address (), 0ull);
  EXPECT_EQ (wp.size (), 0x40ull);
}

TEST (Watchpoint, CtorLargeAlignedSizeRoundTrips)
{
  /* A page-sized (4KB) aligned watch must not grow.  Exercises the
     stable_bits computation on a span >= one page.  */
  auto tp = make_test_process ();
  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, /* requested_address */ 0x10000ull,
    /* requested_size */ 0x1000ull, AMD_DBGAPI_WATCHPOINT_KIND_ALL);
  EXPECT_EQ (wp.address (), 0x10000ull);
  EXPECT_EQ (wp.size (), 0x1000ull);
}

TEST (Watchpoint, CtorPreservesKindForAllEnumerators)
{
  /* The ctor doesn't validate kind -- kind() must round-trip whatever
     was passed.  Mapping to os_watch_mode_t happens at insert time and
     is covered by AgentWatchpoint.InsertTranslatesKindToWatchMode.  */
  auto tp = make_test_process ();
  for (auto k : { AMD_DBGAPI_WATCHPOINT_KIND_LOAD,
                  AMD_DBGAPI_WATCHPOINT_KIND_STORE_AND_RMW,
                  AMD_DBGAPI_WATCHPOINT_KIND_RMW,
                  AMD_DBGAPI_WATCHPOINT_KIND_ALL })
    {
      auto &wp = tp.process->create<watchpoint_t> (*tp.process, 0x4000ull,
                                                   0x40ull, k);
      EXPECT_EQ (wp.kind (), k);
    }
}

/* ------------------------------------------------------------------ */
/* agent_t::insert_watchpoint / remove_watchpoint                      */
/* ------------------------------------------------------------------ */

TEST (AgentWatchpoint, InsertCallsSetAddressWatchAndRecordsPointer)
{
  auto tp = make_test_process ();
  auto info = make_debugging_agent_info (7);
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x1000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);

  /* LOAD -> os_watch_mode_t::read.  Driver returns os_watch_id 2.  */
  EXPECT_CALL (*tp.driver,
               set_address_watch (7u, /* address */ agent_address_t{ 0x1000 },
                                  /* mask    */ agent_address_t{ ~uint64_t{ 0x3F } },
                                  os_watch_mode_t::read, _))
    .WillOnce (
      DoAll (SetArgPointee<4> (2u), Return (AMD_DBGAPI_STATUS_SUCCESS)));

  agent.insert_watchpoint (wp);

  /* Recorded at m_watchpoints[os_watch_id].  */
  EXPECT_EQ (agent.get_watchpoint (2), &wp);
}

TEST (AgentWatchpoint, InsertTranslatesKindToWatchMode)
{
  /* Spot-check the kind -> watch_mode mapping for RMW and ALL.  */
  auto tp = make_test_process ();
  auto info = make_debugging_agent_info (3);
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  auto &wp_rmw = tp.process->create<watchpoint_t> (
    *tp.process, 0x1000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_RMW);
  auto &wp_all = tp.process->create<watchpoint_t> (
    *tp.process, 0x2000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_ALL);

  EXPECT_CALL (*tp.driver,
               set_address_watch (3u, _, _, os_watch_mode_t::atomic, _))
    .WillOnce (
      DoAll (SetArgPointee<4> (0u), Return (AMD_DBGAPI_STATUS_SUCCESS)));
  agent.insert_watchpoint (wp_rmw);

  EXPECT_CALL (*tp.driver,
               set_address_watch (3u, _, _, os_watch_mode_t::all, _))
    .WillOnce (
      DoAll (SetArgPointee<4> (1u), Return (AMD_DBGAPI_STATUS_SUCCESS)));
  agent.insert_watchpoint (wp_all);
}

TEST (AgentWatchpoint, InsertThrowsNoWatchpointAvailable)
{
  auto tp = make_test_process ();
  auto info = make_debugging_agent_info (5);
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x1000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);

  EXPECT_CALL (*tp.driver, set_address_watch (_, _, _, _, _))
    .WillOnce (Return (AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE));

  try
    {
      agent.insert_watchpoint (wp);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_NO_WATCHPOINT_AVAILABLE);
    }
}

TEST (AgentWatchpoint, RemoveCallsClearAddressWatchWithStoredId)
{
  auto tp = make_test_process ();
  auto info = make_debugging_agent_info (9);
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x1000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);

  EXPECT_CALL (*tp.driver, set_address_watch (_, _, _, _, _))
    .WillOnce (
      DoAll (SetArgPointee<4> (1u), Return (AMD_DBGAPI_STATUS_SUCCESS)));
  agent.insert_watchpoint (wp);

  /* remove must call clear_address_watch with (os_agent_id, os_watch_id=1).  */
  EXPECT_CALL (*tp.driver, clear_address_watch (9u, 1u))
    .WillOnce (Return (AMD_DBGAPI_STATUS_SUCCESS));
  agent.remove_watchpoint (wp);
}

TEST (AgentWatchpoint, RemoveIsNoOpWhenWatchpointNotInserted)
{
  auto tp = make_test_process ();
  auto info = make_debugging_agent_info (11);
  auto &agent = tp.process->create<agent_t> (*tp.process, gfx942 (), info);

  auto &wp = tp.process->create<watchpoint_t> (
    *tp.process, 0x1000ull, 0x40ull, AMD_DBGAPI_WATCHPOINT_KIND_LOAD);

  /* Never inserted -> agent must NOT call the driver.  */
  EXPECT_CALL (*tp.driver, clear_address_watch (_, _)).Times (0);
  agent.remove_watchpoint (wp);
}
