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

/* Unit tests for displaced_stepping_t at the object level (no real wave,
   no queue suspend, no instruction-buffer allocation):

     * Construction with a legal instruction and std::nullopt destination
       (the "simulate, don't single-step" path).  Round-trips original_
       instruction(), from(), to() == std::nullopt, and queue()/agent()/
       process()/architecture() backlinks.
     * retain() / release() reference counting.  release() at refcount 1
       triggers process().destroy(); the dtor asserts m_reference_count
       == 0, so this also verifies no leak.
     * get_info(PROCESS) returns the owning process id.
     * get_info(unknown query) throws api_error_t carrying
       INVALID_ARGUMENT.

   Construction is via process->create<displaced_stepping_t>(...) because
   release() destroys through the process handle set; stack construction
   would crash on release.  The queue is a test_compute_queue_t set to
   suspended so no queue-state machinery runs.  */

#include "agent.h"
#include "amd-dbgapi.h"
#include "architecture.h"
#include "displaced_stepping.h"
#include "exception.h"
#include "os_driver.h"
#include "process.h"
#include "queue.h"

#include "support/mock_os_driver.h"
#include "support/process_test_access.h"
#include "support/test_queue.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

using ::testing::NiceMock;
using ::testing::Return;

using amd::dbgapi::agent_t;
using amd::dbgapi::api_error_t;
using amd::dbgapi::architecture_t;
using amd::dbgapi::displaced_stepping_t;
using amd::dbgapi::instruction_t;
using amd::dbgapi::legal_instruction;
using amd::dbgapi::os_agent_info_t;
using amd::dbgapi::os_driver_t;
using amd::dbgapi::process_t;
using amd::dbgapi::queue_t;
using amd::dbgapi::test::MockOsDriver;
using amd::dbgapi::test::make_test_os_queue_info;
using amd::dbgapi::test::test_compute_queue_t;

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

/* Build an instruction_t marked legal so is_valid() returns true without
   running the disassembler.  Four arbitrary bytes are enough; the test
   never executes the instruction.  */
instruction_t
make_legal_instruction (const architecture_t &arch)
{
  std::vector<std::byte> bytes (4, std::byte{ 0 });
  return instruction_t (legal_instruction, arch, std::move (bytes));
}

/* Fixture wiring up process -> agent -> suspended queue -> arch handle.
   Tests build displaced_stepping_t via process->create so release() can
   destroy through the handle set.  */
class DisplacedSteppingFixture
{
public:
  DisplacedSteppingFixture ()
    : m_tp (make_test_process ()), m_arch (gfx942 ()),
      m_agent (&m_tp.process->create<agent_t> (*m_tp.process, m_arch,
                                               make_os_agent_info ())),
      m_queue (amd_dbgapi_queue_id_t{ 11 }, *m_agent,
               make_test_os_queue_info ())
  {
    m_queue.set_state (queue_t::state_t::suspended);
  }

  process_t &process () { return *m_tp.process; }
  queue_t &queue () { return m_queue; }
  const architecture_t &arch () const { return *m_arch; }

private:
  test_process_t m_tp;
  const architecture_t *m_arch;
  agent_t *m_agent;
  test_compute_queue_t m_queue;
};

} /* namespace */

TEST (DisplacedStepping, ConstructsWithNullToAndExposesBacklinks)
{
  ASSERT_NE (gfx942 (), nullptr);
  DisplacedSteppingFixture fx;

  amd::dbgapi::agent_address_t from{ 0xabcdef00ull };
  auto &ds = fx.process ().create<displaced_stepping_t> (
    fx.queue (), make_legal_instruction (fx.arch ()), from, std::nullopt);

  EXPECT_TRUE (ds.original_instruction ().is_valid ());
  EXPECT_EQ (ds.original_instruction ().size (), 4u);
  EXPECT_EQ (ds.from (), from);
  EXPECT_FALSE (ds.to ().has_value ());
  EXPECT_EQ (&ds.queue (), &fx.queue ());
  EXPECT_EQ (&ds.process (), &fx.process ());
  EXPECT_EQ (&ds.architecture (), &fx.arch ());

  /* Tear down through the process so dtor's refcount==0 assert holds.  */
  fx.process ().destroy (&ds);
}

TEST (DisplacedStepping, RetainReleaseDestroysAtZeroRefCount)
{
  ASSERT_NE (gfx942 (), nullptr);
  DisplacedSteppingFixture fx;

  auto &ds = fx.process ().create<displaced_stepping_t> (
    fx.queue (), make_legal_instruction (fx.arch ()),
    amd::dbgapi::agent_address_t{ 0x1000 }, std::nullopt);
  auto id = ds.id ();

  displaced_stepping_t::retain (&ds);
  /* Still alive after retain.  */
  EXPECT_EQ (fx.process ().find (id), &ds);

  displaced_stepping_t::release (&ds);
  /* release() at refcount 1 destroys through the handle set.  */
  EXPECT_EQ (fx.process ().find (id), nullptr);
}

TEST (DisplacedStepping, GetInfoProcessRoundTripsId)
{
  ASSERT_NE (gfx942 (), nullptr);
  DisplacedSteppingFixture fx;

  auto &ds = fx.process ().create<displaced_stepping_t> (
    fx.queue (), make_legal_instruction (fx.arch ()),
    amd::dbgapi::agent_address_t{ 0x2000 }, std::nullopt);

  amd_dbgapi_process_id_t pid{};
  ds.get_info (AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS, sizeof (pid), &pid);
  EXPECT_EQ (pid.handle, 1u);

  fx.process ().destroy (&ds);
}

TEST (DisplacedStepping, GetInfoUnknownQueryThrowsInvalidArgument)
{
  ASSERT_NE (gfx942 (), nullptr);
  DisplacedSteppingFixture fx;

  auto &ds = fx.process ().create<displaced_stepping_t> (
    fx.queue (), make_legal_instruction (fx.arch ()),
    amd::dbgapi::agent_address_t{ 0x3000 }, std::nullopt);

  uint64_t sink = 0;
  EXPECT_THROW (
    ds.get_info (
      static_cast<amd_dbgapi_displaced_stepping_info_t> (0x7fffffff),
      sizeof (sink), &sink),
    api_error_t);

  fx.process ().destroy (&ds);
}

TEST (DisplacedStepping, GetInfoNullValuePointerThrowsInvalidArgument)
{
  ASSERT_NE (gfx942 (), nullptr);
  DisplacedSteppingFixture fx;

  auto &ds = fx.process ().create<displaced_stepping_t> (
    fx.queue (), make_legal_instruction (fx.arch ()),
    amd::dbgapi::agent_address_t{ 0x4000 }, std::nullopt);

  try
    {
      ds.get_info (AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS,
                   sizeof (amd_dbgapi_process_id_t), nullptr);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (), AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT);
    }

  fx.process ().destroy (&ds);
}

TEST (DisplacedStepping, GetInfoWrongValueSizeThrowsInvalidArgumentCompatibility)
{
  ASSERT_NE (gfx942 (), nullptr);
  DisplacedSteppingFixture fx;

  auto &ds = fx.process ().create<displaced_stepping_t> (
    fx.queue (), make_legal_instruction (fx.arch ()),
    amd::dbgapi::agent_address_t{ 0x5000 }, std::nullopt);

  amd_dbgapi_process_id_t pid{};
  try
    {
      /* Pass a size that does not match sizeof(amd_dbgapi_process_id_t).  */
      ds.get_info (AMD_DBGAPI_DISPLACED_STEPPING_INFO_PROCESS, sizeof (pid) + 1,
                   &pid);
      FAIL () << "expected api_error_t";
    }
  catch (const api_error_t &e)
    {
      EXPECT_EQ (e.code (),
                 AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT_COMPATIBILITY);
    }

  fx.process ().destroy (&ds);
}
