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

/* End-to-end smoke test for the Commit 11 + Commit 12 infrastructure:

     1) MockOsDriver (a GMock-generated os_driver_t) is constructible
        and answers is_valid() truthfully when programmed.

     2) process_test_access::make() routes through the test-only
        friend-key constructor in process_t, accepts the supplied
        os_driver, and exposes it via process_t::os_driver().

     3) Destroying the process_t releases the mock (no leak, no
        un-expected-call warning).

   The interesting per-method behavior of MockOsDriver is exercised
   in later commits — agent/queue/process-lifecycle tests in
   Commits 13-14 — so this file stays a thin "the seam works" test.  */

#include "amd-dbgapi.h"
#include "os_driver.h"
#include "process.h"

#include "support/mock_os_driver.h"
#include "support/process_test_access.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <utility>

using ::testing::NiceMock;
using ::testing::Return;

using amd::dbgapi::os_driver_t;
using amd::dbgapi::process_t;
using amd::dbgapi::test::MockOsDriver;

namespace
{

/* The client_process_id is stored by value but never dereferenced in the
   test ctor path, so any unique pointer-sized value works as a fake.  */
amd_dbgapi_client_process_id_t
fake_client_id ()
{
  static int sentinel = 0;
  return reinterpret_cast<amd_dbgapi_client_process_id_t> (&sentinel);
}

} /* namespace */

TEST (MockOsDriverSmoke, IsConstructibleAndAnswersIsValid)
{
  MockOsDriver driver;
  EXPECT_CALL (driver, is_valid ()).WillOnce (Return (true));
  EXPECT_TRUE (driver.is_valid ());
}

TEST (MockOsDriverSmoke, InjectsIntoProcessViaFriendKey)
{
  auto driver = std::make_unique<NiceMock<MockOsDriver>> ();

  /* The test ctor calls is_valid() once during construction; program
     a default Return(true) for any call.  */
  ON_CALL (*driver, is_valid ()).WillByDefault (Return (true));

  MockOsDriver *driver_raw = driver.get ();

  auto proc = process_test_access::make (
    amd_dbgapi_process_id_t{ 1 }, fake_client_id (),
    /* os_process_id */ std::nullopt,
    std::unique_ptr<os_driver_t> (driver.release ()));

  ASSERT_NE (proc, nullptr);
  EXPECT_EQ (&proc->os_driver (), driver_raw);
  EXPECT_EQ (proc->id ().handle, 1u);
  EXPECT_TRUE (proc->from_core ()) /* no os_process_id supplied  */;
}

TEST (MockOsDriverSmoke, ProcessDestructionReleasesMock)
{
  /* No EXPECT_CALL — NiceMock keeps uninteresting calls quiet, and the
     test passes iff no leak is reported when the unique_ptr drops.  */
  auto driver = std::make_unique<NiceMock<MockOsDriver>> ();
  ON_CALL (*driver, is_valid ()).WillByDefault (Return (true));

  auto proc = process_test_access::make (
    amd_dbgapi_process_id_t{ 2 }, fake_client_id (), std::nullopt,
    std::unique_ptr<os_driver_t> (driver.release ()));

  proc.reset ();
  SUCCEED ();
}
