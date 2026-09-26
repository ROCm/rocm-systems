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

/* GTest fixtures for feature tests.

   - dbgapi_fixture_t initializes the library in SetUp() and finalizes
     it in TearDown().  Suitable for any test that only needs a live
     library handle; it does NOT attach to any process.

   - driver_fixture_t adds a GTEST_SKIP at SetUp() time when the kernel
     driver isn't reachable, so the same binary works on host VMs.

   - gpu_fixture_t skips additionally when no GPU node is reported by
     the driver topology.  Subclasses that need a specific gfx target
     can call require_gfx(version_int) from SetUp().

   Subclasses can override callbacks() to supply richer behavior than
   the minimal defaults.  */

#ifndef DBGAPI_FEATURE_SUPPORT_DBGAPI_FIXTURE_H
#define DBGAPI_FEATURE_SUPPORT_DBGAPI_FIXTURE_H

#include "amd-dbgapi.h"

#include "capabilities.h"
#include "minimal_callbacks.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>

namespace amd::dbgapi::test
{

class dbgapi_fixture_t : public ::testing::Test
{
protected:
  virtual const amd_dbgapi_callbacks_s &callbacks () const
  {
    return minimal_callbacks ();
  }

  void SetUp () override
  {
    auto &cb = callbacks ();
    ASSERT_EQ (
      amd_dbgapi_initialize (
        const_cast<amd_dbgapi_callbacks_s *> (&cb)),
      AMD_DBGAPI_STATUS_SUCCESS);
    m_initialized = true;
  }

  void TearDown () override
  {
    if (m_initialized)
      {
        EXPECT_EQ (amd_dbgapi_finalize (), AMD_DBGAPI_STATUS_SUCCESS);
        m_initialized = false;
      }
  }

private:
  bool m_initialized = false;
};

class driver_fixture_t : public dbgapi_fixture_t
{
protected:
  void SetUp () override
  {
    if (!is_driver_available ())
      GTEST_SKIP () << "kernel driver not available on this host";
    dbgapi_fixture_t::SetUp ();
  }
};

class gpu_fixture_t : public driver_fixture_t
{
protected:
  void SetUp () override
  {
    if (!is_gpu_available ())
      GTEST_SKIP () << "no GPU reported by driver topology";
    driver_fixture_t::SetUp ();
  }

  /* Skip the test unless the host has at least one GPU matching
     gfx_target_version (e.g. 90402 for gfx942).  Call from a derived
     test's SetUp() *after* gpu_fixture_t::SetUp().  */
  void require_gfx (uint32_t version)
  {
    auto targets = enumerate_gfx_targets ();
    if (std::find (targets.begin (), targets.end (), version)
        == targets.end ())
      GTEST_SKIP () << "no GPU with gfx_target_version " << version;
  }
};

} /* namespace amd::dbgapi::test */

#endif /* DBGAPI_FEATURE_SUPPORT_DBGAPI_FIXTURE_H */
