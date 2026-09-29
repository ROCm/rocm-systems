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

/* Feature-test harness smoke test.

   This file exists to prove the Commit 17 infrastructure compiles,
   links against the shipping amd-dbgapi public library, and exercises
   each tier of fixture:

     * Capabilities probes are callable and self-consistent
       (is_gpu_available implies is_driver_available; the gfx_targets
       list is non-empty iff a GPU is reported).

     * dbgapi_fixture_t initializes the library against the public
       amd_dbgapi_callbacks_s + amd_dbgapi_initialize entry point and
       can query the library version.  Runs on every host (no driver
       required).

     * driver_fixture_t skips cleanly when the driver is absent and
       passes through to dbgapi_fixture_t when present.

     * gpu_fixture_t skips additionally when no GPU is reported.

   Per-arch tests are deferred to Commit 18; this file only asserts the
   "harness works" property.  */

#include "amd-dbgapi.h"

#include "support/capabilities.h"
#include "support/dbgapi_fixture.h"

#include <gtest/gtest.h>

using amd::dbgapi::test::dbgapi_fixture_t;
using amd::dbgapi::test::driver_fixture_t;
using amd::dbgapi::test::enumerate_gfx_targets;
using amd::dbgapi::test::gpu_fixture_t;
using amd::dbgapi::test::is_driver_available;
using amd::dbgapi::test::is_gpu_available;
using amd::dbgapi::test::is_kfd_available;

/* ------------------------------------------------------------------ */
/* Capability probes — no fixture                                      */
/* ------------------------------------------------------------------ */

TEST (FeatureSmokeCapabilities, ProbesAreSelfConsistent)
{
  if (is_gpu_available ())
    {
      EXPECT_TRUE (is_driver_available ())
        << "is_gpu_available() must imply is_driver_available()";
    }

  auto targets = enumerate_gfx_targets ();
  EXPECT_EQ (!targets.empty (), is_gpu_available ());
}

TEST (FeatureSmokeCapabilities, ProbesAreCheapAndIdempotent)
{
  /* Calling them twice must give the same answer and must not fault.  */
  bool a = is_kfd_available ();
  bool b = is_kfd_available ();
  EXPECT_EQ (a, b);
}

/* ------------------------------------------------------------------ */
/* Base fixture — initialize / finalize without a driver               */
/* ------------------------------------------------------------------ */

using FeatureSmokeFixture = dbgapi_fixture_t;

TEST_F (FeatureSmokeFixture, LibraryReportsAVersion)
{
  uint32_t major = 0, minor = 0, patch = 0;
  amd_dbgapi_get_version (&major, &minor, &patch);
  /* Library version 0.0.0 would mean we initialized but the build is
     broken.  Just assert non-zero major-or-minor.  */
  EXPECT_TRUE (major != 0 || minor != 0)
    << "library reported version 0.0.0";
}

TEST_F (FeatureSmokeFixture, StatusStringIsNonNull)
{
  const char *s = nullptr;
  EXPECT_EQ (
    amd_dbgapi_get_status_string (AMD_DBGAPI_STATUS_SUCCESS, &s),
    AMD_DBGAPI_STATUS_SUCCESS);
  EXPECT_NE (s, nullptr);
}

/* ------------------------------------------------------------------ */
/* Driver fixture — skips when /dev/kfd or KMD is absent               */
/* ------------------------------------------------------------------ */

using FeatureSmokeDriver = driver_fixture_t;

TEST_F (FeatureSmokeDriver, ReachesLibraryWithDriverAvailable)
{
  /* If we got here SetUp() didn't skip, so the driver must be present.  */
  EXPECT_TRUE (is_driver_available ());
}

/* ------------------------------------------------------------------ */
/* GPU fixture — skips when no GPU is reported                         */
/* ------------------------------------------------------------------ */

using FeatureSmokeGpu = gpu_fixture_t;

TEST_F (FeatureSmokeGpu, AtLeastOneGfxTargetIsReported)
{
  auto targets = enumerate_gfx_targets ();
  EXPECT_FALSE (targets.empty ());
}
