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

/* spawn_hip_workload: launch the idle_kernel workload child and
   block until it prints "READY\n", so dbgapi attach is guaranteed
   to race against a queue with at least one in-flight dispatch.

   The workload binary is located at runtime by looking next to the
   running test executable for "dbgapi_test_idle_kernel".  Both
   supported layouts colocate the workload with the test binaries:
   the build tree puts everything under build/test/feature/, and the
   install tree puts everything under tests/rocdbgapi/.  No paths
   are baked in at compile time.

   If the workload binary isn't found (e.g. HIP wasn't available at
   configure time so it was never built, or the install dropped it),
   spawn_hip_workload() returns an invalid idle_child_t and the test
   should GTEST_SKIP.  */

#ifndef DBGAPI_FEATURE_SUPPORT_SPAWN_HIP_WORKLOAD_H
#define DBGAPI_FEATURE_SUPPORT_SPAWN_HIP_WORKLOAD_H

#include "spawn_idle_child.h"

namespace amd::dbgapi::test
{

/* Spawn the idle_kernel workload and block until it signals READY
   on its stdout.  Returns an invalid handle if the workload binary
   is unavailable or fails to launch.  */
idle_child_t spawn_hip_workload ();

/* True if the workload binary path was wired in at compile time
   (i.e. HIP was found at configure time).  Useful for tests that
   want a single GTEST_SKIP at SetUp() rather than spawn-and-check.  */
bool hip_workload_available ();

} /* namespace amd::dbgapi::test */

#endif /* DBGAPI_FEATURE_SUPPORT_SPAWN_HIP_WORKLOAD_H */
