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

   The child binary path is baked in at compile time via the
   DBGAPI_HIP_IDLE_WORKLOAD_PATH define, which the CMake target sets
   to the absolute path of the built idle_kernel executable.  Tests
   that link against this support code must therefore be added
   alongside the workload registration so the macro is provided; if
   it is missing the test target won't compile.

   If HIP wasn't found at configure time, the workload target isn't
   built and DBGAPI_HIP_IDLE_WORKLOAD_PATH is set to an empty string;
   spawn_hip_workload() returns an invalid idle_child_t in that case
   and the test should GTEST_SKIP.  */

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
