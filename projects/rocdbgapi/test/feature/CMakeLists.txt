## Copyright (c) 2026 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

# Feature tests for rocdbgapi.
#
# Tests added here may require the kernel driver (KFD on Linux, KMD on
# Windows) and, for the GPU sub-tier, a real GPU plus a HIP runtime.
# Each test must skip cleanly (GTEST_SKIP) when its prerequisites are
# absent so the same binary works on hosts without hardware.

# Feature-test harness smoke test.  Compiles + links against the
# shipping amd-dbgapi public library, exercises the capability probes
# in support/capabilities.{h,cpp}, and walks the three fixture tiers
# defined in support/dbgapi_fixture.h (no driver / driver / GPU).
# Marked REQUIRES_DRIVER for CTest label routing, but the individual
# tests use GTEST_SKIP at runtime so the binary still runs cleanly on
# driver-less hosts.
add_dbgapi_feature_test(dbgapi_feature_smoke_test
  SOURCES
    smoke_test.cpp
    support/capabilities.cpp
  REQUIRES_DRIVER
  LIBS amd-dbgapi)

