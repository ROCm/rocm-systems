## SPDX-License-Identifier: MIT
## SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

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

