## SPDX-License-Identifier: MIT
## SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

# Unit tests for rocdbgapi.
#
# Tests added here must NOT depend on /dev/kfd, dxg/libdxg, or a real
# GPU.

# Toolchain smoke test.  Confirms that GoogleTest is wired up on the
# host platform.  Uses the add_dbgapi_unit_test() helper which applies
# the standard labels ("dbgapi", "unit", OS label), C++17 settings,
# GTest/GMock linkage, and CTest working directory.
add_dbgapi_unit_test(dbgapi_smoke_test SOURCES smoke_test.cpp)

