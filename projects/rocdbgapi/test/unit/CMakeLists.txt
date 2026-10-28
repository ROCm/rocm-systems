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

# handle_object<>: the CRTP base + handle_object_set_t<> bookkeeping used
# by process/agent/queue/wave/...  Tested with a synthetic handle/object
# pair so the test doesn't drag in agent_t / queue_t or any architecture
# state.
add_dbgapi_unit_test(dbgapi_handle_object_test
  SOURCES handle_object_test.cpp
  LIBS amd-dbgapi-internal)

