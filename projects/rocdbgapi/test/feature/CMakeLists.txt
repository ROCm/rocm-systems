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
#
# Populated by later commits:
#   Commit 17       - capability probes, base fixture
#   Commit 18       - init/topology against real KFD/KMD
#   Commit 19       - self-attach tests, portable spawn_idle_child
#   Commit 20       - HIP no-op workload runner
#   Commits 21-23   - wave inspection, control flow, exceptions on real GPU

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

# Architecture queries through the public amd_dbgapi API.
# The architecture registry is populated at static-init, so these
# tests don't need the kernel driver — they run on every host and
# complement the unit-test version (test/unit/architecture_test.cpp,
# which goes through the internal seam).  Value-add over the unit
# test: exercises the public symbol exports, ABI, and the
# allocate_memory callback wiring for string / breakpoint-instruction
# outputs.  No REQUIRES_DRIVER flag.
add_dbgapi_feature_test(dbgapi_feature_architecture_query_test
  SOURCES
    architecture_query_test.cpp
    support/capabilities.cpp
  LIBS amd-dbgapi)

# Public-API handle-misuse sweep.  Exercises the "find returns nullptr
# -> INVALID_*_ID" path on every object type by passing a bogus handle
# to each get_info / list / mutation entry point.  No driver required;
# the library is initialized but no process is attached.
add_dbgapi_feature_test(dbgapi_feature_handle_misuse_test
  SOURCES
    handle_misuse_test.cpp
    support/capabilities.cpp
  LIBS amd-dbgapi)

# Self-attach feature tests.  Exercises amd_dbgapi_process_attach /
# amd_dbgapi_process_detach against the current process and a freshly
# spawned idle child.  No GPU work; the kernel driver is still
# required (KFD on Linux) so the target is REQUIRES_DRIVER and skips
# cleanly on hosts without /dev/kfd.
add_dbgapi_feature_test(dbgapi_feature_self_attach_test
  SOURCES
    self_attach_test.cpp
    support/capabilities.cpp
    support/spawn_idle_child.cpp
  REQUIRES_DRIVER
  LIBS amd-dbgapi)

# HIP idle workload + workload-attach feature test.
#
# Built only when find_package(hip) succeeds; otherwise the workload
# binary doesn't exist and spawn_hip_workload() — which probes for
# "dbgapi_test_idle_kernel" next to the running test executable at
# runtime — returns invalid so the test GTEST_SKIPs.  This keeps the
# same test binary working on hosts that have a GPU but no HIP, and
# avoids baking absolute build-tree paths into the binary so the
# install tree (tests/rocdbgapi/) works the same way as the build
# tree (build/test/feature/).
find_package(hip QUIET)
if(hip_FOUND)
  enable_language(HIP)
  add_executable(dbgapi_test_idle_kernel workloads/idle_kernel.hip)
  set_target_properties(dbgapi_test_idle_kernel PROPERTIES
    HIP_ARCHITECTURES "gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030")
  target_link_libraries(dbgapi_test_idle_kernel PRIVATE hip::host)

  # Ship the workload next to the test binaries so the runtime
  # lookup in spawn_hip_workload.cpp finds it.
  install(TARGETS dbgapi_test_idle_kernel
    RUNTIME DESTINATION tests/rocdbgapi
    COMPONENT tests)
else()
  message(STATUS
    "[dbgapi tests] HIP not found — GPU workload tests will GTEST_SKIP")
endif()

add_dbgapi_feature_test(dbgapi_feature_hip_workload_test
  SOURCES
    hip_workload_test.cpp
    support/capabilities.cpp
    support/spawn_idle_child.cpp
    support/spawn_hip_workload.cpp
  REQUIRES_DRIVER
  REQUIRES_GPU
  LIBS amd-dbgapi)
if(hip_FOUND)
  add_dependencies(dbgapi_feature_hip_workload_test dbgapi_test_idle_kernel)
endif()

# Wave inspection — list-and-classify queries against waves from the
# idle HIP kernel.  Same prerequisites as the workload test.
add_dbgapi_feature_test(dbgapi_feature_wave_inspection_test
  SOURCES
    wave_inspection_test.cpp
    support/capabilities.cpp
    support/spawn_idle_child.cpp
    support/spawn_hip_workload.cpp
  REQUIRES_DRIVER
  REQUIRES_GPU
  LIBS amd-dbgapi)
if(hip_FOUND)
  add_dependencies(dbgapi_feature_wave_inspection_test dbgapi_test_idle_kernel)
endif()

# Control flow — wave_stop / WAVE_STOP event / WAVE_INFO_PC / wave_resume
# against the spinning idle HIP kernel.  Same prerequisites as the wave
# inspection test.
add_dbgapi_feature_test(dbgapi_feature_control_flow_test
  SOURCES
    control_flow_test.cpp
    support/capabilities.cpp
    support/spawn_idle_child.cpp
    support/spawn_hip_workload.cpp
  REQUIRES_DRIVER
  REQUIRES_GPU
  LIBS amd-dbgapi)
if(hip_FOUND)
  add_dependencies(dbgapi_feature_control_flow_test dbgapi_test_idle_kernel)
endif()

# Displaced stepping + exception delivery against the spinning idle HIP
# kernel.  Same prerequisites as the other GPU tests.
add_dbgapi_feature_test(dbgapi_feature_displaced_stepping_test
  SOURCES
    displaced_stepping_test.cpp
    support/capabilities.cpp
    support/spawn_idle_child.cpp
    support/spawn_hip_workload.cpp
  REQUIRES_DRIVER
  REQUIRES_GPU
  LIBS amd-dbgapi)
if(hip_FOUND)
  add_dependencies(dbgapi_feature_displaced_stepping_test dbgapi_test_idle_kernel)
endif()
