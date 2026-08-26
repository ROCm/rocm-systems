# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# HRR project tests.
#
#   unit/        CPU-focused format, recovery, and playback-invariant tests.
#   integration/ GPU capture/replay round-trip tests (require an AMD GPU and a
#                capture-enabled HIP runtime).
#
# The HRR tests use a small local harness (hrr_test_common.hh / hrr_test_process.hh)
# instead of the hip-tests Catch2 infrastructure.

option(HRR_BUILD_INTEGRATION_TESTS "Build HRR GPU capture/replay integration tests" ON)

# Catch2 test framework. Prefer an installed Catch2 (as the hip-tests suite
# does); fall back to fetching the pinned tarball used across rocm-systems.
find_package(Catch2 3.8.1 QUIET)
if(NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/Catch2-3.8.1.tar.gz"
  )
  FetchContent_MakeAvailable(Catch2)
endif()

# HIP is needed for the HIP headers pulled in by the local test harness and the
# playback helper header, and for hip::host linkage in the integration tests.
# The unit tests make no HIP runtime calls, but their sources still include
# <hip/hip_runtime.h>, so the HIP package (headers) is required to build them.
find_package(hip CONFIG REQUIRED)

add_subdirectory(unit)

if(HRR_BUILD_INTEGRATION_TESTS)
  if(NOT HRR_BUILD_PLAYBACK)
    message(FATAL_ERROR "HRR_BUILD_INTEGRATION_TESTS requires HRR_BUILD_PLAYBACK because integration tests spawn hrr-playback")
  endif()
  add_subdirectory(integration)
endif()
