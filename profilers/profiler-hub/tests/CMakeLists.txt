# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------ #
# Unit Tests (Google Test)
# ------------------------------------------------------------ #

if(PROFILER_HUB_BUILD_TESTS)
    add_subdirectory(unit)
    # Gives ctest a subdirs(integration) entry. The pytest build step fills
    # that directory's CTestTestfile.cmake and leaves this file's subdirs(unit)
    # line alone.
    add_subdirectory(integration)
endif()

# ------------------------------------------------------------ #
# Performance Benchmarks (Google Benchmark)
# ------------------------------------------------------------ #

if(PROFILER_HUB_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
