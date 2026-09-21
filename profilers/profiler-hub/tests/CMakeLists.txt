# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ------------------------------------------------------------ #
# Unit Tests (Google Test)
# ------------------------------------------------------------ #

if(PROFILER_HUB_BUILD_TESTS)
    add_subdirectory(unit)
endif()

# ------------------------------------------------------------ #
# Performance Benchmarks (Google Benchmark)
# ------------------------------------------------------------ #

if(PROFILER_HUB_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
