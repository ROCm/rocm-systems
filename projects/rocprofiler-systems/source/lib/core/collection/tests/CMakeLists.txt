# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(core_collection_tests_sources
    test_fork_gate.cpp
    test_guarded_read.cpp
    test_sample_collector.cpp
    test_scheduler.cpp
    test_source_runner.cpp
    test_tick_grid.cpp
)

add_library(core-collection-tests OBJECT ${core_collection_tests_sources})

target_link_libraries(
    core-collection-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
)

target_include_directories(core-collection-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)
