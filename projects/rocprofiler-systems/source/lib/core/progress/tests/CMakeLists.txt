# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(progress_tests_sources test_format_bar.cpp test_bar_silent.cpp test_tracker.cpp)

add_library(progress-tests OBJECT ${progress_tests_sources})

target_link_libraries(
    progress-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-core-library
)

target_include_directories(progress-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)
