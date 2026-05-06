# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

add_library(procfs-driver-tests OBJECT test_driver.cpp test_driver_parsing.cpp)

target_link_libraries(
    procfs-driver-tests
    PUBLIC
        rocprofiler-systems-pmc-library
        rocprofiler-systems-googletest-library
        rocprofiler-systems-logger
)

target_include_directories(
    procfs-driver-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
)
