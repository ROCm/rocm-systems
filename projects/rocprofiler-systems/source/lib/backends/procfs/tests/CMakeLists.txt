# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(procfs-backend-tests OBJECT test_backend.cpp test_backend_parsing.cpp)

target_link_libraries(
    procfs-backend-tests
    PUBLIC rocprofiler-systems-googletest-library rocprofiler-systems-logger
)

target_include_directories(
    procfs-backend-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
)
