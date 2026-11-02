# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(posix-backend-tests OBJECT test_backend.cpp)

target_link_libraries(
    posix-backend-tests
    PUBLIC
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-posix-backend
)
