# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    containers-tests
    OBJECT
    test_operators.cpp
    test_c_array.cpp
    test_static_vector.cpp
    test_aligned_static_vector.cpp
    test_stable_vector.cpp
)

target_link_libraries(
    containers-tests
    PRIVATE rocprofiler-systems-googletest-library rocprofiler-systems-containers-library
)

target_include_directories(containers-tests PRIVATE ${PROJECT_SOURCE_DIR}/source/lib)
