# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Deliberately does not link rocprofiler-systems-hipfile-backend: that target pulls in
# hip::hipfile and therefore <hipfile.h>. These tests drive backend<> through a mock
# wrapper with test-local fake structs, so they build and run on machines with no
# hipFile package installed.
add_library(hipfile-backend-tests OBJECT test_backend.cpp)

target_link_libraries(
    hipfile-backend-tests
    PUBLIC
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-compile-definitions
)

target_include_directories(
    hipfile-backend-tests
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${PROJECT_BINARY_DIR}/source/lib
)
