# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

add_library(rccl-unit-tests OBJECT test_rccl_unit.cpp)

target_link_libraries(
    rccl-unit-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-rocm
)

target_include_directories(
    rccl-unit-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib ${CMAKE_BINARY_DIR}/source/lib/
)

add_library(roctx-client-tests OBJECT test_roctx_client.cpp)

target_link_libraries(
    roctx-client-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-rocm
        rocprofiler-systems::rocprofiler-systems-timemory
)

# TODO: clean up this
target_include_directories(
    roctx-client-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${CMAKE_BINARY_DIR}/source/lib/
)
