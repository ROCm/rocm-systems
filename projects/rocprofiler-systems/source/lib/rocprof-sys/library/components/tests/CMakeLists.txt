# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Component Tests
add_library(
    lib-components-tests
    OBJECT
    test_shmem_gotcha.cpp
    test_ucx_gotcha.cpp
    test_pthread_mutex_gotcha.cpp
    test_category_region.cpp
)

target_link_libraries(
    lib-components-tests
    PUBLIC
        rocprofiler-systems-headers # defines.hpp
        rocprofiler-systems-logger
        rocprofiler-systems-googletest-library
        rocprofiler-systems-timemory
        rocprofiler-systems-core-library
        # category_region.hpp pulls in tracing/annotation/rocprofiler-sdk/perfetto headers
        rocprofiler-systems::rocprofiler-systems-interface-library
)

target_include_directories(
    lib-components-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys ${PROJECT_SOURCE_DIR}/source/lib
)
