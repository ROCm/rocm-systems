# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Component Tests
add_library(lib-components-tests OBJECT test_shmem_gotcha.cpp)

target_link_libraries(
    lib-components-tests
    PUBLIC
        rocprofiler-systems-headers # defines.hpp
        rocprofiler-systems-googletest-library
        rocprofiler-systems-timemory
        rocprofiler-systems-libva # needed for include path to va/va.h
        rocprofiler-systems-object-library
        rocprofiler-systems-perfetto
        rocprofiler-systems-core-library
        rocprofiler-systems-logger # spdlog for headers pulled in via category_region -> trace_cache
)

target_include_directories(
    lib-components-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys ${PROJECT_SOURCE_DIR}/source/lib
)
