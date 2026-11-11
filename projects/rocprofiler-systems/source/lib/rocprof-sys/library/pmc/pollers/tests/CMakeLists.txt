# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The pollers run on the mock backends next to the collector devices, so no AMD SMI,
# procfs, hipFile or SDK runtime is needed.
add_library(
    pmc-pollers-tests
    OBJECT
    test_cpu.cpp
    test_gpu_smi.cpp
    test_hipfile.cpp
    test_nic_smi.cpp
)

# The GPU perf-counter device needs the ROCm >= 6.4 device counting service
if(
    (ROCPROFSYS_ROCM_VERSION_MAJOR GREATER 6)
    OR (
        ROCPROFSYS_ROCM_VERSION_MAJOR EQUAL 6
        AND ROCPROFSYS_ROCM_VERSION_MINOR GREATER_EQUAL 4
    )
)
    target_sources(pmc-pollers-tests PRIVATE test_gpu_perf_counter.cpp)
endif()

target_link_libraries(
    pmc-pollers-tests
    PRIVATE
        rocprofiler-systems-googletest-library
        rocprofiler-systems-core-library
        rocprofiler-systems-logger
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-rocm
        rocprofiler-systems::rocprofiler-systems-timemory
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-interface-library
)

target_include_directories(
    pmc-pollers-tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}/source/lib
        ${PROJECT_SOURCE_DIR}/source/lib/rocprof-sys
        ${PROJECT_BINARY_DIR}/source/lib
)
