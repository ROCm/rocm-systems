# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    avail-catalog-tests
    OBJECT
    test_catalog.cpp
    test_cpu.cpp
    test_traces.cpp
    test_gpu_inventory.cpp
    test_gpu_metrics.cpp
    test_nic.cpp
)

target_link_libraries(
    avail-catalog-tests
    PUBLIC
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-avail-library
        rocprofiler-systems::rocprofiler-systems-amd-smi-backend
        rocprofiler-systems::rocprofiler-systems-procfs-backend
)

target_include_directories(
    avail-catalog-tests
    PRIVATE ${PROJECT_SOURCE_DIR}/source/lib/backends/amd_smi/tests
)
