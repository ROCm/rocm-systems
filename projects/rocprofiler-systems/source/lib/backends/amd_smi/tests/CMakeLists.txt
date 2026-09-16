# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(amd-smi-backend-tests OBJECT test_backend.cpp test_device_backend.cpp)

target_link_libraries(
    amd-smi-backend-tests
    PUBLIC
        rocprofiler-systems-googletest-library
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-amd-smi-backend
        rocprofiler-systems::rocprofiler-systems-compile-definitions
)

target_include_directories(amd-smi-backend-tests PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_definitions(
    amd-smi-backend-tests
    PUBLIC ROCPROFSYS_BUILD_AINIC=1 AMD_SMI_SDMA_SUPPORTED=1
)
