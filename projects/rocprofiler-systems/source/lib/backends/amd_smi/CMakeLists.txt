# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(
    rocprofiler-systems-amd-smi-backend
    STATIC
    backend.cpp
    backend.hpp
    gpu_backend.hpp
    gpu_types.hpp
    nic_backend.hpp
    nic_types.hpp
)
add_library(
    rocprofiler-systems::rocprofiler-systems-amd-smi-backend
    ALIAS rocprofiler-systems-amd-smi-backend
)

set_target_properties(rocprofiler-systems-amd-smi-backend PROPERTIES LINKER_LANGUAGE CXX)

target_link_libraries(
    rocprofiler-systems-amd-smi-backend
    PUBLIC
        rocprofiler-systems::rocprofiler-systems-headers
        rocprofiler-systems::rocprofiler-systems-compile-definitions
        rocprofiler-systems::rocprofiler-systems-rocm
        rocprofiler-systems-core-library
)

if(ROCPROFSYS_BUILD_AINIC)
    target_compile_definitions(
        rocprofiler-systems-amd-smi-backend
        PUBLIC ROCPROFSYS_BUILD_AINIC=1
    )
endif()
