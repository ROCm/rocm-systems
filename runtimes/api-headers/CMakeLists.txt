# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

add_library(rocm_runtime_api_headers INTERFACE)
target_include_directories(
    rocm_runtime_api_headers
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
add_library(rocm_runtime::runtime_headers ALIAS rocm_runtime_api_headers)

option(ABCE_BUILD_TESTS "Build the GPU-free ABCE unit tests" OFF)
if(ABCE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/abce)
endif()
