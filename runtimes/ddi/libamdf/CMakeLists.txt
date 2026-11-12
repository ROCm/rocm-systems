# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

project(ROCmRuntimeAMDF VERSION 0.1 LANGUAGES C)

runtime_rust_library(runtime_amdf_static
  PACKAGE libamdf LIBRARY amdf TYPE STATIC OUTPUT_NAME amdf
)
add_library(rocm_runtime::amdf_static ALIAS runtime_amdf_static)
target_link_libraries(
    runtime_amdf_static
    INTERFACE rocm_runtime::runtime_headers
)

if(BUILD_TESTING)
    add_executable(amdf_smoke_static tests/abi/native_c_smoke.c)
    target_compile_features(amdf_smoke_static PRIVATE c_std_11)
    target_link_libraries(
        amdf_smoke_static PRIVATE rocm_runtime::amdf_static
    )
    add_test(NAME amdf.smoke.static COMMAND amdf_smoke_static)
    set_tests_properties(amdf.smoke.static PROPERTIES LABELS "host;abi")
    add_executable(amdf_layout tests/abi/amdf_layout.c)
    target_compile_features(amdf_layout PRIVATE c_std_11)
    target_link_libraries(amdf_layout PRIVATE rocm_runtime::runtime_headers)
    add_test(NAME amdf.layout COMMAND amdf_layout)
    set_tests_properties(amdf.layout PROPERTIES LABELS "host;abi")
endif()
