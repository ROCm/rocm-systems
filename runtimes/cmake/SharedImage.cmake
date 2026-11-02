# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# The one shared image carries the HSA native ABI version and both public ABIs.
set(_runtime_hsa_abi_version "1.21.0")
string(REGEX MATCH "^[0-9]+" _runtime_hsa_abi_major "${_runtime_hsa_abi_version}")

runtime_rust_library(runtime_shared_image
  PACKAGE rocddi-frontends LIBRARY rocddi_runtime TYPE SHARED
  OUTPUT_NAME hsa-runtime64 VERSION "${_runtime_hsa_abi_version}"
  SOVERSION "${_runtime_hsa_abi_major}"
  SONAME_ENV ROCM_RUNTIME_HSA_SONAME
)
add_library(runtime_hsa_shared ALIAS runtime_shared_image)
add_library(runtime_amdf_shared ALIAS runtime_shared_image)
add_library(rocm_runtime::hsa_shared ALIAS runtime_shared_image)
add_library(rocm_runtime::amdf_shared ALIAS runtime_shared_image)
target_link_libraries(
    runtime_shared_image
    INTERFACE rocm_runtime::runtime_headers
)
# Older AMDF binaries request the former libamdf.so.0 SONAME.
set_property(GLOBAL APPEND PROPERTY _runtime_link_commands
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
    "libhsa-runtime64.so.${_runtime_hsa_abi_major}"
    "${_runtime_library_dir}/libamdf.so.0"
)
set_property(GLOBAL APPEND PROPERTY _runtime_link_commands
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
    "libamdf.so.0"
    "${_runtime_library_dir}/libamdf.so"
)
set_property(GLOBAL APPEND PROPERTY _runtime_link_commands
    COMMAND "${CMAKE_COMMAND}" -E create_symlink
    "libhsa-runtime64.so.${_runtime_hsa_abi_major}"
    "${_runtime_library_dir}/libhsa_runtime64.so"
)
set_property(GLOBAL APPEND PROPERTY _runtime_byproducts
    "${_runtime_library_dir}/libamdf.so.0"
    "${_runtime_library_dir}/libamdf.so"
    "${_runtime_library_dir}/libhsa_runtime64.so"
)

if(BUILD_TESTING)
    add_executable(amdf_smoke_shared
        "${CMAKE_CURRENT_LIST_DIR}/../ddi/libamdf/tests/abi/native_c_smoke.c")
    target_compile_features(amdf_smoke_shared PRIVATE c_std_11)
    target_link_libraries(amdf_smoke_shared PRIVATE rocm_runtime::amdf_shared)
    add_test(NAME amdf.smoke.shared COMMAND amdf_smoke_shared)
    set_tests_properties(amdf.smoke.shared PROPERTIES LABELS "host;abi")
endif()
