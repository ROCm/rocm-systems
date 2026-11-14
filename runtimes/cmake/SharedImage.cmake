# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# One shared image carries the HSA SONAME and the AMDF, HSA and ABCE public ABIs.
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
add_library(runtime_abce_shared INTERFACE)
add_library(rocm_runtime::abce_shared ALIAS runtime_abce_shared)
target_link_libraries(runtime_abce_shared INTERFACE runtime_shared_image)
target_compile_definitions(runtime_abce_shared INTERFACE ABCE_ENABLE_IMPLEMENTATION=0)
target_link_libraries(
    runtime_shared_image
    INTERFACE rocm_runtime::runtime_headers
)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_property(GLOBAL APPEND PROPERTY _runtime_link_commands
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "libhsa-runtime64.so.${_runtime_hsa_abi_major}"
        "${_runtime_library_dir}/libabce.so.1"
        COMMAND "${CMAKE_COMMAND}" -E create_symlink
        "libabce.so.1" "${_runtime_library_dir}/libabce.so"
    )
    set_property(GLOBAL APPEND PROPERTY _runtime_byproducts
        "${_runtime_library_dir}/libabce.so.1"
        "${_runtime_library_dir}/libabce.so"
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
endif()

# Install the same image and relative aliases used in the build tree. The
# package adds an ABCE facade; it never creates another runtime process state.
if(WIN32)
    install(FILES "${_runtime_library_dir}/rocddi_runtime.dll"
        DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT abce)
    install(FILES "${_runtime_library_dir}/rocddi_runtime.dll.lib"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT abce)
else()
    install(FILES
        "${_runtime_library_dir}/libhsa-runtime64.so.${_runtime_hsa_abi_version}"
        "${_runtime_library_dir}/libhsa-runtime64.so.${_runtime_hsa_abi_major}"
        "${_runtime_library_dir}/libhsa-runtime64.so"
        "${_runtime_library_dir}/libabce.so.1"
        "${_runtime_library_dir}/libabce.so"
        "${_runtime_library_dir}/libamdf.so.0"
        "${_runtime_library_dir}/libamdf.so"
        "${_runtime_library_dir}/libhsa_runtime64.so"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT abce)
endif()
include(CMakePackageConfigHelpers)
configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/ROCmRuntimeABCEProvider.cmake.in"
    "${_runtime_binary_dir}/ROCmRuntimeABCEProvider.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ROCmRuntimeABCE"
    PATH_VARS CMAKE_INSTALL_BINDIR CMAKE_INSTALL_LIBDIR)
install(FILES "${_runtime_binary_dir}/ROCmRuntimeABCEProvider.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/ROCmRuntimeABCE" COMPONENT abce)

if(BUILD_TESTING)
    add_test(NAME abce.cmake.consumers
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_LIST_DIR}/../ddi/libabce/tests/abi/check_cmake.py"
            --build "${_runtime_binary_dir}" --cmake "${CMAKE_COMMAND}")
    set_tests_properties(abce.cmake.consumers PROPERTIES LABELS "host;abi;build")
    add_test(NAME abce.abi.layout
        COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_LIST_DIR}/../ddi/libabce/tests/abi/check_layout.py")
    set_tests_properties(abce.abi.layout PROPERTIES LABELS "host;abi")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME abce.abi.exports
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_LIST_DIR}/../ddi/libabce/tests/abi/check_exports.py"
                "$<TARGET_FILE:runtime_shared_image>")
        set_tests_properties(abce.abi.exports PROPERTIES LABELS "host;abi")
    endif()
    add_executable(amdf_smoke_shared
        "${CMAKE_CURRENT_LIST_DIR}/../ddi/libamdf/tests/abi/native_c_smoke.c")
    target_compile_features(amdf_smoke_shared PRIVATE c_std_11)
    target_link_libraries(amdf_smoke_shared PRIVATE rocm_runtime::amdf_shared)
    add_test(NAME amdf.smoke.shared COMMAND amdf_smoke_shared)
    set_tests_properties(amdf.smoke.shared PROPERTIES LABELS "host;abi")
    if(WIN32)
        set_tests_properties(amdf.smoke.shared PROPERTIES
            ENVIRONMENT_MODIFICATION "PATH=path_list_prepend:${_runtime_library_dir}")
    endif()
endif()
