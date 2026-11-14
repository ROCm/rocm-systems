# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_library(rocm_runtime_api_headers INTERFACE)
target_include_directories(
    rocm_runtime_api_headers
    SYSTEM
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
add_library(rocm_runtime::runtime_headers ALIAS rocm_runtime_api_headers)
set_target_properties(rocm_runtime_api_headers PROPERTIES EXPORT_NAME runtime_headers)

add_library(rocm_runtime_abce_headers INTERFACE)
add_library(rocm_runtime::abce_headers ALIAS rocm_runtime_abce_headers)
set_target_properties(rocm_runtime_abce_headers PROPERTIES EXPORT_NAME abce_headers)
target_link_libraries(rocm_runtime_abce_headers INTERFACE rocm_runtime::runtime_headers)
target_compile_definitions(rocm_runtime_abce_headers INTERFACE ABCE_ENABLE_IMPLEMENTATION=1)

option(ABCE_HEADER_ONLY "Use the ABCE header implementation" ON)
add_library(rocm_runtime_abce INTERFACE)
add_library(rocm_runtime::abce ALIAS rocm_runtime_abce)
target_link_libraries(rocm_runtime_abce INTERFACE rocm_runtime::runtime_headers)
target_compile_definitions(rocm_runtime_abce INTERFACE
    ABCE_ENABLE_IMPLEMENTATION=$<BOOL:${ABCE_HEADER_ONLY}>)
if(NOT ABCE_HEADER_ONLY)
    # The full runtime defines this provider later. CMake diagnoses a missing
    # provider at generation time in headers-only parent projects.
    target_link_libraries(rocm_runtime_abce INTERFACE rocm_runtime::abce_shared)
endif()

# Export the concrete targets. The facade is selected by the consuming project
# at find_package time, independently of the option used to build this package.
set(_abce_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/ROCmRuntimeABCE")
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT abce)
install(TARGETS rocm_runtime_api_headers rocm_runtime_abce_headers
    EXPORT ROCmRuntimeABCEHeaders COMPONENT abce)
install(EXPORT ROCmRuntimeABCEHeaders FILE ROCmRuntimeABCEHeaders.cmake
    NAMESPACE rocm_runtime:: DESTINATION "${_abce_package_dir}" COMPONENT abce)
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ROCmRuntimeABCEConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ROCmRuntimeABCEConfig.cmake"
    INSTALL_DESTINATION "${_abce_package_dir}")
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ROCmRuntimeABCEConfigVersion.cmake"
    VERSION 1.0.0 COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ROCmRuntimeABCEConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ROCmRuntimeABCEConfigVersion.cmake"
    DESTINATION "${_abce_package_dir}" COMPONENT abce)

option(ABCE_BUILD_TESTS "Build the GPU-free ABCE unit tests" OFF)
if(ABCE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests/abce)
endif()
