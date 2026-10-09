# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(ROCmRuntimeAPI VERSION 0.1.0 LANGUAGES NONE)
endif()

# The implementation-only umbrella remains private to source-tree consumers.
add_library(rocm_runtime_api_headers INTERFACE)
target_include_directories(
    rocm_runtime_api_headers
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
add_library(rocm_runtime::runtime_headers ALIAS rocm_runtime_api_headers)

# Copy only this public family so the target cannot expose HSA or Linux UAPI.
file(GLOB _amdf_headers CONFIGURE_DEPENDS "include/amdf/*.h")
foreach(_header IN LISTS _amdf_headers)
    get_filename_component(_name "${_header}" NAME)
    configure_file("${_header}" "public/amdf/${_name}" COPYONLY)
endforeach()
add_library(rocm_runtime_amdf_headers INTERFACE)
set_target_properties(rocm_runtime_amdf_headers PROPERTIES EXPORT_NAME AMDF)
target_include_directories(
    rocm_runtime_amdf_headers
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/public>"
        "$<INSTALL_INTERFACE:include>"
)
add_library(ROCmRuntimeAPI::AMDF ALIAS rocm_runtime_amdf_headers)
include(CMakePackageConfigHelpers)
configure_package_config_file(
    cmake/ROCmRuntimeAPIConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/ROCmRuntimeAPIConfig.cmake"
    INSTALL_DESTINATION lib/cmake/ROCmRuntimeAPI
)
install(TARGETS rocm_runtime_amdf_headers EXPORT ROCmRuntimeAPITargets)
install(
    EXPORT ROCmRuntimeAPITargets
    NAMESPACE ROCmRuntimeAPI::
    DESTINATION lib/cmake/ROCmRuntimeAPI
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/ROCmRuntimeAPIConfig.cmake"
    DESTINATION lib/cmake/ROCmRuntimeAPI
)
install(FILES ${_amdf_headers} DESTINATION include/amdf)
install(FILES README.md DESTINATION share/ROCmRuntimeAPI)
