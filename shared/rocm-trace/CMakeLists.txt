# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

# ROCm trace contract headers: provider names and GUIDs, keywords, levels and the
# enable check shared by every ROCm library that writes trace events, and by the
# tools that read them. Header-only.
#
# In-tree consumers add this directory; others use the installed package:
#     find_package(rocm-trace CONFIG REQUIRED)
#     target_link_libraries(mytool PRIVATE rocm::trace_values)  # tools
#     target_link_libraries(myruntime PRIVATE rocm::trace)      # runtimes

cmake_minimum_required(VERSION 3.16)
project(rocm-trace VERSION 1.0.0 LANGUAGES C)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(ROCM_TRACE_TOP_LEVEL ON)
else()
    set(ROCM_TRACE_TOP_LEVEL OFF)
endif()

include(GNUInstallDirs)

# Prefixed, so a parent project's own BUILD_TESTS or install policy is not adopted.
option(ROCM_TRACE_BUILD_TESTS "Build the rocm-trace tests" OFF)
option(
    ROCM_TRACE_INSTALL
    "Install the headers and the CMake package"
    ${ROCM_TRACE_TOP_LEVEL}
)

# Names, GUIDs, keywords and levels: no dependencies, for runtimes and tools.
add_library(rocm-trace-values INTERFACE)
add_library(rocm::trace_values ALIAS rocm-trace-values)
set_target_properties(rocm-trace-values PROPERTIES EXPORT_NAME trace_values)
target_include_directories(
    rocm-trace-values
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Adds the enable check for runtimes. It needs a TraceLogging implementation (the
# Windows SDK, or the LTTng flavour of microsoft/tracelogging with liblttng-ust),
# which the runtime links: this target names none.
add_library(rocm-trace INTERFACE)
add_library(rocm::trace ALIAS rocm-trace)
set_target_properties(rocm-trace PROPERTIES EXPORT_NAME trace)
target_link_libraries(rocm-trace INTERFACE rocm-trace-values)

if(ROCM_TRACE_INSTALL)
    include(CMakePackageConfigHelpers)
    set(ROCM_TRACE_CMAKE_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/rocm-trace)

    install(TARGETS rocm-trace-values rocm-trace EXPORT rocm-trace-targets)
    install(
        DIRECTORY include/rocm
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT devel
    )
    install(
        EXPORT rocm-trace-targets
        NAMESPACE rocm::
        DESTINATION ${ROCM_TRACE_CMAKE_DIR}
        COMPONENT devel
    )

    configure_package_config_file(
        cmake/rocm-trace-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/rocm-trace-config.cmake
        INSTALL_DESTINATION ${ROCM_TRACE_CMAKE_DIR}
    )
    # Values are only ever added, so any newer 1.x provides everything an older 1.x had.
    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/rocm-trace-config-version.cmake
        COMPATIBILITY SameMajorVersion
        ARCH_INDEPENDENT
    )
    install(
        FILES
            ${CMAKE_CURRENT_BINARY_DIR}/rocm-trace-config.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/rocm-trace-config-version.cmake
        DESTINATION ${ROCM_TRACE_CMAKE_DIR}
        COMPONENT devel
    )
    # The same targets from the build tree, for a superbuild that has not installed yet.
    export(
        EXPORT rocm-trace-targets
        NAMESPACE rocm::
        FILE ${CMAKE_CURRENT_BINARY_DIR}/rocm-trace-targets.cmake
    )
endif()

if(ROCM_TRACE_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
