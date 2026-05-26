# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

if(NOT CMAKE_PROJECT_NAME OR CMAKE_PROJECT_NAME STREQUAL "simdojo")
    project(simdojo VERSION 1.0.0 LANGUAGES CXX)
    set(SIMDOJO_STANDALONE TRUE)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

include(GNUInstallDirs)
find_package(Threads REQUIRED)

add_library(
    simdojo
    src/component.cpp
    src/simulation.cpp
    src/topology.cpp
)
add_library(simdojo::simdojo ALIAS simdojo)

set_target_properties(simdojo PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(
    simdojo
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(simdojo PUBLIC Threads::Threads)

if(MSVC)
    target_compile_options(simdojo PRIVATE /W4 /WX)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(
        simdojo
        PRIVATE -Wall -Wextra -Wpedantic -Werror
    )
    # Hide symbols only for static builds; shared libraries need them exported.
    get_target_property(_simdojo_type simdojo TYPE)
    if(_simdojo_type STREQUAL "STATIC_LIBRARY")
        target_compile_options(simdojo PRIVATE -fvisibility=hidden)
    endif()
endif()

# --- Tests ---
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

# --- Install / export (standalone or when parent opts in) ---
if(SIMDOJO_STANDALONE OR SIMDOJO_INSTALL)
    include(CMakePackageConfigHelpers)

    install(
        TARGETS simdojo
        EXPORT simdojo-targets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    install(
        DIRECTORY include/simdojo
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING
        PATTERN "*.h"
    )

    install(
        EXPORT simdojo-targets
        FILE simdojo-targets.cmake
        NAMESPACE simdojo::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simdojo
    )

    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/simdojo-config.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/simdojo-config.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simdojo
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/simdojo-config-version.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    install(
        FILES
            "${CMAKE_CURRENT_BINARY_DIR}/simdojo-config.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/simdojo-config-version.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/simdojo
    )
endif()
