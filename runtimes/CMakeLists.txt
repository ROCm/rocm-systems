# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25)
project(ROCmRuntimes LANGUAGES C CXX)

if(NOT CMAKE_GENERATOR STREQUAL "Ninja" OR CMAKE_CONFIGURATION_TYPES)
    message(
        FATAL_ERROR
        "The initial runtimes build requires single-configuration Ninja"
    )
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Runtime build configuration" FORCE)
endif()
set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel
)

include(CTest)
# Plain C/C++ components and headers.
add_subdirectory(api-headers)

# ============================================================================
# Rust components: all Rust subdirectories must be bracketed by initialize/finalize.
include(cmake/RuntimeRust.cmake)
runtime_rust_initialize()
add_subdirectory(ddi)
include(cmake/SharedImage.cmake)
add_subdirectory(hsa)
runtime_rust_finalize()
# ============================================================================

if(BUILD_TESTING)
    add_test(
        NAME runtimes.build_helpers
        COMMAND
            "${Python3_EXECUTABLE}" -m unittest discover -s
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tests" -v
    )
    set_tests_properties(runtimes.build_helpers PROPERTIES LABELS "host;build")
    add_test(
        NAME runtimes.rust
        COMMAND
            ${_runtime_cargo_command} test --workspace --all-targets
            --all-features --frozen --profile "${_runtime_profile}"
    )
    set_tests_properties(
        runtimes.rust
        PROPERTIES
            WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            LABELS "host;rust"
            RESOURCE_LOCK runtime_cargo
    )
endif()
