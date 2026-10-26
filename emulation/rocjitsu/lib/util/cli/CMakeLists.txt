# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.22)

# Keep the parser independently buildable without ISA generation or a ROCm SDK.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(util_cli LANGUAGES CXX)
    include(CTest)
    add_library(util INTERFACE)
    target_include_directories(
        util
        INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../include
    )
    if(BUILD_TESTING)
        find_package(GTest REQUIRED)
    endif()
endif()

add_library(util_cli STATIC cli.cpp)
target_include_directories(util_cli PUBLIC include)
target_link_libraries(util_cli PUBLIC util)
target_compile_features(util_cli PUBLIC cxx_std_20)
set_target_properties(
    util_cli
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(util_cli PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
