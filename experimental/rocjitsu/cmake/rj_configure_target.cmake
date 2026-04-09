# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Configure common properties on any rocjitsu target (library, executable, test).
#
# Sets include directories, compile flags, schema dependencies, and optionally
# device kernel definitions. Call after add_executable/add_library and
# target_link_libraries.
#
# Usage: rj_configure_target(<target> [OPTIONS...])
#   Options:
#     PRIVATE_INCLUDES  - add rocjitsu include/src dirs as PRIVATE (default)
#     GENERATED         - add the generated FlatBuffers header dir
#     DEVICE_KERNELS    - add HAS_DEVICE_KERNELS definition and dependency if available
#     SCHEMAS           - add dependency on flatbuffers_schemas target
#     WARNINGS          - add warning flags (-Wall/-Wextra/-Werror on GCC/Clang, /W4/WX on MSVC)
#     HIDDEN            - add -fvisibility=hidden (GCC/Clang only; MSVC hides by default)
#     CONFIG_DIRS       - add SCHEMA_DIR and CONFIG_DIR definitions
#
# Shorthand presets:
#     LIB     = PRIVATE_INCLUDES GENERATED WARNINGS HIDDEN SCHEMAS
#     TEST    = PRIVATE_INCLUDES GENERATED WARNINGS SCHEMAS CONFIG_DIRS DEVICE_KERNELS
#     TOOL    = PRIVATE_INCLUDES GENERATED WARNINGS SCHEMAS
function(rj_configure_target target)
    set(options
        PRIVATE_INCLUDES GENERATED DEVICE_KERNELS SCHEMAS
        WARNINGS HIDDEN CONFIG_DIRS
        LIB TEST TOOL
    )
    cmake_parse_arguments(ARG "${options}" "" "" ${ARGN})

    # Expand presets.
    if(ARG_LIB)
        set(ARG_PRIVATE_INCLUDES TRUE)
        set(ARG_GENERATED TRUE)
        set(ARG_WARNINGS TRUE)
        set(ARG_HIDDEN TRUE)
        set(ARG_SCHEMAS TRUE)
    endif()
    if(ARG_TEST)
        set(ARG_PRIVATE_INCLUDES TRUE)
        set(ARG_GENERATED TRUE)
        set(ARG_WARNINGS TRUE)
        set(ARG_SCHEMAS TRUE)
        set(ARG_CONFIG_DIRS TRUE)
        set(ARG_DEVICE_KERNELS TRUE)
    endif()
    if(ARG_TOOL)
        set(ARG_PRIVATE_INCLUDES TRUE)
        set(ARG_GENERATED TRUE)
        set(ARG_WARNINGS TRUE)
        set(ARG_SCHEMAS TRUE)
    endif()

    # Include directories.
    if(ARG_PRIVATE_INCLUDES)
        target_include_directories(${target} PRIVATE
            ${PROJECT_SOURCE_DIR}/lib/rocjitsu/include
            ${PROJECT_SOURCE_DIR}/lib/rocjitsu/src)
    endif()
    if(ARG_GENERATED)
        target_include_directories(${target} PRIVATE ${GENERATED_DIR})
    endif()

    # Compile flags.
    if(ARG_WARNINGS)
        if(MSVC)
            target_compile_options(${target} PRIVATE /W4 /WX)
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
        endif()
    endif()
    if(ARG_HIDDEN)
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            target_compile_options(${target} PRIVATE -fvisibility=hidden)
        endif()
        # MSVC hides all symbols by default; no flag needed.
    endif()

    # Definitions.
    if(ARG_CONFIG_DIRS)
        target_compile_definitions(${target} PRIVATE
            SCHEMA_DIR="${CMAKE_SOURCE_DIR}/schemas"
            CONFIG_DIR="${CMAKE_SOURCE_DIR}/configs")
    endif()

    # Schema dependency.
    if(ARG_SCHEMAS)
        add_dependencies(${target} flatbuffers_schemas)
    endif()

    # Device kernel support.
    if(ARG_DEVICE_KERNELS AND HAS_DEVICE_KERNELS)
        target_compile_definitions(${target} PRIVATE
            HAS_DEVICE_KERNELS=1
            KERNEL_DIR="${KERNEL_OUTPUT_DIR}")
        add_dependencies(${target} device_kernels)
    endif()
endfunction()
