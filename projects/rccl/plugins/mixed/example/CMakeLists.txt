#
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for more license information
#

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(rccl-mixed-example LANGUAGES C)
    set(MIXED_STANDALONE ON)
else()
    set(MIXED_STANDALONE OFF)
endif()

set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
)

# Create shared library
add_library(nccl-mixed-example SHARED ${SRC_FILES})

# Set include directories
target_include_directories(nccl-mixed-example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(nccl-mixed-example PRIVATE NCCL_OS_LINUX)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    target_compile_definitions(nccl-mixed-example PRIVATE NCCL_OS_WINDOWS)
endif()

# Set output name to match Makefile
set_target_properties(nccl-mixed-example PROPERTIES
    OUTPUT_NAME "rccl-mixed"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
)

if(NOT MIXED_STANDALONE)
    set_target_properties(nccl-mixed-example PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
    )
endif()
