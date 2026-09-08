#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for more license information
#

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(rccl-gin-example LANGUAGES C)
    set(GIN_EXAMPLE_STANDALONE ON)
else()
    set(GIN_EXAMPLE_STANDALONE OFF)
endif()

set(SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
)

# Create shared library
add_library(nccl-gin-example SHARED ${SRC_FILES})

# Set include directories
target_include_directories(nccl-gin-example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
)

# Set output name to match Makefile
set_target_properties(nccl-gin-example PROPERTIES
    OUTPUT_NAME "rccl-gin-example"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
)

if(NOT GIN_EXAMPLE_STANDALONE)
    set_target_properties(nccl-gin-example PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
    )
endif()
