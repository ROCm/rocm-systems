# SPDX-License-Identifier: MIT
#
# Copyright (c) OdinLink-Five contributors
# https://github.com/Geramy/OdinLink-Five
#
# OdinLink Thunderbolt 5 / USB4 RCCL net plugin (librccl-net-odl_tb5.so)
#
# Can be built two ways:
#   1. Standalone: cmake -S . -B build && cmake --build build
#   2. As part of the main RCCL build via add_subdirectory(plugins/net/odinlink-tb5)
#
# Runtime requires the OdinLink kernel module (odl_tb5.ko) from
# https://github.com/Geramy/OdinLink-Five

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    cmake_minimum_required(VERSION 3.16)
    project(rccl-net-odl-tb5 LANGUAGES C)
    set(ODL_TB5_STANDALONE ON)
else()
    set(ODL_TB5_STANDALONE OFF)
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(rccl-net-odl_tb5 SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/plugin.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/odl_tb5_dev.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/odl_tb5_xfer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/odl_tb5_peer.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/odl_tb5_completion.c
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/odl_tb5_stream.c
)

target_include_directories(rccl-net-odl_tb5 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/nccl
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

target_compile_options(rccl-net-odl_tb5 PRIVATE -Wall -Wextra)

set_target_properties(rccl-net-odl_tb5 PROPERTIES
    OUTPUT_NAME "rccl-net-odl_tb5"
    PREFIX "lib"
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    C_STANDARD 11
    C_STANDARD_REQUIRED ON
)

target_link_libraries(rccl-net-odl_tb5 PRIVATE Threads::Threads ${CMAKE_DL_LIBS})

if(NOT ODL_TB5_STANDALONE)
    set_target_properties(rccl-net-odl_tb5 PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/test/unit/plugins
    )
endif()
