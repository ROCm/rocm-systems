# ##############################################################################
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
#
# ##############################################################################

cmake_minimum_required(VERSION 3.15)

project(rocdxg_vram_budget_test LANGUAGES CXX)

enable_testing()

add_executable(rocdxg_vram_budget_test vram_budget_test.cpp)
target_compile_features(rocdxg_vram_budget_test PRIVATE cxx_std_11)
target_include_directories(
    rocdxg_vram_budget_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

add_test(NAME rocdxg_vram_budget_test COMMAND rocdxg_vram_budget_test)

# CmdbufFrameRing is pure arithmetic over the submission ordinal and includes
# nothing but <cstdint>, so this test needs neither the Windows SDK nor the DXG
# interop headers and runs on the build host.
add_executable(cmdbuf_frame_ring_test cmdbuf_frame_ring_test.cpp)

target_include_directories(cmdbuf_frame_ring_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_compile_features(cmdbuf_frame_ring_test PRIVATE cxx_std_17)

add_test(NAME cmdbuf_frame_ring COMMAND cmdbuf_frame_ring_test)
