# ##############################################################################
#
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
#
# ##############################################################################

# CmdbufFrameRing is pure arithmetic over the submission ordinal and includes
# nothing but <cstdint>, so this test needs neither the Windows SDK nor the DXG
# interop headers and runs on the build host.
add_executable(cmdbuf_frame_ring_test cmdbuf_frame_ring_test.cpp)

target_include_directories(cmdbuf_frame_ring_test
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_compile_features(cmdbuf_frame_ring_test PRIVATE cxx_std_17)

add_test(NAME cmdbuf_frame_ring COMMAND cmdbuf_frame_ring_test)
