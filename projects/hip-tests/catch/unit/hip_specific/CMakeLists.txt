# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Common Tests - Test independent of all platforms

set(TEST_SRC
    hip_hc_8pk.cc
    hip_check_VGPRs.cpp
)
hip_add_exe_to_target(NAME HipSpecificTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc::hiprtc
                      PROPERTY CXX_STANDARD 17)
