# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    launch_bounds.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
    hip_add_exe_to_target(NAME LaunchBoundsTest
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
    hip_add_exe_to_target(NAME LaunchBoundsTest
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS hiprtc::hiprtc)
endif()
