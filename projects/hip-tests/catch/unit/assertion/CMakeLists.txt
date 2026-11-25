# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

if(HIP_PLATFORM MATCHES "nvidia")
    set(TEST_SRC
        assert.cc
    )
    hip_add_exe_to_target(NAME AssertionTest
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
    set(TEST_SRC
        static_assert.cc
        assert.cc
    )
    hip_add_exe_to_target(NAME AssertionTest
                          TEST_SRC ${TEST_SRC}
                          TEST_TARGET_NAME build_tests
                          LINKER_LIBS hiprtc::hiprtc)
endif()
