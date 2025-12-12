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

if(UNIX)
file(GLOB NEGATIVE_TEST_SRC
     "static_assert_kernels_positive.cc"
     "static_assert_kernels_negative.cc")

file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

#add_test(NAME Unit_StaticAssert_Positive_Basic
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         static_assert_kernels_positive.cc 2)
#
#add_test(NAME Unit_StaticAssert_Negative_Basic
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         static_assert_kernels_negative.cc 2)
endif()
