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
if(UNIX)
file(GLOB NEGATIVE_TEST_SRC
     "hip_hc_8pk_negative_kernels.cc" )

file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

#add_test(NAME Unit_Device__hip_hc_8pk_Negative
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         hip_hc_8pk_negative_kernels.cc 90)
endif()
