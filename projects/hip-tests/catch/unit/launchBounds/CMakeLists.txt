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

if(UNIX)
file(GLOB NEGATIVE_TEST_SRC
    "launch_bounds_compiler_error_kernels.cc"
    "launch_bounds_parse_error_kernels.cc")

file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

#if(HIP_PLATFORM MATCHES "nvidia")
# set(EXPECTED_ERRORS 2)
#elseif(HIP_PLATFORM MATCHES "amd")
# set(EXPECTED_ERRORS 3)
#endif()
#
#add_test(NAME Unit_Kernel_Launch_bounds_Negative_Parameters_CompilerError
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         launch_bounds_compiler_error_kernels.cc ${EXPECTED_ERRORS})
#
#if(HIP_PLATFORM MATCHES "amd")
#	add_test(NAME Unit_Kernel_Launch_bounds_Negative_Parameters_ParseError
#        COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#        ./src ${HIP_PLATFORM} ${HIP_PATH}
#        launch_bounds_parse_error_kernels.cc 0 0)
#endif()
endif()
