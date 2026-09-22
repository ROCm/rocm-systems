# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
    __syncthreads.cc
    __syncthreads_count.cc
    __syncthreads_and.cc
    __syncthreads_or.cc
)

# the last argument linker libraries is required for this test but optional to the function
if(HIP_PLATFORM MATCHES "nvidia")
hip_add_exe_to_target(NAME SyncthreadsTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
hip_add_exe_to_target(NAME SyncthreadsTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS hiprtc::hiprtc)
endif()

if(UNIX)
file(GLOB NEGATIVE_TEST_SRC
     "__syncthreads_count_negative_kernels.cc"
     "__syncthreads_and_negative_kernels.cc"
     "__syncthreads_or_negative_kernels.cc"
    )

file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

#add_test(NAME Unit___syncthreads_count_Negative_Parameters
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         __syncthreads_count_negative_kernels.cc 2)
#
#add_test(NAME Unit___syncthreads_and_Negative_Parameters
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         __syncthreads_and_negative_kernels.cc 2)
#
#add_test(NAME Unit___syncthreads_or_Negative_Parameters
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         __syncthreads_or_negative_kernels.cc 2)
endif()
