# Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Common Tests - Test independent of all platforms

set(TEST_SRC
    memcpy.cc
    memset.cc
)

if(HIP_PLATFORM MATCHES "nvidia")
  set(LINKER_LIBS nvrtc)
elseif(HIP_PLATFORM MATCHES "amd")
  set(LINKER_LIBS hiprtc)
endif()

if(HIP_PLATFORM MATCHES "amd")
  hip_add_exe_to_target(NAME DeviceMemoryTest
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    LINKER_LIBS ${LINKER_LIBS}
                    PROPERTY CXX_STANDARD 17)
elseif (HIP_PLATFORM MATCHES "nvidia")
  hip_add_exe_to_target(NAME DeviceMemoryTest
                    TEST_SRC ${TEST_SRC}
                    TEST_TARGET_NAME build_tests
                    LINKER_LIBS ${LINKER_LIBS}
                    COMPILE_OPTIONS -std=c++17)
endif()

if(UNIX)
file(GLOB NEGATIVE_TEST_SRC
     "memcpy_negative_kernels.cc"
     "memset_negative_kernels.cc"
    )

file(COPY ${NEGATIVE_TEST_SRC} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src)
set_property(GLOBAL APPEND PROPERTY G_INSTALL_SRC_FILES ${NEGATIVE_TEST_SRC})

#add_test(NAME Unit_Device_memcpy_Negative
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         memcpy_negative_kernels.cc 4)
#
#add_test(NAME Unit_Device_memset_Negative
#         COMMAND ${Python3_EXECUTABLE} ../compileAndCaptureOutput.py
#         ./src ${HIP_PLATFORM} ${HIP_PATH}
#         memset_negative_kernels.cc 4)
endif()