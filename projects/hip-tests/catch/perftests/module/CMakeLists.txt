# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(TEST_SRC
	hipModuleLoad.cc
)


add_custom_target(saxpy.code
                  COMMAND ${CMAKE_CXX_COMPILER} --genco ${OFFLOAD_ARCH_STR}
                  ${CMAKE_CURRENT_SOURCE_DIR}/saxpy.cpp
                  -o ${CMAKE_CURRENT_BINARY_DIR}/../../perftests/module/saxpy.code
                  -I${HIP_PATH}/include/ --rocm-path=${ROCM_PATH}
                  -I${CMAKE_CURRENT_SOURCE_DIR}/../../include)

set_property(GLOBAL APPEND PROPERTY G_INSTALL_CUSTOM_TARGETS
  ${CMAKE_CURRENT_BINARY_DIR}/saxpy.code
)
if(HIP_PLATFORM MATCHES "amd")
    set(RTCLIB "hiprtc")
else()
    set(RTCLIB "nvrtc")
endif()

hip_add_exe_to_target(NAME perfModuleTest
  TEST_SRC ${TEST_SRC}
  TEST_TARGET_NAME perf_test
  LINKER_LIBS ${RTCLIB}
  COMMON_SHARED_SRC ${COMMON_SHARED_SRC}
  COMPILE_OPTIONS -std=c++17)

add_dependencies(perfModuleTest saxpy.code)

