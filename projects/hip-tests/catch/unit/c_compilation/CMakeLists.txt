# Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

set(TEST_SRC
    hipGetDeviceProp.cc
)

set (PLATFORM_DEFINE __HIP_PLATFORM_AMD__)
if(HIP_PLATFORM MATCHES "nvidia")
    set (PLATFORM_DEFINE __HIP_PLATFORM_NVIDIA__)
endif()

# Creating Custom object file
add_custom_target(devprop_c_custom
    COMMAND ${HIP_PATH}/bin/hipcc
            -c ${CMAKE_CURRENT_SOURCE_DIR}/hipGetDeviceProp.c
            -I${HIP_PATH}/include
            -D${PLATFORM_DEFINE}
            --hip-path=${HIP_PATH}
            -o hipGetDeviceProp.o
    BYPRODUCTS hipGetDeviceProp.o
)

add_library(devprop_c OBJECT IMPORTED)
set_property(TARGET devprop_c PROPERTY IMPORTED_OBJECTS "${CMAKE_CURRENT_BINARY_DIR}/hipGetDeviceProp.o")

hip_add_exe_to_target(NAME CTests
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests
                      LINKER_LIBS devprop_c)

add_dependencies(CTests devprop_c_custom)
