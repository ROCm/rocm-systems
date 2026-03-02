# Copyright (c) 2021 Advanced Micro Devices, Inc. All Rights Reserved.
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

if(CMAKE_BUILD_TYPE MATCHES "^Debug$")
    add_definitions(-DHT_LOG_ENABLE)
endif()

set(MAIN_SRC
    main.cc
    hip_test_context.cc
    hip_test_features.cc)
add_library(Main_Object EXCLUDE_FROM_ALL OBJECT ${MAIN_SRC})
set_source_files_properties(${MAIN_SRC} PROPERTIES LANGUAGE HIP)
if(ENABLE_SPIRV)
    target_compile_definitions(Main_Object PRIVATE ENABLE_SPIRV)
endif()
target_link_libraries(Main_Object PRIVATE Catch2::Catch2)
