################################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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
################################################################################

cmake_minimum_required ( VERSION 2.8.12 )
set ( CMAKE_VERBOSE_MAKEFILE TRUE CACHE BOOL "Verbose Output" FORCE )

set ( RUN_SCRIPT "${TEST_DIR}/run.sh" )
set ( TEST_HSA "${TEST_DIR}/hsa/test" )
execute_process ( COMMAND sh -xc "if [ ! -e ${TEST_DIR}/hsa ] ; then git clone https://github.com/ROCmSoftwarePlatform/hsa-class.git ${TEST_DIR}/hsa; fi" )
execute_process ( COMMAND sh -xc "if [ -e ${TEST_DIR}/hsa ] ; then cd ${TEST_DIR}/hsa && git pull --rebase; fi" )

## Util sources
file( GLOB UTIL_SRC "${TEST_HSA}/util/*.cpp" )

## Build test library
set ( TEST_LIB "tracer_tool" )
set ( TEST_LIB_SRC ${TEST_DIR}/tool/tracer_tool.cpp ${UTIL_SRC} )
add_library ( ${TEST_LIB} SHARED ${TEST_LIB_SRC} )
target_include_directories ( ${TEST_LIB} PRIVATE ${TEST_HSA} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} )
target_link_libraries ( ${TEST_LIB} ${ROCTRACER_TARGET} ${HSA_RUNTIME_LIB} c stdc++ dl pthread rt )

set ( TEST_DIR ${TEST_HSA} )
add_subdirectory ( ${TEST_DIR} ${PROJECT_BINARY_DIR}/test/hsa )
execute_process ( COMMAND sh -xc "cp ${RUN_SCRIPT} ${PROJECT_BINARY_DIR}" )
