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

set ( EXE_NAME "ctrl" )

if ( NOT DEFINED TEST_DIR )
  set ( TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
  project ( ${EXE_NAME} )
  ## Set build environment
  include ( env )
endif ()

## Util sources
file( GLOB UTIL_SRC "${TEST_DIR}/util/*.cpp" )

## Standalone test sources
set ( STEXE_NAME "standalone_test" )
set ( STTST_SRC
  ${TEST_DIR}/app/standalone_test.cpp
  ${TEST_DIR}/ctrl/test_hsa.cpp
)

## Intercept test sources
set ( INEXE_NAME "intercept_test" )
set ( INTST_SRC
  ${TEST_DIR}/app/intercept_test.cpp
  ${TEST_DIR}/ctrl/test_hsa.cpp
)

## Test control sources
set ( CTRL_SRC
  ${TEST_DIR}/app/test.cpp
  ${TEST_DIR}/ctrl/test_hsa.cpp
)

## Dummy kernel
set ( DUMMY_NAME dummy_kernel )
execute_process ( COMMAND sh -xc "${TEST_DIR}/../bin/build_kernel.sh ${TEST_DIR}/${DUMMY_NAME}/${DUMMY_NAME} ${PROJECT_BINARY_DIR}" )

## Test kernel
set ( TEST_NAME simple_convolution )
set ( KERN_SRC ${TEST_DIR}/${TEST_NAME}/${TEST_NAME}.cpp )
execute_process ( COMMAND sh -xc "${TEST_DIR}/../bin/build_kernel.sh ${TEST_DIR}/${TEST_NAME}/${TEST_NAME} ${PROJECT_BINARY_DIR}" )

## Building standalone test executable
add_executable ( ${STEXE_NAME} ${STTST_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${STEXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} )
target_link_libraries( ${STEXE_NAME} ${ROCPROFILER_TARGET} ${HSA_RUNTIME_LIB} ${HSA_KMT_LIB} c stdc++ dl pthread rt )

## Building intercept test executable
add_library ( ${INEXE_NAME} SHARED ${INTST_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${INEXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} )
target_link_libraries( ${INEXE_NAME} ${ROCPROFILER_TARGET} ${HSA_RUNTIME_LIB} ${HSA_KMT_LIB} c stdc++ dl pthread rt )

## Building ctrl test executable
add_executable ( ${EXE_NAME} ${CTRL_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} )
target_link_libraries( ${EXE_NAME} ${HSA_RUNTIME_LIB} ${HSA_KMT_LIB} c stdc++ dl pthread rt )
execute_process ( COMMAND sh -xc "cp ${TEST_DIR}/run.sh ${PROJECT_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "cp ${TEST_DIR}/tool/*.xml ${PROJECT_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "mkdir -p ${PROJECT_BINARY_DIR}/RESULTS" )

## Build test library
set ( TEST_LIB "tool" )
set ( TEST_LIB_SRC ${TEST_DIR}/tool/tool.cpp ${UTIL_SRC} )
add_library ( ${TEST_LIB} SHARED ${TEST_LIB_SRC} )
target_include_directories ( ${TEST_LIB} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} )
target_link_libraries ( ${TEST_LIB} ${ROCPROFILER_TARGET} ${HSA_RUNTIME_LIB} c stdc++ dl pthread rt )
