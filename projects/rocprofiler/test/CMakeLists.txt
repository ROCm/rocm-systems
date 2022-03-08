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

set ( EXE_NAME "rocprof-ctrl" )

if ( NOT DEFINED TEST_DIR )
  set ( TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
  project ( ${EXE_NAME} C CXX )
  ## Set build environment
  include ( env )
endif ()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

## C test
add_executable ( "c_test" ${TEST_DIR}/app/c_test.c )
target_include_directories ( "c_test" PRIVATE ${ROOT_DIR} $<TARGET_PROPERTY:hsa-runtime64::hsa-runtime64,INTERFACE_INCLUDE_DIRECTORIES> )

## Util sources
file( GLOB UTIL_SRC "${TEST_DIR}/util/*.cpp" )

## Standalone test sources
set ( ST_EXE_NAME "standalone_test" )
set ( ST_TST_SRC
  ${TEST_DIR}/app/standalone_test.cpp
  ${TEST_DIR}/ctrl/test_hsa.cpp
)

## Standalone intercept test sources
set ( STIN_EXE_NAME "stand_intercept_test" )
set ( STIN_TST_SRC
  ${TEST_DIR}/app/stand_intercept_test.cpp
  ${TEST_DIR}/ctrl/test_hsa.cpp
)

## Intercept test sources
set ( IN_EXE_NAME "intercept_test" )
set ( IN_TST_SRC
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
## Test kernel
set ( TEST_NAME simple_convolution )
set ( KERN_SRC ${TEST_DIR}/${TEST_NAME}/${TEST_NAME}.cpp )

## Building test kernels
add_custom_target( mytest
  COMMAND sh -xc "${TEST_DIR}/../bin/build_kernel.sh ${TEST_DIR}/${DUMMY_NAME}/${DUMMY_NAME} ${PROJECT_BINARY_DIR} '${ROCM_ROOT_DIR}' '${GPU_TARGETS}'"
  COMMAND sh -xc "${TEST_DIR}/../bin/build_kernel.sh ${TEST_DIR}/${TEST_NAME}/${TEST_NAME} ${PROJECT_BINARY_DIR} '${ROCM_ROOT_DIR}' '${GPU_TARGETS}'"
)

## Building standalone test executable
add_executable ( ${ST_EXE_NAME} ${ST_TST_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${ST_EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} )
target_link_libraries ( ${ST_EXE_NAME} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 hsakmt::hsakmt Threads::Threads dl )

## Building standalone intercept test executable
add_executable ( ${STIN_EXE_NAME} ${STIN_TST_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${STIN_EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} )
target_link_libraries ( ${STIN_EXE_NAME} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 hsakmt::hsakmt Threads::Threads dl )

## Building intercept test executable
add_library ( ${IN_EXE_NAME} SHARED ${IN_TST_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${IN_EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} )
target_link_libraries ( ${IN_EXE_NAME} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 hsakmt::hsakmt Threads::Threads dl )

## Building ctrl test executable
add_executable ( ${EXE_NAME} ${CTRL_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} )
target_link_libraries ( ${EXE_NAME} hsa-runtime64::hsa-runtime64 hsakmt::hsakmt Threads::Threads dl )
execute_process ( COMMAND sh -xc "cp ${TEST_DIR}/run.sh ${PROJECT_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "cp ${TEST_DIR}/tool/*.xml ${PROJECT_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "mkdir -p ${PROJECT_BINARY_DIR}/RESULTS" )

## Build test library
set ( TEST_LIB "rocprof-tool" )
set ( TEST_LIB_SRC ${TEST_DIR}/tool/tool.cpp ${UTIL_SRC} )
add_library ( ${TEST_LIB} SHARED ${TEST_LIB_SRC} )
target_include_directories ( ${TEST_LIB} PRIVATE ${TEST_DIR} ${ROOT_DIR} )
target_link_libraries ( ${TEST_LIB} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 Threads::Threads dl )

## Build memory test bench
add_custom_target( mbench
  COMMAND sh -xc "cp -r ${TEST_DIR}/memory_validation ${PROJECT_BINARY_DIR}/test/."
  COMMAND make -C "${PROJECT_BINARY_DIR}/test/memory_validation"
)

## Copy OCL test
execute_process ( COMMAND sh -xc "cp -r ${TEST_DIR}/ocl ${PROJECT_BINARY_DIR}/test/." )
