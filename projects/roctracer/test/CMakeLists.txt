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

cmake_minimum_required ( VERSION 3.5.0 )
set ( CMAKE_VERBOSE_MAKEFILE TRUE CACHE BOOL "Verbose Output" FORCE )

if ( NOT DEFINED ROOT_DIR )
  MESSAGE ( FATAL_ERROR "ROOT_DIR is not defined" )
endif ()
if ( NOT DEFINED INC_PATH )
  set ( INC_PATH "${ROOT_DIR}/inc" )
endif ()
if ( NOT DEFINED LIB_PATH )
  set ( LIB_PATH "${PROJECT_BINARY_DIR}" )
endif ()
if ( NOT DEFINED TEST_DIR )
  set ( TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
  ## Adding default path cmake modules
  list ( APPEND CMAKE_MODULE_PATH "${ROOT_DIR}/cmake_modules" )
  ## Include common cmake modules
  include ( utils )
  ## Set build environment
  include ( env )
endif ()

## Path to HSA test
set ( HSA_TEST_DIR "${TEST_DIR}/hsa/test" )
set ( HSA_REV "f8b3870" )

## test run script
set ( RUN_SCRIPT "${TEST_DIR}/run.sh" )

## build HIP tests
set ( GEN_INC_DIR ${PROJECT_BINARY_DIR}/inc )
set ( INC_PATH "${INC_PATH} ${GEN_INC_DIR}" )
set ( TEST_ENV ROCM_PATH=${ROCM_ROOT_DIR} HSA_PATH=${ROCM_ROOT_DIR}/hsa INC_PATH=${INC_PATH} LIB_PATH=${LIB_PATH} HIPCC_VERBOSE=3 )
add_custom_target( mytest
  COMMAND ${TEST_ENV} make -C "${TEST_DIR}/MatrixTranspose" -j1
  COMMAND sh -xc "cp ${TEST_DIR}/MatrixTranspose/MatrixTranspose ${PROJECT_BINARY_DIR}/test"
  COMMAND ${TEST_ENV} make -C "${TEST_DIR}/MatrixTranspose_test" -j1
  COMMAND sh -xc "cp ${TEST_DIR}/MatrixTranspose_test/MatrixTranspose ${PROJECT_BINARY_DIR}/test/MatrixTranspose_test"
  COMMAND HIP_API_ACTIVITY_ON=1 ${TEST_ENV} make -C "${TEST_DIR}/MatrixTranspose_test" -j1
  COMMAND sh -xc "cp ${TEST_DIR}/MatrixTranspose_test/MatrixTranspose ${PROJECT_BINARY_DIR}/test/MatrixTranspose_hipaact_test"
  COMMAND MGPU_TEST=1 ${TEST_ENV} make -C "${TEST_DIR}/MatrixTranspose_test" -j1
  COMMAND sh -xc "cp ${TEST_DIR}/MatrixTranspose_test/MatrixTranspose ${PROJECT_BINARY_DIR}/test/MatrixTranspose_mgpu"
  COMMAND C_TEST=1 ${TEST_ENV} make -C "${TEST_DIR}/MatrixTranspose_test" -j1
  COMMAND sh -xc "cp ${TEST_DIR}/MatrixTranspose_test/MatrixTranspose ${PROJECT_BINARY_DIR}/test/MatrixTranspose_ctest"
# copy traces
  COMMAND sh -xc "cp ${TEST_DIR}/golden_traces/*_trace.txt ${PROJECT_BINARY_DIR}/test/"
  COMMAND sh -xc "cp ${TEST_DIR}/golden_traces/tests_trace_cmp_levels.txt ${PROJECT_BINARY_DIR}/test/"
)

## Build HSA test
execute_process ( COMMAND sh -xc "if [ ! -e ${TEST_DIR}/hsa ] ; then git clone https://github.com/ROCmSoftwarePlatform/hsa-class.git ${TEST_DIR}/hsa; fi" )
execute_process ( COMMAND sh -xc "if [ -e ${TEST_DIR}/hsa ] ; then cd ${TEST_DIR}/hsa && git fetch origin && git checkout ${HSA_REV}; fi" )
set ( TMP ${TEST_DIR} )
set ( TEST_DIR ${HSA_TEST_DIR} )
add_subdirectory ( ${HSA_TEST_DIR} ${PROJECT_BINARY_DIR}/test/hsa )
set ( TEST_DIR ${TMP} )

## Util sources
file( GLOB UTIL_SRC "${HSA_TEST_DIR}/util/*.cpp" )

if ( DEFINED ROCTRACER_TARGET )
  ## Build test library
  set ( TEST_LIB "tracer_tool" )
  set ( TEST_LIB_SRC ${TEST_DIR}/tool/tracer_tool.cpp ${UTIL_SRC} )
  add_library ( ${TEST_LIB} SHARED ${TEST_LIB_SRC} )
  target_include_directories ( ${TEST_LIB} PRIVATE ${HSA_TEST_DIR} ${ROOT_DIR} ${ROOT_DIR}/inc ${HSA_RUNTIME_INC_PATH} ${HIP_INC_DIR} ${HSA_KMT_INC_PATH} ${GEN_INC_DIR} )
  target_link_libraries ( ${TEST_LIB} ${ROCTRACER_TARGET} ${HSA_RUNTIME_LIB} c stdc++ dl pthread rt numa )
  install ( TARGETS ${TEST_LIB} LIBRARY DESTINATION ${DEST_NAME}/tool )
endif ()

## Build hsaco_test.cpp referenc test
set ( CO_LIB_NAME "hsaco_test" )
set ( CO_LIB_SRC ${TEST_DIR}/app/hsaco_test.cpp )
add_library ( ${CO_LIB_NAME} SHARED ${CO_LIB_SRC} )
target_include_directories ( ${CO_LIB_NAME} PRIVATE ${HSA_RUNTIME_INC_PATH} )
target_link_libraries ( ${CO_LIB_NAME} ${HSA_RUNTIME_LIB} c stdc++ )

## Build codeobj event test
set ( CO_LIB_NAME "codeobj_test" )
set ( CO_LIB_SRC ${TEST_DIR}/app/codeobj_test.cpp )
add_library ( ${CO_LIB_NAME} SHARED ${CO_LIB_SRC} )
target_include_directories ( ${CO_LIB_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${ROOT_DIR}/inc ${GEN_INC_DIR} ${HSA_RUNTIME_INC_PATH} ${ROCM_INC_PATH} )
target_link_libraries ( ${CO_LIB_NAME} ${ROCTRACER_TARGET} c stdc++ )

## copying run script
execute_process ( COMMAND sh -xc "cp ${RUN_SCRIPT} ${PROJECT_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "ln -s run.sh ${PROJECT_BINARY_DIR}/run_ci.sh" )
## copying tests output check script
execute_process ( COMMAND sh -xc "cp ${ROOT_DIR}/script/check_trace.py ${PROJECT_BINARY_DIR}/test/." )
