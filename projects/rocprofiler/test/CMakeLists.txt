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
cmake_minimum_required(VERSION 3.16.0)
set ( EXE_NAME "rocprof-ctrl" )

if ( NOT DEFINED TEST_DIR )
  set ( TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
  project ( ${EXE_NAME} C CXX )
  ## Set build environment
  include ( env )
endif ()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_definitions(-DAMD_INTERNAL_BUILD=1)

find_package(Threads REQUIRED)
find_package(hsa-runtime64 REQUIRED CONFIG HINTS ${CMAKE_INSTALL_PREFIX} PATHS ${ROCM_PATH})
get_property(HSA_RUNTIME_INCLUDE_DIRECTORIES TARGET hsa-runtime64::hsa-runtime64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
find_file(HSA_H hsa.h
  PATHS ${HSA_RUNTIME_INCLUDE_DIRECTORIES}
  PATH_SUFFIXES hsa
  NO_DEFAULT_PATH
  REQUIRED)
get_filename_component(HSA_RUNTIME_INC_PATH ${HSA_H} DIRECTORY)
include_directories(${HSA_RUNTIME_INC_PATH})

## C test
add_executable ( "c_test" ${TEST_DIR}/app/c_test.c )
target_include_directories ( "c_test" PRIVATE ${ROOT_DIR} ${PROJECT_SOURCE_DIR}/include $<TARGET_PROPERTY:hsa-runtime64::hsa-runtime64,INTERFACE_INCLUDE_DIRECTORIES> )
install(TARGETS "c_test" DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1/test COMPONENT tests)

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
# add_custom_target( mytest
#   COMMAND sh -xc "${TEST_DIR}/../bin/build_kernel.sh ${TEST_DIR}/${DUMMY_NAME}/${DUMMY_NAME} ${PROJECT_BINARY_DIR} '${ROCM_ROOT_DIR}' '${GPU_TARGETS}'"
#   COMMAND sh -xc "${TEST_DIR}/../bin/build_kernel.sh ${TEST_DIR}/${TEST_NAME}/${TEST_NAME} ${PROJECT_BINARY_DIR} '${ROCM_ROOT_DIR}' '${GPU_TARGETS}'"
# )

find_package(Clang REQUIRED CONFIG
             PATHS ${ROCM_ROOT_DIR}
             PATH_SUFFIXES "llvm/lib/cmake/clang")

## Build the hsa (standalone) simple_convolution test
function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
    "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 --hip-device-lib-path=${ROCM_ROOT_DIR}/amdgcn/bitcode
    -Wl,--build-id=sha1 -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -mcode-object-version=4 -o ${OUTPUT_FILE} ${INPUT_FILE}")
  add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/${OUTPUT_FILE}
    COMMAND clang ${CLANG_ARG_LIST}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    DEPENDS ${INPUT_FILE} clang
    COMMENT "Building ${OUTPUT_FILE}..."
    VERBATIM)
  install(FILES ${PROJECT_BINARY_DIR}/${OUTPUT_FILE} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1 COMPONENT tests)
  set(HSACO_TARGET_LIST ${HSACO_TARGET_LIST} ${PROJECT_BINARY_DIR}/${OUTPUT_FILE} PARENT_SCOPE)
endfunction(generate_hsaco)

separate_arguments(GPU_TARGETS)
list(LENGTH GPU_TARGETS list_count)
if (${list_count} LESS_EQUAL 1)
  string(REPLACE " " ";" GPU_LIST ${GPU_TARGETS})
  string(REPLACE "," ";" GPU_LIST ${GPU_TARGETS})
else()
  set(GPU_LIST ${GPU_TARGETS})
endif()
foreach(target_id ${GPU_LIST})
  ## generate kernel bitcodes
  generate_hsaco(${target_id} ${TEST_DIR}/${TEST_NAME}/${TEST_NAME}.cl ${target_id}_SimpleConvolution.hsaco)
  generate_hsaco(${target_id} ${TEST_DIR}/${DUMMY_NAME}/${DUMMY_NAME}.cl ${target_id}_DummyKernel.hsaco)
endforeach(target_id)
add_custom_target(test DEPENDS ${HSACO_TARGET_LIST})
add_custom_target(mytest DEPENDS ${HSACO_TARGET_LIST})
add_custom_command(TARGET mytest POST_BUILD
  DEPENDS mytest
  COMMAND $<$<CONFIG:release>:${CPACK_STRIP_EXECUTABLE}>
  ARGS ${HSACO_TARGET_LIST})

## Building standalone test executable
add_executable ( ${ST_EXE_NAME} ${ST_TST_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${ST_EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${PROJECT_SOURCE_DIR}/include )
target_link_libraries ( ${ST_EXE_NAME} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 hsakmt::hsakmt Threads::Threads stdc++fs dl )
install(TARGETS ${ST_EXE_NAME} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1/test COMPONENT tests)

## Building standalone intercept test executable
add_executable ( ${STIN_EXE_NAME} ${STIN_TST_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${STIN_EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${PROJECT_SOURCE_DIR}/include )
target_link_libraries ( ${STIN_EXE_NAME} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 hsakmt::hsakmt Threads::Threads stdc++fs dl )
install(TARGETS ${STIN_EXE_NAME} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1/test COMPONENT tests)

## Building intercept test executable
add_library ( ${IN_EXE_NAME} SHARED ${IN_TST_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${IN_EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${PROJECT_SOURCE_DIR}/include )
target_link_libraries ( ${IN_EXE_NAME} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 hsakmt::hsakmt Threads::Threads stdc++fs dl )
install(TARGETS ${IN_EXE_NAME} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1/test COMPONENT tests)

## Building ctrl test executable
add_executable ( ${EXE_NAME} ${CTRL_SRC} ${UTIL_SRC} ${KERN_SRC} )
target_include_directories ( ${EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${PROJECT_SOURCE_DIR}/include )
target_link_libraries ( ${EXE_NAME} hsa-runtime64::hsa-runtime64 hsakmt::hsakmt Threads::Threads stdc++fs dl )
execute_process ( COMMAND sh -xc "cp ${TEST_DIR}/run.sh ${PROJECT_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "cp ${TEST_DIR}/tool/*.xml ${PROJECT_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "mkdir -p ${PROJECT_BINARY_DIR}/RESULTS" )
## TODO(aelwazir): Should be replaced by the current location in the main CMakeLists.txt
install(TARGETS ${EXE_NAME} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1/test COMPONENT tests)

## Build test library
set ( TEST_LIB "rocprof-tool" )
set ( TEST_LIB_SRC ${TEST_DIR}/tool/tool.cpp ${UTIL_SRC} )
add_library ( ${TEST_LIB} SHARED ${TEST_LIB_SRC} )
target_include_directories ( ${TEST_LIB} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${PROJECT_SOURCE_DIR}/include )
target_link_libraries ( ${TEST_LIB} ${ROCPROFILER_TARGET} hsa-runtime64::hsa-runtime64 Threads::Threads atomic dl )
## TODO(aelwazir): Should be replaced by the current location in the main CMakeLists.txt
install(TARGETS ${TEST_LIB} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1/test COMPONENT tests)

## Build memory test bench
add_custom_target( mbench
  COMMAND sh -xc "cp -r ${TEST_DIR}/memory_validation ${PROJECT_BINARY_DIR}/test/."
  COMMAND make -C "${PROJECT_BINARY_DIR}/test/memory_validation"
)

## Copy OCL test
execute_process ( COMMAND sh -xc "cp -r ${TEST_DIR}/ocl ${PROJECT_BINARY_DIR}/test/." )
install(DIRECTORY ${TEST_DIR}/ocl DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1/test COMPONENT tests)

file(GLOB TEST_XML_FILES ${CMAKE_CURRENT_SOURCE_DIR}/tool/*.xml)
install(
  FILES ${TEST_XML_FILES}
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1
  COMPONENT tests)

install(
  FILES ${PROJECT_BINARY_DIR}/run.sh
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/tests-v1
  PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE
  COMPONENT tests)