cmake_minimum_required ( VERSION 3.5.0 )
set ( CMAKE_VERBOSE_MAKEFILE TRUE CACHE BOOL "Verbose Output" FORCE )

set ( EXE_NAME "ctrl" )

if ( NOT DEFINED TEST_DIR )
  set ( TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR} )
  project ( ${EXE_NAME} )
  ## Set build environment
  include ( env )
  set ( ROCPROFILER_TARGET "rocprofiler64" )
endif ()

## Util sources
file( GLOB UTIL_SRC "${TEST_DIR}/util/*.cpp" )

## Test control sources
set ( CTRL_SRC
  ${TEST_DIR}/ctrl/test.cpp
  ${TEST_DIR}/ctrl/test_hsa.cpp
)

## Test kernels sources
set ( TEST_NAME simple_convolution )
set ( KERN_SRC ${TEST_DIR}/${TEST_NAME}/${TEST_NAME}.cpp )
execute_process ( COMMAND sh -xc "cp ${TEST_DIR}/${TEST_NAME}/*.hsaco ${PROJECT_BINARY_DIR}" )

## Building test executable
add_executable ( ${EXE_NAME} ${KERN_SRC} ${CTRL_SRC} ${UTIL_SRC} )
target_include_directories ( ${EXE_NAME} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} )
target_link_libraries( ${EXE_NAME} ${ROCPROFILER_TARGET} ${HSA_RUNTIME_LIB} c stdc++ dl pthread rt atomic )
execute_process ( COMMAND sh -xc "cp ${TEST_DIR}/run.sh ${PROJECT_BINARY_DIR}" )
execute_process ( COMMAND sh -xc "cp ${TEST_DIR}/*.xml ${PROJECT_BINARY_DIR}" )

## Build test library
set ( TEST_LIB "tool" )
set ( TEST_LIB_SRC ${TEST_DIR}/ctrl/tool.cpp )
add_library ( ${TEST_LIB} SHARED ${TEST_LIB_SRC} )
target_include_directories ( ${TEST_LIB} PRIVATE ${TEST_DIR} ${ROOT_DIR} ${HSA_RUNTIME_INC_PATH} )
target_link_libraries( ${TEST_LIB} ${ROCPROFILER_TARGET} ${HSA_RUNTIME_LIB} c stdc++ dl pthread rt atomic )

