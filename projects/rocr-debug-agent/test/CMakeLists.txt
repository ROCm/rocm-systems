################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2018-2020, Advanced Micro Devices, Inc. All rights reserved.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal with the Software without restriction, including without limitation
## the rights to use, copy, modify, merge, publish, distribute, sublicense,
## and/or sell copies of the Software, and to permit persons to whom the
## Software is furnished to do so, subject to the following conditions:
##
##  - Redistributions of source code must retain the above copyright notice,
##    this list of conditions and the following disclaimers.
##  - Redistributions in binary form must reproduce the above copyright
##    notice, this list of conditions and the following disclaimers in
##    the documentation and/or other materials provided with the distribution.
##  - Neither the names of Advanced Micro Devices, Inc,
##    nor the names of its contributors may be used to endorse or promote
##    products derived from this Software without specific prior written
##    permission.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
## THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
## OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
## ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
## DEALINGS WITH THE SOFTWARE.
##
################################################################################

cmake_minimum_required(VERSION 3.12)

project(rocm-debug-agent-test)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

# CMAKE_MODULE_PATH set to default
set(CMAKE_MODULE_PATH "/opt/rocm/lib/cmake/hip" CACHE STRING "HIP CMAKE Module Path")
find_package(HIP MODULE QUIET)
# Mandate correct CMAKE_MODULE_PATH to be provided if Module not found in default
if( NOT HIP_FOUND )
  message(FATAL_ERROR
          "HIP Module not found in Module Path ${CMAKE_MODULE_PATH}, \
           Please pass the correct Module search Path \
           'cmake -DCMAKE_MODULE_PATH=/opt/rocm-<version>/lib/cmake/hip'")
  return()
endif()

set(CMAKE_HIP_ARCHITECTURES OFF)

set(MAIN_SOURCES
	debug_agent_test.cpp
	vector_add_assert_trap.cpp
	sigquit.cpp
	print_all_waves.cpp
	snapshot_objfile_on_load.cpp
	vector_add_normal.cpp
	vector_add_memory_fault.cpp
	save_code_objects.cpp)
set(NO_DEBUG_SOURCES
	vector_add_assert_trap_no_debug_info.cpp)
set(SOURCES ${MAIN_SOURCES} ${NO_DEBUG_SOURCES})

set_source_files_properties(${SOURCES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

add_custom_command(
  OUTPUT "snapshot_objfile_on_load.hipfb"
  COMMAND ${HIP_HIPCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/snapshot_objfile_on_load.cpp --genco -o snapshot_objfile_on_load.hipfb
  DEPENDS "snapshot_objfile_on_load.cpp"
  COMMENT "Building snapshot_objfile_on_load.hipfb"
  VERBATIM)

add_custom_command(
  OUTPUT "save_code_objects.hipfb"
  COMMAND ${HIP_HIPCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/save_code_objects.cpp --genco -o save_code_objects.hipfb
  DEPENDS "save_code_objects.cpp"
  COMMENT "Building save_code_objects.hipfb"
  VERBATIM)

hip_add_library(rocm-debug-agent-no-dbgapifo ${NO_DEBUG_SOURCES})

hip_add_executable(rocm-debug-agent-test ${MAIN_SOURCES} ${NODEBUG_OBJ} CLANG_OPTIONS -ggdb)
target_link_libraries(rocm-debug-agent-test rocm-debug-agent-no-dbgapifo)

add_custom_target(snapshot_objfile_on_load-tgt DEPENDS snapshot_objfile_on_load.hipfb)
add_custom_target(save_code_objects-tgt DEPENDS save_code_objects.hipfb)

add_dependencies(rocm-debug-agent-test snapshot_objfile_on_load-tgt save_code_objects-tgt)

file(GLOB HEADERS "*.h")

install(FILES CMakeLists.txt run-test.py ${SOURCES} ${HEADERS}
  DESTINATION src/rocm-debug-agent-test
  COMPONENT tests)

enable_testing()
add_test(NAME rocm-debug-agent-test
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run-test.py ${CMAKE_CURRENT_BINARY_DIR})
