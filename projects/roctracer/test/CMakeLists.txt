################################################################################
## Copyright (c) 2018-2022 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.
################################################################################

add_custom_target(mytest) # builds all the tests
add_dependencies(mytest roctracer_tool ctrl)

# Add a 'check' target to run the tests.
add_custom_target(check COMMAND ${PROJECT_BINARY_DIR}/run.sh DEPENDS mytest)

## build HIP tests
set(GEN_INC_DIR ${PROJECT_BINARY_DIR}/inc)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/opt/rocm/hip/cmake")
find_package(HIP REQUIRED MODULE)

set_source_files_properties(MatrixTranspose/MatrixTranspose.cpp MatrixTranspose_test/MatrixTranspose.cpp
  PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

hip_add_executable(MatrixTranspose EXCLUDE_FROM_ALL MatrixTranspose/MatrixTranspose.cpp)
target_include_directories(MatrixTranspose PRIVATE ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(MatrixTranspose PRIVATE roctracer roctx)
add_dependencies(mytest MatrixTranspose)

function(build_matrix_transpose_test OUTPUT_FILE DEFINITIONS)
  hip_add_executable(${OUTPUT_FILE} EXCLUDE_FROM_ALL MatrixTranspose_test/MatrixTranspose.cpp)
  target_compile_definitions(${OUTPUT_FILE} PRIVATE ITERATIONS=100 HIP_TEST=1 ${DEFINITIONS})
  target_include_directories(${OUTPUT_FILE} PRIVATE ${PROJECT_SOURCE_DIR}/inc ${GEN_INC_DIR})
  target_link_libraries(${OUTPUT_FILE} PRIVATE roctracer roctx)
  add_dependencies(mytest ${OUTPUT_FILE})
endfunction(build_matrix_transpose_test)


build_matrix_transpose_test(MatrixTranspose_test "")
build_matrix_transpose_test(MatrixTranspose_hipaact_test HIP_API_ACTIVITY_ON=1)
build_matrix_transpose_test(MatrixTranspose_mgpu MGPU_TEST=1)

add_custom_command(OUTPUT MatrixTranspose.c
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/MatrixTranspose_test/MatrixTranspose.cpp MatrixTranspose.c)

hip_add_executable(MatrixTranspose_ctest EXCLUDE_FROM_ALL MatrixTranspose.c)
target_compile_definitions(MatrixTranspose_ctest PRIVATE HIP_TEST=0 __HIP_PLATFORM_HCC__)
target_include_directories(MatrixTranspose_ctest PRIVATE ${PROJECT_SOURCE_DIR}/inc ${GEN_INC_DIR})
target_link_libraries(MatrixTranspose_ctest PRIVATE roctracer roctx)
add_dependencies(mytest MatrixTranspose_ctest)

file(GLOB files "${CMAKE_CURRENT_SOURCE_DIR}/golden_traces/*_trace.txt")
foreach(file ${files})
  execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${file} ${PROJECT_BINARY_DIR}/test/)
endforeach()
execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/golden_traces/tests_trace_cmp_levels.txt ${PROJECT_BINARY_DIR}/test/)

## Build HSA test
add_subdirectory(hsa/test ${PROJECT_BINARY_DIR}/test/hsa)

if(DEFINED ROCTRACER_TARGET)
  ## Build the tracer_tool library
  file(GLOB TRACER_TOOL_SOURCES "tool/*.cpp" "${PROJECT_SOURCE_DIR}/src/util/*.cpp")
  add_library(roctracer_tool SHARED ${TRACER_TOOL_SOURCES})
  target_compile_definitions(roctracer_tool PRIVATE HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_HCC__)
  target_include_directories(roctracer_tool PRIVATE hsa/test ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/inc ${HIP_INCLUDE_DIRECTORIES} ${GEN_INC_DIR})
  target_link_libraries(roctracer_tool ${ROCTRACER_TARGET} hsa-runtime64::hsa-runtime64 Threads::Threads atomic dl)
  set_target_properties(roctracer_tool PROPERTIES CXX_VISIBILITY_PRESET hidden LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tool/exportmap)
  target_link_options(roctracer_tool PRIVATE -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/tool/exportmap -Wl,--no-undefined)
  install(TARGETS roctracer_tool LIBRARY DESTINATION lib/${ROCTRACER_NAME})
endif ()

## Build hsaco_test.cpp referenc test
add_library(hsaco_test SHARED EXCLUDE_FROM_ALL app/hsaco_test.cpp)
target_compile_definitions(hsaco_test PRIVATE AMD_INTERNAL_BUILD)
target_link_libraries(hsaco_test hsa-runtime64::hsa-runtime64)
add_dependencies(mytest hsaco_test)

## Build codeobj event test
add_library(codeobj_test SHARED EXCLUDE_FROM_ALL app/codeobj_test.cpp)
target_include_directories(codeobj_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/inc ${GEN_INC_DIR})
target_link_libraries(codeobj_test roctracer)
add_dependencies(mytest codeobj_test)

## Build the trace_buffer test
add_executable(trace_buffer EXCLUDE_FROM_ALL directed/trace_buffer.cpp)
target_include_directories(trace_buffer PRIVATE ${PROJECT_SOURCE_DIR}/test/tool)
target_link_libraries(trace_buffer Threads::Threads atomic)
add_dependencies(mytest trace_buffer)

## copying run script
execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/run.sh ${PROJECT_BINARY_DIR})
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink run.sh ${PROJECT_BINARY_DIR}/run_ci.sh)
## copying tests output check script
execute_process(COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_SOURCE_DIR}/script/check_trace.py ${PROJECT_BINARY_DIR}/test/.)
