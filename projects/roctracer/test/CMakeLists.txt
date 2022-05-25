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

# Set the HIP language runtime link flags as FindHIP does not set them.
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG})
set(CMAKE_EXECUTABLE_RUNTIME_HIP_FLAG_SEP ${CMAKE_SHARED_LIBRARY_RUNTIME_CXX_FLAG_SEP})
set(CMAKE_EXECUTABLE_RPATH_LINK_HIP_FLAG ${CMAKE_SHARED_LIBRARY_RPATH_LINK_CXX_FLAG})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "/opt/rocm/hip/cmake")
set(CMAKE_HIP_ARCHITECTURES OFF)
find_package(HIP REQUIRED MODULE)

find_package(Clang REQUIRED CONFIG
             PATHS "/opt/rocm/"
             PATH_SUFFIXES "llvm/lib/cmake/clang")

## Add a custom targets to build and run all the tests
add_custom_target(mytest)
add_dependencies(mytest roctracer_tool)
add_custom_target(check COMMAND ${PROJECT_BINARY_DIR}/run.sh DEPENDS mytest)

## Build MatrixTranspose
set_source_files_properties(hip/MatrixTranspose.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(MatrixTranspose EXCLUDE_FROM_ALL hip/MatrixTranspose.cpp)
target_include_directories(MatrixTranspose PRIVATE ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(MatrixTranspose PRIVATE roctracer roctx)
add_dependencies(mytest MatrixTranspose)

## Build MatrixTranspose_test, MatrixTranspose_hipaact_test and MatrixTranspose_mgpu
set_source_files_properties(app/MatrixTranspose_test.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
function(build_matrix_transpose_test OUTPUT_FILE DEFINITIONS)
  hip_add_executable(${OUTPUT_FILE} EXCLUDE_FROM_ALL app/MatrixTranspose_test.cpp)
  target_compile_definitions(${OUTPUT_FILE} PRIVATE ITERATIONS=100 HIP_TEST=1 ${DEFINITIONS})
  target_include_directories(${OUTPUT_FILE} PRIVATE ${PROJECT_SOURCE_DIR}/inc)
  target_link_libraries(${OUTPUT_FILE} PRIVATE roctracer roctx)
  add_dependencies(mytest ${OUTPUT_FILE})
endfunction(build_matrix_transpose_test)

build_matrix_transpose_test(MatrixTranspose_test "")
build_matrix_transpose_test(MatrixTranspose_hipaact_test HIP_API_ACTIVITY_ON=1)
build_matrix_transpose_test(MatrixTranspose_mgpu MGPU_TEST=1)

## Build MatrixTranspose MatrixTranspose_ctest
add_custom_command(OUTPUT MatrixTranspose.c
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/app/MatrixTranspose_test.cpp MatrixTranspose.c)
hip_add_executable(MatrixTranspose_ctest EXCLUDE_FROM_ALL MatrixTranspose.c)
target_compile_definitions(MatrixTranspose_ctest PRIVATE HIP_TEST=0 __HIP_PLATFORM_HCC__)
target_include_directories(MatrixTranspose_ctest PRIVATE ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(MatrixTranspose_ctest PRIVATE roctracer roctx)
add_dependencies(mytest MatrixTranspose_ctest)

## Build hsaco_test reference test
add_library(hsaco_test SHARED EXCLUDE_FROM_ALL app/hsaco_test.cpp)
target_compile_definitions(hsaco_test PRIVATE AMD_INTERNAL_BUILD)
target_link_libraries(hsaco_test hsa-runtime64::hsa-runtime64)
add_dependencies(mytest hsaco_test)

## Build codeobj event test
add_library(codeobj_test SHARED EXCLUDE_FROM_ALL app/codeobj_test.cpp)
target_include_directories(codeobj_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(codeobj_test roctracer)
add_dependencies(mytest codeobj_test)

## Build the trace_buffer test
add_executable(trace_buffer EXCLUDE_FROM_ALL directed/trace_buffer.cpp)
target_include_directories(trace_buffer PRIVATE ${PROJECT_SOURCE_DIR}/src/tracer_tool)
target_link_libraries(trace_buffer Threads::Threads atomic)
add_dependencies(mytest trace_buffer)

## Build the hsa (standalone) copy test
function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
    "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
  add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/${OUTPUT_FILE}
    COMMAND clang ${CLANG_ARG_LIST}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    DEPENDS ${INPUT_FILE} clang
    COMMENT "Building ${OUTPUT_FILE}..."
    VERBATIM)
  set(HSACO_TARGET_LIST ${HSACO_TARGET_LIST} ${PROJECT_BINARY_DIR}/${OUTPUT_FILE} PARENT_SCOPE)
endfunction(generate_hsaco)

foreach(target_id ${GPU_TARGETS})
  ## generate kernel bitcodes
  generate_hsaco(${target_id} ${CMAKE_CURRENT_SOURCE_DIR}/hsa/copy.cl ${target_id}_copy.hsaco)
endforeach(target_id)
add_custom_target(hsaco_targets DEPENDS ${HSACO_TARGET_LIST})

add_executable(copy EXCLUDE_FROM_ALL hsa/copy.cpp)
target_link_libraries(copy hsa-runtime64::hsa-runtime64 Threads::Threads dl)
add_dependencies(copy hsaco_targets)
add_dependencies(mytest copy)

## Build the ROCTX test
set_source_files_properties(app/roctx_test.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(roctx_test EXCLUDE_FROM_ALL app/roctx_test.cpp)
target_link_libraries(roctx_test Threads::Threads roctx)
add_dependencies(mytest roctx_test)

## Build the backward compatibility test
add_executable(backward_compat_test EXCLUDE_FROM_ALL app/backward_compat_test.cpp)
target_link_libraries(backward_compat_test roctracer)
add_dependencies(mytest backward_compat_test)

add_executable(load_unload_reload_test EXCLUDE_FROM_ALL hsa/load_unload_reload.cpp)
target_link_libraries(load_unload_reload_test hsa-runtime64::hsa-runtime64)
add_dependencies(mytest load_unload_reload_test)

## Copy the golden traces and test scripts
configure_file(run.sh ${PROJECT_BINARY_DIR} COPYONLY)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink run.sh ${PROJECT_BINARY_DIR}/run_ci.sh)
configure_file(${PROJECT_SOURCE_DIR}/script/check_trace.py ${PROJECT_BINARY_DIR}/test/check_trace.py COPYONLY)

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "golden_traces/tests_trace_cmp_levels.txt" "golden_traces/*_trace.txt")
foreach(file ${files})
  configure_file(${file} ${PROJECT_BINARY_DIR}/test/${file} COPYONLY)
endforeach()
