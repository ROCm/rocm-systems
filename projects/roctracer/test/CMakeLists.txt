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

get_property(HSA_RUNTIME_INCLUDE_DIRECTORIES TARGET hsa-runtime64::hsa-runtime64 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)

# Find amdclang++ compiler for HIP language support
if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

# Enable HIP as a project language
enable_language(HIP)

# Set HIP compilation flags to match CXX flags
foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

# Set HIP standard flags
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

find_package(Clang REQUIRED CONFIG
             PATHS "${ROCM_PATH}"
             PATH_SUFFIXES "llvm/lib/cmake/clang")

## Add a custom targets to build and run all the tests
add_custom_target(mytest)
add_dependencies(mytest roctracer_tool hip_stats)
add_custom_target(check COMMAND ${PROJECT_BINARY_DIR}/run.sh DEPENDS mytest)

## Helper function to create HIP executables with common settings
function(add_hip_executable TARGET_NAME SOURCE_FILE)
  set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE HIP)
  add_executable(${TARGET_NAME} ${SOURCE_FILE})
  target_link_options(${TARGET_NAME} PRIVATE "-Wl,--build-id=md5")
endfunction()

## Build MatrixTranspose
add_hip_executable(MatrixTranspose hip/MatrixTranspose.cpp)
target_include_directories(MatrixTranspose PRIVATE ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(MatrixTranspose PRIVATE roctracer roctx)
add_dependencies(mytest MatrixTranspose)

## Build MatrixTranspose_test, MatrixTranspose_hipaact_test and MatrixTranspose_mgpu
function(build_matrix_transpose_test OUTPUT_FILE DEFINITIONS)
  add_hip_executable(${OUTPUT_FILE} app/MatrixTranspose_test.cpp)
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
add_hip_executable(MatrixTranspose_ctest MatrixTranspose.c)
target_compile_definitions(MatrixTranspose_ctest PRIVATE HIP_TEST=0 __HIP_PLATFORM_AMD__)
target_include_directories(MatrixTranspose_ctest PRIVATE ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(MatrixTranspose_ctest PRIVATE roctracer roctx)
add_dependencies(mytest MatrixTranspose_ctest)

## Build codeobj event test
add_library(codeobj_test SHARED app/codeobj_test.cpp)
target_include_directories(codeobj_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(codeobj_test roctracer)
add_dependencies(mytest codeobj_test)
install(TARGETS codeobj_test DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/test COMPONENT tests)

## Build the hsa (standalone) copy test
function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
    "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0 -Wl,--build-id=sha1
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")
  add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/${OUTPUT_FILE}
    COMMAND clang ${CLANG_ARG_LIST}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    DEPENDS ${INPUT_FILE} clang
    COMMENT "Building ${OUTPUT_FILE}..."
    VERBATIM)
  install(FILES ${PROJECT_BINARY_DIR}/${OUTPUT_FILE} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME} COMPONENT tests)
  set(HSACO_TARGET_LIST ${HSACO_TARGET_LIST} ${PROJECT_BINARY_DIR}/${OUTPUT_FILE} PARENT_SCOPE)
endfunction(generate_hsaco)

separate_arguments(GPU_TARGETS)
list(LENGTH GPU_TARGETS list_count)
if (${list_count} LESS_EQUAL 1)
  string(REPLACE " " ";" GPU_LIST "${GPU_TARGETS}")
  string(REPLACE "," ";" GPU_LIST "${GPU_TARGETS}")
else()
  set(GPU_LIST ${GPU_TARGETS})
endif()

foreach(target_id ${GPU_LIST})
  ## generate kernel bitcodes
  generate_hsaco(${target_id} ${CMAKE_CURRENT_SOURCE_DIR}/hsa/copy.cl ${target_id}_copy.hsaco)
endforeach(target_id)
add_custom_target(hsaco_targets DEPENDS ${HSACO_TARGET_LIST})

add_executable(copy hsa/copy.cpp)
target_link_libraries(copy hsa-runtime64::hsa-runtime64 Threads::Threads dl)
add_dependencies(copy hsaco_targets)
add_dependencies(mytest copy)

## Build the ROCTX test
add_hip_executable(roctx_test app/roctx_test.cpp)
target_link_libraries(roctx_test Threads::Threads roctx)
add_dependencies(mytest roctx_test)

## Build the backward compatibility test
add_executable(backward_compat_test app/backward_compat_test.cpp)
target_link_libraries(backward_compat_test roctracer)
add_dependencies(mytest backward_compat_test)

add_executable(load_unload_reload_test hsa/load_unload_reload.cpp)
target_link_libraries(load_unload_reload_test hsa-runtime64::hsa-runtime64)
add_dependencies(mytest load_unload_reload_test)

## Build the trace_buffer test
add_executable(trace_buffer directed/trace_buffer.cpp)
target_include_directories(trace_buffer PRIVATE ${PROJECT_SOURCE_DIR}/src/tracer_tool)
target_link_libraries(trace_buffer Threads::Threads atomic)
add_dependencies(mytest trace_buffer)

## Build the memory_pool test
add_executable(memory_pool directed/memory_pool.cpp)
target_include_directories(memory_pool PRIVATE ${PROJECT_SOURCE_DIR}/src/roctracer ${PROJECT_SOURCE_DIR}/inc)
target_link_libraries(memory_pool Threads::Threads atomic)
add_dependencies(mytest memory_pool)

## Build the activity_and_callback test
add_hip_executable(activity_and_callback directed/activity_and_callback.cpp)
target_link_libraries(activity_and_callback roctracer)
add_dependencies(mytest activity_and_callback)

## Build the multi_pool_activities test
add_hip_executable(multi_pool_activities directed/multi_pool_activities.cpp)
target_link_libraries(multi_pool_activities roctracer)
add_dependencies(mytest multi_pool_activities)

## Build the dlopen test
add_executable(dlopen directed/dlopen.cpp)
target_include_directories(dlopen PRIVATE ${PROJECT_SOURCE_DIR}/inc ${HSA_RUNTIME_INCLUDE_DIRECTORIES})
target_link_libraries(dlopen dl)
add_dependencies(mytest dlopen)

## Copy the golden traces and test scripts
configure_file(run.sh ${PROJECT_BINARY_DIR} COPYONLY)
execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink run.sh ${PROJECT_BINARY_DIR}/run_ci.sh)
install(PROGRAMS ${PROJECT_BINARY_DIR}/run.sh RENAME "run_tests.sh" DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME} COMPONENT tests)

configure_file(${PROJECT_SOURCE_DIR}/script/check_trace.py ${PROJECT_BINARY_DIR}/test/check_trace.py COPYONLY)
install(PROGRAMS ${PROJECT_BINARY_DIR}/test/check_trace.py DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/test COMPONENT tests)

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "golden_traces/tests_trace_cmp_levels.txt" "golden_traces/*_trace.txt")
foreach(file ${files})
  configure_file(${file} ${PROJECT_BINARY_DIR}/test/${file} COPYONLY)
endforeach()
install(DIRECTORY ${PROJECT_BINARY_DIR}/test/golden_traces DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/test COMPONENT tests)

# install all executables
set(all_executables)
get_property(all_targets DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY BUILDSYSTEM_TARGETS)
foreach(target IN LISTS all_targets)
  get_target_property(target_type ${target} TYPE)
  if (target_type STREQUAL "EXECUTABLE")
    list(APPEND all_executables ${target})
  endif()
endforeach()
install(TARGETS ${all_executables} DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/${PROJECT_NAME}/test COMPONENT tests)
