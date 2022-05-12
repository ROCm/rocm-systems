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

find_package(Clang REQUIRED CONFIG
  PATHS
    "/opt/rocm/"
  PATH_SUFFIXES
    "llvm/lib/cmake/clang")

function(generate_hsaco TARGET_ID INPUT_FILE OUTPUT_FILE)
  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
    "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${OUTPUT_FILE} ${INPUT_FILE}")

  ## Add custom command to produce a code object file.
  ## This depends on the kernel source file & compiler.
  ## It does not pickup devicelib changes.  It is not clear
  ## how to do that after conversion to --rocm-path is done.
  add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/${OUTPUT_FILE}
    COMMAND clang ${CLANG_ARG_LIST}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    DEPENDS ${INPUT_FILE} clang
    COMMENT "Building ${OUTPUT_FILE}..."
    VERBATIM)

  set(HSACO_TARGET_LIST ${HSACO_TARGET_LIST} ${PROJECT_BINARY_DIR}/${OUTPUT_FILE} PARENT_SCOPE)
endfunction(generate_hsaco)

function(build_kernel_for_devices KERNEL_FILE KERNEL_NAME)
  foreach(target_id ${GPU_TARGETS})
    ## generate kernel bitcodes
    generate_hsaco(${target_id} ${KERNEL_FILE} ${target_id}_${KERNEL_NAME}.hsaco)
  endforeach(target_id)
  set(HSACO_TARGET_LIST ${HSACO_TARGET_LIST} PARENT_SCOPE)
endfunction(build_kernel_for_devices)

set(HSACO_TARGET_LIST "")
build_kernel_for_devices("${CMAKE_CURRENT_SOURCE_DIR}/dummy_kernel/dummy_kernel.cl" "DummyKernel")
build_kernel_for_devices("${CMAKE_CURRENT_SOURCE_DIR}/simple_convolution/simple_convolution.cl" "SimpleConvolution")
add_custom_target(hsaco_targets DEPENDS ${HSACO_TARGET_LIST})

## Test kernel
set(TEST_NAME simple_convolution)
set(KERN_SRC ${TEST_NAME}/${TEST_NAME}.cpp)

## Util sources
file(GLOB UTIL_SRC "util/*.cpp")

## Test control sources
set(CTRL_SRC
  app/test.cpp
  ctrl/test_hsa.cpp)

## Building ctrl test executable
add_executable(ctrl EXCLUDE_FROM_ALL ${CTRL_SRC} ${UTIL_SRC} ${KERN_SRC})
target_compile_definitions(ctrl PRIVATE AMD_INTERNAL_BUILD)
target_include_directories(ctrl PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR})
target_link_libraries(ctrl hsa-runtime64::hsa-runtime64 Threads::Threads dl)
add_dependencies(ctrl hsaco_targets)
add_dependencies(mytest ctrl)
