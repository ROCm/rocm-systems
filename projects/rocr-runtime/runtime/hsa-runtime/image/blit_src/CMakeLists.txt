################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
##
## Developed by:
##
##                 AMD Research and AMD HSA Software Development
##
##                 Advanced Micro Devices, Inc.
##
##                 www.amd.com
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

cmake_minimum_required ( VERSION 3.7 )

# Import target 'clang'
find_package(Clang REQUIRED HINTS ${CMAKE_PREFIX_PATH}/llvm PATHS /opt/rocm/llvm )

# Determine the target devices if not specified
if (NOT DEFINED TARGET_DEVICES)
  set (TARGET_DEVICES "gfx700;gfx701;gfx702;gfx801;gfx802;gfx803;gfx805;gfx810"
                      "gfx900;gfx902;gfx904;gfx906;gfx908;gfx909;gfx90a;gfx90c;gfx942;gfx950"
                      "gfx1010;gfx1011;gfx1012;gfx1013;gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036"
                      "gfx1100;gfx1101;gfx1102;gfx1103;gfx1150;gfx1151;gfx1152;gfx1153;gfx1200;gfx1201")
endif()
set( TARGET_DEVICES ${TARGET_DEVICES} CACHE STRING "Build targets" FORCE )

if(${CMAKE_VERBOSE_MAKEFILE})
  get_property(clang_path TARGET clang PROPERTY LOCATION)
  message("Using clang from: ${clang_path}")
  message("Build Setting:")
  message("  Target Devices*: ${TARGET_DEVICES}")
  message("  (Specify \";\" separated list of target IDs.)")
  message("       Clang path: ${clang_path}")
endif()

##==========================================
##  Add custom command to generate a kernel code object file
##==========================================
function(gen_kernel_bc TARGET_ID INPUT_FILE OUTPUT_FILE)

  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
    "-O2 -x cl -Xclang -finclude-default-header -cl-denorms-are-zero -cl-std=CL2.0
    -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -mcode-object-version=4
    -o ${OUTPUT_FILE} ${INPUT_FILE}")

  ## Add custom command to produce a code object file.
  ## This depends on the kernel source file & compiler.
  add_custom_command(OUTPUT ${OUTPUT_FILE} COMMAND clang ${CLANG_ARG_LIST}
    DEPENDS ${INPUT_FILE} clang
    COMMENT "BUILDING bitcode for ${OUTPUT_FILE}..."
    VERBATIM)

if(${CMAKE_VERBOSE_MAKEFILE})
  message("      Kernel Source: " ${INPUT_FILE})
  message("     Kernel Bitcode: " ${OUTPUT_FILE})
endif()

endfunction(gen_kernel_bc)

##==========================================
## Find device code object name and forward to custom command
##==========================================
function(build_kernel BLIT_NAME TARGET_ID)

  ## generate kernel bitcodes
  set (CODE_OBJECT_FILE "${BLIT_NAME}_${TARGET_ID}")
  set (CL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/imageblit_kernels.cl)
  gen_kernel_bc(${TARGET_ID} ${CL_FILE} ${CODE_OBJECT_FILE})

  ## Build a list of code object file names
  ## These will be target dependencies.
  set (HSACO_TARG_LIST ${HSACO_TARG_LIST} "${CODE_OBJECT_FILE}" PARENT_SCOPE)

endfunction(build_kernel)

##==========================================
## Build the kernel for a list of devices
##==========================================
function(build_kernel_for_devices BLIT_NAME)

  set(HSACO_TARG_LIST "")

  foreach(dev ${TARGET_DEVICES})
    if(${CMAKE_VERBOSE_MAKEFILE})
      message("\n  Generating: ${dev} ...")
    endif()
    build_kernel(${BLIT_NAME} ${dev})
  endforeach(dev)

  set(HSACO_TARG_LIST ${HSACO_TARG_LIST} PARENT_SCOPE)

endfunction(build_kernel_for_devices)

##==========================================
## Create BLIT Code Object blobs file
##==========================================
function(generate_blit_file BFILE)

  ## Add a custom command that generates opencl_blit_objects.cpp
  ## This depends on all the generated code object files and the C++ generator script.
  add_custom_command(OUTPUT ${BFILE}.cpp
                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/create_hsaco_ascii_file.sh ${CMAKE_CURRENT_BINARY_DIR}/${BFILE}.cpp
                     DEPENDS ${HSACO_TARG_LIST} create_hsaco_ascii_file.sh )

  ## Export a target that builds (and depends on) opencl_blit_objects.cpp
  add_custom_target( ${BFILE} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${BFILE}.cpp )

endfunction(generate_blit_file)

build_kernel_for_devices("ocl_blit_object")
generate_blit_file("opencl_blit_objects")
