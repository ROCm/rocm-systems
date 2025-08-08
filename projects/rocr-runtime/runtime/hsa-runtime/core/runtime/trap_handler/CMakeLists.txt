################################################################################
##
## The University of Illinois/NCSA
## Open Source License (NCSA)
##
## Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
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

# Import target 'clang' and 'llvm-objcopy'
find_package(Clang REQUIRED HINTS ${CMAKE_PREFIX_PATH}/llvm PATHS /opt/rocm/llvm )
find_package(LLVM REQUIRED HINTS ${CMAKE_PREFIX_PATH}/llvm PATHS /opt/rocm/llvm )

set (TARGET_DEVS "gfx900;gfx942;gfx950;gfx1010;gfx1030;gfx1100;gfx1200")
set (POSTFIX "9;942;950;1010;10;11;12")
set (SOURCE_SUFFIX ";;;;;;_gfx12")

if(${CMAKE_VERBOSE_MAKEFILE})
  get_property(clang_path TARGET clang PROPERTY LOCATION)
  get_property(objcopy_path TARGET llvm-objcopy PROPERTY LOCATION)
  message("Using clang from: ${clang_path}")
  message("Using llvm-objcopy from: ${objcopy_path}")
  message("Trap handlers assembled for: ${TARGET_DEVS}")
endif()

##==========================================
##  Add custom command to generate a kernel code object file
##==========================================
function(gen_kernel_bc TARGET_ID INPUT_FILE OUTPUT_FILE)

  set(CODE_OBJECT "${OUTPUT_FILE}.hsaco")

  separate_arguments(CLANG_ARG_LIST UNIX_COMMAND
  "-x assembler -target amdgcn-amd-amdhsa -mcpu=${TARGET_ID} -o ${CODE_OBJECT} ${INPUT_FILE}")

  ## Add custom command to produce a code object file.
  add_custom_command(OUTPUT ${CODE_OBJECT} COMMAND clang ${CLANG_ARG_LIST}
    DEPENDS ${INPUT_FILE} clang
    COMMENT "BUILDING bitcode for ${OUTPUT_FILE}..."
    VERBATIM)

  separate_arguments(OBJCOPY_ARG_LIST UNIX_COMMAND "--dump-section=.text=${OUTPUT_FILE} ${CODE_OBJECT}")

  ## Extract .text segment
  add_custom_command(OUTPUT ${OUTPUT_FILE}
                     COMMAND llvm-objcopy ${OBJCOPY_ARG_LIST}
                     DEPENDS ${CODE_OBJECT} llvm-objcopy
                     COMMENT "Extracting binary for ${OUTPUT_FILE}..."
                     VERBATIM)

  if(${CMAKE_VERBOSE_MAKEFILE})
    message("     Trap Handler Source: " ${INPUT_FILE})
    message("     Trap Handler Binary: " ${OUTPUT_FILE})
  endif()

endfunction(gen_kernel_bc)

##==========================================
## Find device code object name and forward to custom command
##==========================================
function(build_kernel TRAP_HANDLER_NAME TARGET_ID POSTFIX SOURCE_SUFFIX)

  ## generate trap handler object code files
  set (CODE_OBJECT_FILE "${TRAP_HANDLER_NAME}_${POSTFIX}")
  set (TRAP_FILE "${CMAKE_CURRENT_SOURCE_DIR}/trap_handler${SOURCE_SUFFIX}.s")
  gen_kernel_bc(${TARGET_ID} ${TRAP_FILE} ${CODE_OBJECT_FILE})

  ## Build a list of code object file names
  ## These will be target dependencies.
  set (HSACO_TARG_LIST ${HSACO_TARG_LIST} "${CODE_OBJECT_FILE}" PARENT_SCOPE)

endfunction(build_kernel)

##==========================================
## Build the kernel for a list of devices
##==========================================
function(build_kernel_for_devices TRAP_HANDLER_NAME)

  set(HSACO_TARG_LIST "")

  list(LENGTH TARGET_DEVS dev_count)
  math(EXPR dev_count "${dev_count} - 1")
  foreach(ind RANGE ${dev_count})
    list(GET TARGET_DEVS ${ind} dev)
    list(GET POSTFIX ${ind} post)
    list(GET SOURCE_SUFFIX ${ind} suffix)
    if(${CMAKE_VERBOSE_MAKEFILE})
      message("\n  Generating: ${dev} ...")
    endif()
    build_kernel(${TRAP_HANDLER_NAME} ${dev} ${post} "${suffix}")
  endforeach(ind)

  set(HSACO_TARG_LIST ${HSACO_TARG_LIST} PARENT_SCOPE)

endfunction(build_kernel_for_devices)

##==========================================
## Create Trap Handler Object Code blobs file
##==========================================
function(generate_bytecodeStrm HeaderFILE)

  separate_arguments(ARG_LIST UNIX_COMMAND "${CMAKE_CURRENT_BINARY_DIR}/${HeaderFILE}.h")
  set(ARG_LIST ${ARG_LIST} ${HSACO_TARG_LIST})

  ## Add a custom command that generates amd_trap_handler_v2.h
  ## This depends on all the generated code object files and the C++ generator script.
  add_custom_command(OUTPUT ${HeaderFILE}.h
                     COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/create_trap_handler_header.sh ${ARG_LIST}
                     COMMENT "Collating trap handlers..."
                     DEPENDS ${HSACO_TARG_LIST} create_trap_handler_header.sh )

  ## Export a target that builds (and depends on) amd_trap_handler_v2.h
  add_custom_target( ${HeaderFILE} DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${HeaderFILE}.h )

endfunction(generate_bytecodeStrm)

##==========================================
## Main function calls
##==========================================

build_kernel_for_devices("kCodeTrapHandlerV2")
generate_bytecodeStrm("amd_trap_handler_v2")
