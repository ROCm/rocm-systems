# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

## Extract the per-architecture offload bundles from librccl.so.
## `llvm-objdump --offloading` writes each embedded bundle (host + every gfx
## arch) out as a sibling file, which is the artifact we want.
##
## This script is run via `cmake -P`, so it has no access to the build's cache
## variables; resolve ROCM_PATH from the environment and locate llvm-objdump,
## falling back to PATH.
if(NOT DEFINED ROCM_PATH AND DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH "$ENV{ROCM_PATH}")
endif()

find_program(LLVM_OBJDUMP
    NAMES llvm-objdump
    HINTS "${ROCM_PATH}/llvm/bin" "${ROCM_PATH}/bin"
)
if(NOT LLVM_OBJDUMP)
    set(LLVM_OBJDUMP llvm-objdump)
endif()

execute_process( COMMAND ${LLVM_OBJDUMP} --offloading librccl.so
    RESULT_VARIABLE list_result
    OUTPUT_VARIABLE cmd_output
    ERROR_VARIABLE cmd_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)

if(list_result EQUAL 0)
    message(STATUS "Extracted offload bundles from librccl.so:\n${cmd_output}")
else()
    ## Don't fail the build if extraction fails; just report it.
    message(WARNING "[Error ${list_result}] '${LLVM_OBJDUMP} --offloading' failed. stderr: ${cmd_error}")
endif()

