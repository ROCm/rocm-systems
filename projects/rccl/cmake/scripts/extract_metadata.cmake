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

# Extract embedded device code objects from librccl.so using llvm-objdump.
# Produces files like: librccl.so.0.hipv4-amdgcn-amd-amdhsa--gfx942
#
# Usage:
#   cmake -DROCM_PATH=/opt/rocm -P extract_metadata.cmake

if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm")
endif()

set(OBJDUMP "${ROCM_PATH}/llvm/bin/llvm-objdump")

execute_process(
  COMMAND ${OBJDUMP} --offloading librccl.so
  RESULT_VARIABLE result
  OUTPUT_VARIABLE cmd_output
  ERROR_VARIABLE cmd_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)

if(result EQUAL 0)
  message(STATUS "Extracted offload bundles from librccl.so")
  if(cmd_output)
    message(STATUS "${cmd_output}")
  endif()
else()
  message(WARNING "[Error ${result}] llvm-objdump --offloading failed. stderr: ${cmd_error}")
endif()
