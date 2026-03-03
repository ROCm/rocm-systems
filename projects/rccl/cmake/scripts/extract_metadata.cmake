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

# Extract device code objects from librccl.so using llvm-objdump.
# Replaces the deprecated roc-obj-ls / roc-obj-extract tools.
#
# llvm-objdump --offloading extracts embedded bundles into files named like:
#   librccl.so.0.hipv4-amdgcn-amd-amdhsa--gfx942
# We rename the device images to librccl-<arch>-<index>.co so the
# Standalone.StackSize test can find them via `find ../ -name "librccl*.co"`.

set(EXTRACT_TIMEOUT 10 CACHE STRING "Timeout in seconds for extraction")

if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm")
endif()
set(OBJDUMP "${ROCM_PATH}/llvm/bin/llvm-objdump")

if(NOT EXISTS "${OBJDUMP}")
  message(WARNING "llvm-objdump not found at ${OBJDUMP}; skipping code object extraction")
  return()
endif()

execute_process(
  COMMAND ${OBJDUMP} --offloading librccl.so
  RESULT_VARIABLE extract_result
  OUTPUT_VARIABLE extract_output
  ERROR_VARIABLE  extract_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
  TIMEOUT ${EXTRACT_TIMEOUT}
)

if(NOT extract_result EQUAL 0)
  if(extract_result STREQUAL "TIMEOUT")
    message(WARNING "[Timeout] llvm-objdump did not finish within ${EXTRACT_TIMEOUT}s. stderr: ${extract_error}")
  else()
    message(WARNING "[Error ${extract_result}] llvm-objdump --offloading failed. stderr: ${extract_error}")
  endif()
  return()
endif()

# Parse extracted file list from stdout and rename device images to .co
string(REGEX REPLACE "\n$" "" extract_output "${extract_output}")
string(REPLACE "\n" ";" extract_lines "${extract_output}")

set(_co_index 0)
foreach(line ${extract_lines})
  if(line MATCHES "Extracting offload bundle: (.+\\.hipv4-amdgcn-amd-amdhsa--([a-z0-9]+))$")
    set(_extracted_file "${CMAKE_MATCH_1}")
    set(_arch "${CMAKE_MATCH_2}")
    if(EXISTS "${_extracted_file}")
      set(_co_file "librccl-${_arch}-${_co_index}.co")
      file(RENAME "${_extracted_file}" "${_co_file}")
      math(EXPR _co_index "${_co_index} + 1")
    endif()
  endif()
  # Clean up host bundles
  if(line MATCHES "Extracting offload bundle: (.+\\.host-.+)$")
    set(_host_file "${CMAKE_MATCH_1}")
    if(EXISTS "${_host_file}")
      file(REMOVE "${_host_file}")
    endif()
  endif()
endforeach()

if(_co_index EQUAL 0)
  message(WARNING "No device code objects were extracted from librccl.so")
else()
  message(STATUS "Extracted ${_co_index} device code object(s) from librccl.so")
endif()
