# MIT License
#
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

# These overrides are due to CMake CHECK_SYMBOL_EXISTS modifying CMAKE_CXX_FLAGS to do a test compile,
# while ROCMChecks gives a warning if this variable is modified manually without a target.

# We now choose to disable ROCMChecks for this one case.

set(DISABLE_ROCM_CHECK OFF)

function(rocm_check_toolchain_var var access value list_file)
  if(NOT DISABLE_ROCM_CHECK)
    _rocm_check_toolchain_var("${var}" "${access}" "${value}" "${list_file}")
  endif()
endfunction()

macro(CHECK_SYMBOL_EXISTS)
  set(DISABLE_ROCM_CHECK ON)
  _check_symbol_exists(${ARGN})
  set(DISABLE_ROCM_CHECK OFF)
endmacro()
