# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

# ----------------------------------------------------------------------------------------#
#
# ROCm
#
# ----------------------------------------------------------------------------------------#

set(DEFAULT_ROCM_PATH /opt/rocm CACHE PATH "Default search path for ROCm")
if(EXISTS ${DEFAULT_ROCM_PATH})
    get_filename_component(_DEFAULT_ROCM_PATH "${DEFAULT_ROCM_PATH}" REALPATH)

    if(NOT "${_DEFAULT_ROCM_PATH}" STREQUAL "${DEFAULT_ROCM_PATH}")
        set(ROCPROFSYS_DEFAULT_ROCM_PATH
            "${_DEFAULT_ROCM_PATH}"
            CACHE PATH
            "Default search path for ROCm"
            FORCE
        )
    endif()
endif()

set(CMAKE_PREFIX_PATH ${DEFAULT_ROCM_PATH} ${CMAKE_PREFIX_PATH})
string(
    REPLACE
    ":"
    ";"
    CMAKE_PREFIX_PATH
    "$ENV{CMAKE_PREFIX_PATH};${CMAKE_PREFIX_PATH}"
)
