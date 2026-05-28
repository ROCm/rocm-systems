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

find_path(MSCCLPP_INCLUDE_DIRS
    NAMES mscclpp/gpu.hpp
    HINTS
    ${MSCCLPP_ROOT}/include)

find_library(MSCCLPP_LIBRARIES
    NAMES mscclpp_nccl
    HINTS
    ${MSCCLPP_ROOT}/lib)

include (FindPackageHandleStandardArgs)
find_package_handle_standard_args(mscclpp_nccl DEFAULT_MSG MSCCLPP_INCLUDE_DIRS MSCCLPP_LIBRARIES)
mark_as_advanced(MSCCLPP_INCLUDE_DIRS MSCCLPP_LIBRARIES)
    