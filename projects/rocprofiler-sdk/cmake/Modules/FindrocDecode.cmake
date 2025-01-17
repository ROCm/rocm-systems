################################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc.
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
#
################################################################################

# find rocDecode - library and headers
find_path(
    rocDecode_INCLUDE_DIR
    NAMES rocdecode.h
    PATHS ${ROCM_PATH}/include/rocdecode)
find_library(
    rocDecode_LIBRARY
    NAMES rocdecode
    HINTS ${ROCM_PATH}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocDecode
    FOUND_VAR rocDecode_FOUND
    REQUIRED_VARS rocDecode_INCLUDE_DIR rocDecode_LIBRARY)

if(rocDecode_FOUND)
    if(NOT TARGET rocDecode::rocDecode)
        add_library(rocDecode::rocDecode INTERFACE IMPORTED)
        target_link_libraries(rocDecode::rocDecode INTERFACE ${rocDecode_LIBRARY})
        target_include_directories(rocDecode::rocDecode
                                   INTERFACE ${rocDecode_INCLUDE_DIR})
    endif()
endif()

mark_as_advanced(rocDecode_INCLUDE_DIR rocDecode_LIBRARY)
