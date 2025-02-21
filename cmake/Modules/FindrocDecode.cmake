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

include_guard(DIRECTORY)

# find rocDecode - library and headers
find_path(
    rocDecode_ROOT_DIR
    NAMES include/rocdecode
    HINTS ${ROCM_PATH}
    PATHS ${ROCM_PATH})

mark_as_advanced(rocDecode_ROOT_DIR)

find_path(
    rocDecode_INCLUDE_DIR
    NAMES rocdecode/rocdecode.h
    HINTS ${rocDecode_ROOT_DIR}
    PATHS ${rocDecode_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(
    rocDecode_LIBRARY
    NAMES rocdecode
    HINTS ${rocDecode_ROOT_DIR}
    PATHS ${rocDecode_ROOT_DIR}
    PATH_SUFFIXES lib)

function(_rocdecode_read_version_header _VERSION_VAR)
    if(rocDecode_INCLUDE_DIR AND EXISTS
                                 "${rocDecode_INCLUDE_DIR}/rocdecode/rocdecode_version.h")
        file(READ "${rocDecode_INCLUDE_DIR}/rocdecode/rocdecode_version.h"
             _rocdecode_version)
        macro(_rocdecode_get_version_num _VAR _NAME)
            string(REGEX MATCH "define([ \t]+)${_NAME}([ \t]+)([0-9]+)" _tmp
                         "${_rocdecode_version}")
            set(${_VAR} 0)
            if(_tmp MATCHES "([0-9]+)")
                string(REGEX REPLACE "(.*${_NAME}[ ]+)([0-9]+)" "\\2" ${_VAR} "${_tmp}")
            endif()
        endmacro()

        _rocdecode_get_version_num(_major "ROCDECODE_MAJOR_VERSION")
        _rocdecode_get_version_num(_minor "ROCDECODE_MINOR_VERSION")
        _rocdecode_get_version_num(_patch "ROCDECODE_PATCH_VERSION")
        set(${_VERSION_VAR}
            ${_major}.${_minor}.${_patch}
            PARENT_SCOPE)
    endif()
endfunction()

_rocdecode_read_version_header(rocDecode_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocDecode
    FOUND_VAR rocDecode_FOUND
    VERSION_VAR rocDecode_VERSION
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
