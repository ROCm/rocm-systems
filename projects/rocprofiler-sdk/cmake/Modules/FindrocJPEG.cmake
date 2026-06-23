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

# find rocJPEG - library and headers
find_path(
    rocJPEG_ROOT_DIR
    NAMES include/rocjpeg
    HINTS ${ROCM_PATH}
    PATHS ${ROCM_PATH})

mark_as_advanced(rocJPEG_ROOT_DIR)

find_path(
    rocJPEG_INCLUDE_DIR
    NAMES rocjpeg/rocjpeg.h
    HINTS ${rocJPEG_ROOT_DIR}
    PATHS ${rocJPEG_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(
    rocJPEG_LIBRARY
    NAMES rocjpeg
    HINTS ${rocJPEG_ROOT_DIR}
    PATHS ${rocJPEG_ROOT_DIR}
    PATH_SUFFIXES lib)

function(_rocjpeg_read_version_header _VERSION_VAR)
    if(rocJPEG_INCLUDE_DIR AND EXISTS "${rocJPEG_INCLUDE_DIR}/rocjpeg/rocjpeg_version.h")
        file(READ "${rocJPEG_INCLUDE_DIR}/rocjpeg/rocjpeg_version.h" _rocjpeg_version)
        macro(_rocjpeg_get_version_num _VAR _COMPONENT)
            set(${_VAR} 0)
            # rocJPEG headers define the version as ROCJPEG_VERSION_<COMPONENT>. Also
            # tolerate the alternate ROCJPEG_<COMPONENT>_VERSION ordering.
            foreach(_NAME "ROCJPEG_VERSION_${_COMPONENT}" "ROCJPEG_${_COMPONENT}_VERSION")
                string(REGEX MATCH "define[ \t]+${_NAME}[ \t]+([0-9]+)" _tmp
                             "${_rocjpeg_version}")
                if(_tmp)
                    set(${_VAR} "${CMAKE_MATCH_1}")
                    break()
                endif()
            endforeach()
        endmacro()

        _rocjpeg_get_version_num(_major "MAJOR")
        _rocjpeg_get_version_num(_minor "MINOR")
        _rocjpeg_get_version_num(_patch "PATCH")
        set(${_VERSION_VAR}
            ${_major}.${_minor}.${_patch}
            PARENT_SCOPE)
    endif()
endfunction()

_rocjpeg_read_version_header(rocJPEG_VERSION)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocJPEG
    FOUND_VAR rocJPEG_FOUND
    VERSION_VAR rocJPEG_VERSION
    REQUIRED_VARS rocJPEG_INCLUDE_DIR rocJPEG_LIBRARY)

if(rocJPEG_FOUND)
    if(NOT TARGET rocJPEG::rocJPEG)
        add_library(rocJPEG::rocJPEG INTERFACE IMPORTED)
        target_link_libraries(rocJPEG::rocJPEG INTERFACE ${rocJPEG_LIBRARY})
        target_include_directories(rocJPEG::rocJPEG INTERFACE ${rocJPEG_INCLUDE_DIR})
    endif()
endif()

mark_as_advanced(rocJPEG_INCLUDE_DIR rocJPEG_LIBRARY)
