################################################################################
# Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc.
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

# Prefer the upstream rocjpeg CONFIG package (installed lowercase as `rocjpeg`). Fall back
# to manual path/library discovery for older install layouts. Re-read the version header
# only when CONFIG did not supply a version.
find_package(rocjpeg CONFIG QUIET)

# handle the case where CONFIG is found but does not specify an include directory (which
# generally shouldn't happen)
if(rocjpeg_FOUND AND NOT rocjpeg_INCLUDE_DIR)
    message(
        WARNING
            "Found rocjpeg CONFIG package but it did not specify an include directory. Ignoring CONFIG results."
        )
    set(rocjpeg_FOUND OFF)
endif()

if(rocjpeg_FOUND)
    set(_rocjpeg_FOUND_CONFIG ON)
    # for backwards compatibility, set the root dir to the parent of the include dir
    get_filename_component(_rocjpeg_ROOT_DIR ${rocjpeg_INCLUDE_DIR} DIRECTORY)
    set(rocjpeg_ROOT_DIR
        ${_rocjpeg_ROOT_DIR}
        CACHE INTERNAL "Root directory of rocjpeg installation")
else()
    set(_rocjpeg_FOUND_CONFIG OFF)
    # find rocjpeg - library and headers
    find_path(
        rocjpeg_ROOT_DIR
        NAMES include/rocjpeg
        HINTS ${ROCM_PATH}
        PATHS ${ROCM_PATH})

    mark_as_advanced(rocjpeg_ROOT_DIR)

    find_path(
        rocjpeg_INCLUDE_DIR
        NAMES rocjpeg/rocjpeg.h
        HINTS ${rocjpeg_ROOT_DIR}
        PATHS ${rocjpeg_ROOT_DIR}
        PATH_SUFFIXES include)

    find_library(
        rocjpeg_LIBRARY
        NAMES rocjpeg
        HINTS ${rocjpeg_ROOT_DIR}
        PATHS ${rocjpeg_ROOT_DIR}
        PATH_SUFFIXES lib)
endif()

# if rocjpeg_VERSION is not set by CONFIG or manual discovery, read it from the version
# header
if(NOT rocjpeg_VERSION OR NOT _rocjpeg_FOUND_CONFIG)
    function(_rocjpeg_read_version_header _VERSION_VAR)
        if(rocjpeg_INCLUDE_DIR AND EXISTS
                                   "${rocjpeg_INCLUDE_DIR}/rocjpeg/rocjpeg_version.h")
            file(READ "${rocjpeg_INCLUDE_DIR}/rocjpeg/rocjpeg_version.h" _rocjpeg_version)
            macro(_rocjpeg_get_version_num _VAR)
                foreach(_NAME ${ARGN})
                    string(REGEX MATCH "define([ \t]+)${_NAME}([ \t]+)([0-9]+)" _tmp
                                 "${_rocjpeg_version}")
                    set(${_VAR} 0)
                    if(_tmp MATCHES "([0-9]+)")
                        string(REGEX REPLACE "(.*${_NAME}[ ]+)([0-9]+)" "\\2" ${_VAR}
                                             "${_tmp}")
                        break()
                    endif()
                endforeach()
            endmacro()

            _rocjpeg_get_version_num(_major "ROCJPEG_VERSION_MAJOR"
                                     "ROCJPEG_MAJOR_VERSION")
            _rocjpeg_get_version_num(_minor "ROCJPEG_VERSION_MINOR"
                                     "ROCJPEG_MINOR_VERSION")
            _rocjpeg_get_version_num(_patch "ROCJPEG_VERSION_PATCH"
                                     "ROCJPEG_MICRO_VERSION")
            set(${_VERSION_VAR}
                ${_major}.${_minor}.${_patch}
                PARENT_SCOPE)
        endif()
    endfunction()

    _rocjpeg_read_version_header(rocjpeg_VERSION)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocjpeg
    FOUND_VAR rocjpeg_FOUND
    VERSION_VAR rocjpeg_VERSION
    REQUIRED_VARS rocjpeg_INCLUDE_DIR rocjpeg_LIBRARY)

if(rocjpeg_FOUND)
    if(NOT TARGET rocjpeg::rocjpeg)
        add_library(rocjpeg::rocjpeg INTERFACE IMPORTED)
        target_link_libraries(rocjpeg::rocjpeg INTERFACE ${rocjpeg_LIBRARY})
        target_include_directories(rocjpeg::rocjpeg INTERFACE ${rocjpeg_INCLUDE_DIR})
    endif()
endif()

mark_as_advanced(rocjpeg_INCLUDE_DIR rocjpeg_LIBRARY)
