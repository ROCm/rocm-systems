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

# Prefer the upstream rocdecode CONFIG package (installed lowercase as `rocdecode`). Fall
# back to manual path/library discovery for older install layouts. Re-read the version
# header only when CONFIG did not supply a version.

find_package(rocdecode CONFIG QUIET)

# handle the case where CONFIG is found but does not specify an include directory (which
# generally shouldn't happen)
if(rocdecode_FOUND AND NOT rocdecode_INCLUDE_DIR)
    message(
        WARNING
            "Found rocdecode CONFIG package but it did not specify an include directory. Ignoring CONFIG results."
        )
    set(rocdecode_FOUND OFF)
endif()

if(rocdecode_FOUND)
    set(_rocdecode_FOUND_CONFIG ON)
    # for backwards compatibility, set the root dir to the parent of the include dir
    get_filename_component(_rocdecode_ROOT_DIR ${rocdecode_INCLUDE_DIR} DIRECTORY)
    set(rocdecode_ROOT_DIR
        ${_rocdecode_ROOT_DIR}
        CACHE INTERNAL "Root directory of rocdecode installation")
else()
    set(_rocdecode_FOUND_CONFIG OFF)
    # find rocdecode - library and headers
    find_path(
        rocdecode_ROOT_DIR
        NAMES include/rocdecode
        HINTS ${ROCM_PATH}
        PATHS ${ROCM_PATH})

    mark_as_advanced(rocdecode_ROOT_DIR)

    find_path(
        rocdecode_INCLUDE_DIR
        NAMES rocdecode/rocdecode.h
        HINTS ${rocdecode_ROOT_DIR}
        PATHS ${rocdecode_ROOT_DIR}
        PATH_SUFFIXES include)

    find_library(
        rocdecode_LIBRARY
        NAMES rocdecode
        HINTS ${rocdecode_ROOT_DIR}
        PATHS ${rocdecode_ROOT_DIR}
        PATH_SUFFIXES lib)

endif()

# if rocdecode_VERSION is not set by CONFIG or manual discovery, read it from the version
# header
if(NOT rocdecode_VERSION OR NOT _rocdecode_FOUND_CONFIG)
    function(_rocdecode_read_version_header _VERSION_VAR)
        if(rocdecode_INCLUDE_DIR
           AND EXISTS "${rocdecode_INCLUDE_DIR}/rocdecode/rocdecode_version.h")
            file(READ "${rocdecode_INCLUDE_DIR}/rocdecode/rocdecode_version.h"
                 _rocdecode_version)
            macro(_rocdecode_get_version_num _VAR)
                foreach(_NAME ${ARGN})
                    string(REGEX MATCH "define([ \t]+)${_NAME}([ \t]+)([0-9]+)" _tmp
                                 "${_rocdecode_version}")
                    set(${_VAR} 0)

                    if(_tmp MATCHES "([0-9]+)")
                        string(REGEX REPLACE "(.*${_NAME}[ ]+)([0-9]+)" "\\2" ${_VAR}
                                             "${_tmp}")
                        break()
                    endif()
                endforeach()
            endmacro()

            _rocdecode_get_version_num(_major "ROCDECODE_VERSION_MAJOR"
                                       "ROCDECODE_MAJOR_VERSION")
            _rocdecode_get_version_num(_minor "ROCDECODE_VERSION_MINOR"
                                       "ROCDECODE_MINOR_VERSION")
            _rocdecode_get_version_num(_patch "ROCDECODE_VERSION_PATCH"
                                       "ROCDECODE_MICRO_VERSION")
            set(${_VERSION_VAR}
                ${_major}.${_minor}.${_patch}
                PARENT_SCOPE)
        endif()
    endfunction()

    _rocdecode_read_version_header(rocdecode_VERSION)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocdecode
    FOUND_VAR rocdecode_FOUND
    VERSION_VAR rocdecode_VERSION
    REQUIRED_VARS rocdecode_INCLUDE_DIR rocdecode_LIBRARY)

if(rocdecode_FOUND)
    if(NOT TARGET rocdecode::rocdecode)
        add_library(rocdecode::rocdecode INTERFACE IMPORTED)
        target_link_libraries(rocdecode::rocdecode INTERFACE ${rocdecode_LIBRARY})
        target_include_directories(rocdecode::rocdecode
                                   INTERFACE ${rocdecode_INCLUDE_DIR})
    endif()
endif()

mark_as_advanced(rocdecode_INCLUDE_DIR rocdecode_LIBRARY)
