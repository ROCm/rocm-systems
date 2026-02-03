# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# FindQuill.cmake - Find the Quill asynchronous logging library
#
# This module defines:
#   QUILL_FOUND        - True if Quill was found
#   QUILL_INCLUDE_DIR  - Quill include directories
#   QUILL_TARGET       - Target to link against (quill::quill or Quill)

include(FindPackageHandleStandardArgs)

# First try to find Quill via its CMake config (preferred method)
find_package(quill CONFIG QUIET)

if(quill_FOUND)
    message(STATUS "Found Quill via CMake config")
    set(QUILL_TARGET quill::quill)
    set(QUILL_FOUND TRUE)
else()
    message(STATUS "Quill not found via CMake config. Falling back to find_path...")

    # Try pkg-config
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(quill IMPORTED_TARGET quill)
        if(quill_FOUND)
            message(STATUS "Found Quill via pkg-config")
            set(QUILL_TARGET PkgConfig::quill)
            set(QUILL_FOUND TRUE)
        endif()
    endif()

    # Manual search as fallback
    if(NOT QUILL_FOUND)
        if(WIN32)
            find_path(QUILL_INCLUDE_DIR
                NAMES quill/Backend.h
                PATHS
                    "$ENV{QUILL_ROOT}/include"
                    "$ENV{DK_ROOT}/quill/include"
                    "C:/Program Files/quill/include"
                NO_DEFAULT_PATH
            )
            find_library(QUILL_LIBRARY
                NAMES quill
                PATHS
                    "$ENV{QUILL_ROOT}/lib"
                    "$ENV{DK_ROOT}/quill/lib"
                    "C:/Program Files/quill/lib"
                NO_DEFAULT_PATH
            )
        elseif(UNIX)
            find_path(QUILL_INCLUDE_DIR
                NAMES quill/Backend.h
                PATHS
                    /usr/include
                    /usr/local/include
                    /opt/quill/include
                    "$ENV{QUILL_ROOT}/include"
            )
            find_library(QUILL_LIBRARY
                NAMES quill
                PATHS
                    /usr/lib
                    /usr/lib64
                    /usr/local/lib
                    /usr/local/lib64
                    /opt/quill/lib
                    "$ENV{QUILL_ROOT}/lib"
            )
        endif()

        find_package_handle_standard_args(Quill
            REQUIRED_VARS QUILL_INCLUDE_DIR QUILL_LIBRARY)

        if(Quill_FOUND)
            message(STATUS "Found Quill headers at: ${QUILL_INCLUDE_DIR}")
            message(STATUS "Found Quill library at: ${QUILL_LIBRARY}")
            if(NOT TARGET Quill)
                add_library(Quill UNKNOWN IMPORTED)
                set_target_properties(Quill PROPERTIES
                    IMPORTED_LOCATION "${QUILL_LIBRARY}"
                    INTERFACE_INCLUDE_DIRECTORIES "${QUILL_INCLUDE_DIR}")
            endif()
            set(QUILL_TARGET Quill)
            set(QUILL_FOUND TRUE)
        else()
            message(WARNING "Could not find Quill logging library. "
                "Please install it via: vcpkg install quill, or "
                "set QUILL_ROOT environment variable to the installation path.")
        endif()
    endif()
endif()
