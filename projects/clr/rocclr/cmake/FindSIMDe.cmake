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

include(FindPackageHandleStandardArgs)

find_package(PkgConfig REQUIRED)
if(PkgConfig_FOUND)
    pkg_check_modules(simde IMPORTED_TARGET simde)
endif()

if(PkgConfig_FOUND AND simde_FOUND)
    message(STATUS "Found SIMDe via pkg-config")
    set(SIMDE_TARGET PkgConfig::simde)
else()
    message(STATUS "SIMDe not found via pkg-config. Falling back to find_path...")

    if(WIN32)
        find_path(SIMDE_INCLUDE_DIR
            NAMES simde/simde-common.h
            PATHS
                "$ENV{DK_ROOT}/simde"
            NO_DEFAULT_PATH
    )
    elseif(UNIX)
        find_path(SIMDE_INCLUDE_DIR
            NAMES simde/simde-common.h
            PATHS
                /usr/include
                /usr/local/include
            NO_DEFAULT_PATH
    )
    endif()

    find_package_handle_standard_args(SIMDe
                   REQUIRED_VARS SIMDE_INCLUDE_DIR)
    if(SIMDE_FOUND)
        message(STATUS "Found SIMDe headers at: ${SIMDE_INCLUDE_DIR}")
        if(NOT TARGET SIMDE)
            add_library(SIMDE INTERFACE)
            target_include_directories(SIMDE INTERFACE ${SIMDE_INCLUDE_DIR})
        endif()
        set(SIMDE_TARGET SIMDE)
    else()
        message(WARNING "could not find simde")
    endif()
endif()