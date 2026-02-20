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

#[=======================================================================[.rst:
Findrocpdsna
--------------

Find the rocpdsna library.

Imported Targets
^^^^^^^^^^^^^^^^

``rocpdsna::rocpdsna``
  The rocpdsna library, if found.

Result Variables
^^^^^^^^^^^^^^^^

``rocpdsna_FOUND``
  True if rocpdsna was found.
``rocpdsna_INCLUDE_DIRS``
  Include directories for rocpdsna.
``rocpdsna_LIBRARIES``
  Libraries to link against.

#]=======================================================================]

# If target already exists (from add_subdirectory), we're done
if(TARGET rocpdsna)
    set(rocpdsna_FOUND TRUE)

    # Create namespaced alias if it doesn't exist
    if(NOT TARGET rocpdsna::rocpdsna)
        add_library(rocpdsna::rocpdsna ALIAS rocpdsna)
    endif()

    # Get properties from existing target
    get_target_property(
        rocpdsna_INCLUDE_DIRS
        rocpdsna
        INTERFACE_INCLUDE_DIRECTORIES
    )
    set(rocpdsna_LIBRARIES rocpdsna)

    return()
endif()

# If namespaced target exists (from previous find_package), we're done
if(TARGET rocpdsna::rocpdsna)
    set(rocpdsna_FOUND TRUE)
    return()
endif()

# Search for installed library
find_path(
    rocpdsna_INCLUDE_DIR
    NAMES rocpdsna/storage.hpp
    HINTS ${rocpdsna_ROOT} $ENV{rocpdsna_ROOT} ${CMAKE_BINARY_DIR}/include
    PATH_SUFFIXES include
)

find_library(
    rocpdsna_LIBRARY
    NAMES rocpdsna
    HINTS ${rocpdsna_ROOT} $ENV{rocpdsna_ROOT} ${CMAKE_BINARY_DIR}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocpdsna
    REQUIRED_VARS rocpdsna_LIBRARY rocpdsna_INCLUDE_DIR
)

if(rocpdsna_FOUND)
    set(rocpdsna_INCLUDE_DIRS ${rocpdsna_INCLUDE_DIR})
    set(rocpdsna_LIBRARIES ${rocpdsna_LIBRARY})

    if(NOT TARGET rocpdsna::rocpdsna)
        add_library(rocpdsna::rocpdsna SHARED IMPORTED)
        set_target_properties(
            rocpdsna::rocpdsna
            PROPERTIES
                IMPORTED_LOCATION "${rocpdsna_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${rocpdsna_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(rocpdsna_INCLUDE_DIR rocpdsna_LIBRARY)
