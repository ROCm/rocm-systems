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
Findrocstorage
--------------

Find the rocstorage library.

Imported Targets
^^^^^^^^^^^^^^^^

``rocstorage::rocstorage``
  The rocstorage library, if found.

Result Variables
^^^^^^^^^^^^^^^^

``rocstorage_FOUND``
  True if rocstorage was found.
``rocstorage_INCLUDE_DIRS``
  Include directories for rocstorage.
``rocstorage_LIBRARIES``
  Libraries to link against.

#]=======================================================================]

# If target already exists (from add_subdirectory), we're done
if(TARGET rocstorage)
    set(rocstorage_FOUND TRUE)

    # Create namespaced alias if it doesn't exist
    if(NOT TARGET rocstorage::rocstorage)
        add_library(rocstorage::rocstorage ALIAS rocstorage)
    endif()

    # Get properties from existing target
    get_target_property(rocstorage_INCLUDE_DIRS rocstorage INTERFACE_INCLUDE_DIRECTORIES)
    set(rocstorage_LIBRARIES rocstorage)

    return()
endif()

# If namespaced target exists (from previous find_package), we're done
if(TARGET rocstorage::rocstorage)
    set(rocstorage_FOUND TRUE)
    return()
endif()

# Search for installed library
find_path(
    rocstorage_INCLUDE_DIR
    NAMES rocstorage/storage.hpp
    HINTS ${rocstorage_ROOT} $ENV{rocstorage_ROOT} ${CMAKE_BINARY_DIR}/include
    PATH_SUFFIXES include
)

find_library(
    rocstorage_LIBRARY
    NAMES rocstorage
    HINTS ${rocstorage_ROOT} $ENV{rocstorage_ROOT} ${CMAKE_BINARY_DIR}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    rocstorage
    REQUIRED_VARS rocstorage_LIBRARY rocstorage_INCLUDE_DIR
)

if(rocstorage_FOUND)
    set(rocstorage_INCLUDE_DIRS ${rocstorage_INCLUDE_DIR})
    set(rocstorage_LIBRARIES ${rocstorage_LIBRARY})

    if(NOT TARGET rocstorage::rocstorage)
        add_library(rocstorage::rocstorage SHARED IMPORTED)
        set_target_properties(
            rocstorage::rocstorage
            PROPERTIES
                IMPORTED_LOCATION "${rocstorage_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${rocstorage_INCLUDE_DIR}"
        )
    endif()
endif()

mark_as_advanced(rocstorage_INCLUDE_DIR rocstorage_LIBRARY)
