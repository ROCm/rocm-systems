# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
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

find_library(
    NUMA_LIBRARY
    NAMES numa libnuma.so.1
    HINTS "${ROCM_PATH}/lib/rocm_sysdeps")

find_path(
    NUMA_INCLUDE_DIR
    NAMES numa.h
    HINTS "${ROCM_PATH}/include" "${ROCM_PATH}/lib/rocm_sysdeps/include")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NUMA DEFAULT_MSG NUMA_LIBRARY NUMA_INCLUDE_DIR)

if(NUMA_FOUND)
    set(NUMA_LIBRARIES "${NUMA_LIBRARY}")
    set(NUMA_INCLUDE_DIRS "${NUMA_INCLUDE_DIR}")
endif()

if(NUMA_FOUND AND NOT TARGET NUMA::NUMA)
    add_library(NUMA::NUMA SHARED IMPORTED)
    set_target_properties(
        NUMA::NUMA
        PROPERTIES IMPORTED_LOCATION "${NUMA_LIBRARY}")
    if(NUMA_INCLUDE_DIR)
        set_target_properties(
            NUMA::NUMA
            PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${NUMA_INCLUDE_DIR}")
    endif()
endif()

if(TARGET NUMA::NUMA AND NOT TARGET numa::numa)
    add_library(numa::numa INTERFACE IMPORTED)
    target_link_libraries(numa::numa INTERFACE NUMA::NUMA)
endif()

mark_as_advanced(NUMA_LIBRARY NUMA_INCLUDE_DIR)
