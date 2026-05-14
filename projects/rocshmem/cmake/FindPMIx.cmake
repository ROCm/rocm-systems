###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

# Find pmix installation.
# Different scenarios need to be covered:
#  - pmix at user-provided location (i.e., in PMIX_ROOT)
#  - pmix installed as part of Open MPI, i.e., in the MPI installation directories
#  - pmix deployed with linux distros, Slurm, etc.

find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
  # Figure out and prepend the install dir for MPI
  string(REGEX REPLACE "/include$" "" mpi_dir "${MPI_CXX_HEADER_DIR}")
  foreach (mpiroot "${MPI_ROOT}" "$ENV{MPI_ROOT}" "${mpi_dir}")
    if (mpiroot)
      set(ENV{PKG_CONFIG_PATH} "${mpiroot}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    endif()
  endforeach()
  # prepend PMIX_ROOT
  foreach (pmixroot "${PMIX_ROOT}" "$ENV{PMIX_ROOT}" "${PMIx_ROOT}" "$ENV{PMIx_ROOT}")
    if (pmixroot)
      set(ENV{PKG_CONFIG_PATH} "${pmixroot}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    endif()
  endforeach()
  pkg_check_modules(PC_PMIX QUIET pmix)
endif()

find_path(PMIX_INCLUDE_DIR pmix.h
  HINTS ${PC_PMIX_INCLUDE_DIRS} ${MPI_CXX_HEADER_DIR} ${MPI_ROOT} $ENV{MPI_ROOT}
  PATH_SUFFIXES include)
if (PMIX_INCLUDE_DIR)
  string(REGEX REPLACE "/include$" "" pmix_dir ${PMIX_INCLUDE_DIR})
  find_library(PMIX_LIBRARY pmix PATHS ${pmix_dir} PATH_SUFFIXES lib lib64 NO_DEFAULT_PATH)
endif()

find_package_handle_standard_args(PMIx DEFAULT_MSG
  PMIX_LIBRARY PMIX_INCLUDE_DIR)
mark_as_advanced(PMIX_LIBRARY PMIX_INCLUDE_DIR)

if (PMIx_FOUND)
add_library(PMIx::pmix UNKNOWN IMPORTED)
set_target_properties(PMIx::pmix PROPERTIES
  IMPORTED_LOCATION "${PMIX_LIBRARY}"
  INTERFACE_COMPILE_OPTIONS "${PC_PMIX_CFLAGS_OTHER}"
  INTERFACE_INCLUDE_DIRECTORIES "${PMIX_INCLUDE_DIR}"
)
endif()
