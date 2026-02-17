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

# Search super-project (e.g. amd-mesa) sysdeps first when building in TheRock
if(DEFINED THEROCK_SUPERPROJECT_INCLUDE_DIRS)
  list(APPEND _libva_include_hints ${THEROCK_SUPERPROJECT_INCLUDE_DIRS})
endif()

find_library(LIBVA_LIBRARY NAMES va HINTS ${ROCM_PATH}/lib/rocm_sysdeps/lib /opt/amdgpu/lib/x86_64-linux-gnu /opt/amdgpu/lib64 /usr/lib/x86_64-linux-gnu /usr/lib64)
find_library(LIBVA_DRM_LIBRARY NAMES va-drm HINTS ${ROCM_PATH}/lib/rocm_sysdeps/lib /opt/amdgpu/lib/x86_64-linux-gnu /opt/amdgpu/lib64 /usr/lib/x86_64-linux-gnu /usr/lib64)
find_path(LIBVA_INCLUDE_DIR NAMES va/va.h PATHS ${_libva_include_hints} ${ROCM_PATH}/lib/rocm_sysdeps/include /opt/amdgpu/include /usr/include NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libva DEFAULT_MSG LIBVA_INCLUDE_DIR LIBVA_LIBRARY)
mark_as_advanced(LIBVA_INCLUDE_DIR LIBVA_LIBRARY LIBVA_DRM_LIBRARY)

if(Libva_FOUND)
  # Find VA Version
  file(READ "${LIBVA_INCLUDE_DIR}/va/va_version.h" VA_VERSION_FILE)
  string(REGEX MATCH "VA_MAJOR_VERSION    ([0-9]*)" _ ${VA_VERSION_FILE})
  set(Libva_VERSION_MAJOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "VA_MINOR_VERSION    ([0-9]*)" _ ${VA_VERSION_FILE})
  set(Libva_VERSION_MINOR ${CMAKE_MATCH_1})
  string(REGEX MATCH "VA_MICRO_VERSION    ([0-9]*)" _ ${VA_VERSION_FILE})
  set(Libva_VERSION_MICRO ${CMAKE_MATCH_1})
  set(Libva_VERSION "${Libva_VERSION_MAJOR}.${Libva_VERSION_MINOR}.${Libva_VERSION_MICRO}")
  message("-- ${White}Found Libva Version: ${Libva_VERSION}${ColourReset}")

endif()

if(Libva_FOUND)
  if(NOT TARGET Libva::va)
    add_library(Libva::va UNKNOWN IMPORTED)
    set_target_properties(Libva::va PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBVA_INCLUDE_DIR}"
        IMPORTED_LOCATION "${LIBVA_LIBRARY}")
  endif()
  if(NOT TARGET Libva::va_drm)
    add_library(Libva::va_drm UNKNOWN IMPORTED)
    set_target_properties(Libva::va_drm PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBVA_INCLUDE_DIR}"
      IMPORTED_LOCATION "${LIBVA_DRM_LIBRARY}")
  endif()
  message("-- ${White}Using Libva -- \n\tLibraries:${LIBVA_LIBRARY} \n\tIncludes:${LIBVA_INCLUDE_DIR}${ColourReset}")
  message("-- ${White}Using Libva-drm -- \n\tLibraries:${LIBVA_DRM_LIBRARY}${ColourReset}")
else()
  if(Libva_FIND_REQUIRED)
    message(FATAL_ERROR "{Red}FindLibva -- Libva NOT FOUND${ColourReset}")
  endif()
endif()