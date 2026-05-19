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

# Search super-project (e.g. libdrm) sysdeps first when building in TheRock
if(DEFINED THEROCK_SUPERPROJECT_INCLUDE_DIRS)
  list(APPEND _libdrm_amdgpu_include_hints ${THEROCK_SUPERPROJECT_INCLUDE_DIRS})
endif()

find_library(LIBDRM_AMDGPU_LIBRARY NAMES drm_amdgpu HINTS ${ROCM_PATH}/lib/rocm_sysdeps/lib /opt/amdgpu/lib/x86_64-linux-gnu /opt/amdgpu/lib64 /usr/lib/x86_64-linux-gnu /usr/lib64)
find_path(LIBDRM_AMDGPU_INCLUDE_DIR NAMES libdrm/amdgpu.h libdrm/amdgpu_drm.h PATHS ${_libdrm_amdgpu_include_hints} ${ROCM_PATH}/lib/rocm_sysdeps/include /opt/amdgpu/include /usr/include /usr/ /usr/local/include NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Libdrm_amdgpu DEFAULT_MSG LIBDRM_AMDGPU_INCLUDE_DIR LIBDRM_AMDGPU_LIBRARY)
mark_as_advanced(LIBDRM_AMDGPU_INCLUDE_DIR LIBDRM_AMDGPU_LIBRARY)

if(Libdrm_amdgpu_FOUND)
  if(NOT TARGET Libdrm_amdgpu::drm_amdgpu)
    add_library(Libdrm_amdgpu::drm_amdgpu UNKNOWN IMPORTED)
    set_target_properties(Libdrm_amdgpu::drm_amdgpu PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBDRM_AMDGPU_INCLUDE_DIR}"
      IMPORTED_LOCATION "${LIBDRM_AMDGPU_LIBRARY}")
  endif()
  message("-- ${White}Using Libdrm_amdgpu -- \n\tLibraries:${LIBDRM_AMDGPU_LIBRARY} \n\tIncludes:${LIBDRM_AMDGPU_INCLUDE_DIR} ${ColourReset}")
else()
  if(Libdrm_amdgpu_FIND_REQUIRED)
    message(FATAL_ERROR "{Red}FindLibdrm_amdgpu -- Libdrm_admgpu NOT FOUND${ColourReset}")
  endif()
endif()