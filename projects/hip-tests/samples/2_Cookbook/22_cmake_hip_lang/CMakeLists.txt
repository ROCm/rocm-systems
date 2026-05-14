# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21.3)

# Path setup (must be before project())
if(WIN32)
  if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
      set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE STRING "ROCM Path")
    elseif(DEFINED ENV{HIP_PATH})
      set(ROCM_PATH "$ENV{HIP_PATH}" CACHE STRING "ROCM Path from HIP_PATH")
    endif()
  endif()
  if(DEFINED ROCM_PATH)
    # Convert backslashes to forward slashes for CMake
    file(TO_CMAKE_PATH "${ROCM_PATH}" ROCM_PATH)
    list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
  endif()
elseif(UNIX)
  if(NOT DEFINED ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})
endif()

project(cmake_hip_lang_support VERSION 1.0
        DESCRIPTION "HIP Language Support with upstream CMake"
        LANGUAGES HIP)

# Create the executable
if(TARGET build_cookbook)
set(EXCLUDE_OPTION EXCLUDE_FROM_ALL)
else()
set(EXCLUDE_OPTION )
endif()
add_executable(square2 ${EXCLUDE_OPTION} square.hip)

target_include_directories(square2 PRIVATE ../../common)

# Set RPATH so executable can find HIP libraries at runtime
if(UNIX)
  set_target_properties(square2 PROPERTIES
    BUILD_RPATH "${ROCM_PATH}/lib"
    INSTALL_RPATH "${ROCM_PATH}/lib"
  )
endif()

if(TARGET build_cookbook)
add_dependencies(build_cookbook square2)
endif()
