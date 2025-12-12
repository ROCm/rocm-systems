# Copyright (c) 2019-2026 Advanced Micro Devices, Inc. All rights reserved.
# Simple CMake script to extract ROCm version

if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

# Method 1: Use ROCm's rocm-core CMake package version file
set(ROCM_VERSION_FILE "${ROCM_PATH}/lib/cmake/rocm-core/rocm-core-config-version.cmake")
if(EXISTS "${ROCM_VERSION_FILE}")
  # Read and extract PACKAGE_VERSION from the version cmake file
  file(STRINGS "${ROCM_VERSION_FILE}" version_line REGEX "^set\\(PACKAGE_VERSION")
  if(version_line)
    string(REGEX MATCH "\"([0-9]+\\.[0-9]+\\.[0-9]+)\"" version_match "${version_line}")
    if(CMAKE_MATCH_1)
      execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${CMAKE_MATCH_1}")
      return()
    endif()
  endif()
endif()

# Method 2: Fallback to reading .info/version file
if(EXISTS "${ROCM_PATH}/.info/version")
  file(READ "${ROCM_PATH}/.info/version" rocm_version_string)
  string(REGEX MATCH "([0-9]+)\\.([0-9]+)\\.([0-9]+)" rocm_version_matches ${rocm_version_string})
  if(rocm_version_matches)
    execute_process(COMMAND ${CMAKE_COMMAND} -E echo "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
    return()
  endif()
endif()

# Failed to detect version
message(FATAL_ERROR "Failed to detect ROCm version from ${ROCM_PATH}")
