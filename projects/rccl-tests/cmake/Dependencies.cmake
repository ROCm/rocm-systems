# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

# Dependencies

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# GIT

# Test dependencies
include(FetchContent)

# Find or download/install rocm-cmake project
find_package(ROCmCMakeBuildTools 0.11.0 CONFIG QUIET PATHS "${ROCM_PATH}")
if(NOT ROCmCMakeBuildTools_FOUND)
    find_package(ROCM 0.7.3 CONFIG QUIET PATHS "${ROCM_PATH}") # deprecated fallback
    if(NOT ROCM_FOUND)
        message(STATUS "ROCmCMakeBuildTools not found. Fetching...")
        set(PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern)
        set(rocm_cmake_tag "rocm-6.4.0" CACHE STRING "rocm-cmake tag to download")
        FetchContent_Declare(
            rocm-cmake
            GIT_REPOSITORY https://github.com/ROCm/rocm-cmake.git
            GIT_TAG ${rocm_cmake_tag}
            SOURCE_SUBDIR "DISABLE ADDING TO BUILD"
        )
        FetchContent_MakeAvailable(rocm-cmake)
        find_package(ROCmCMakeBuildTools CONFIG REQUIRED NO_DEFAULT_PATH PATHS "${rocm-cmake_SOURCE_DIR}")
    endif()
endif()

# Find available local ROCM targets
# NOTE: This will eventually be part of ROCm-CMake and should be removed at that time
function(rocm_local_targets VARIABLE)
  set(${VARIABLE} "NOTFOUND" PARENT_SCOPE)
  find_program(_rocm_agent_enumerator rocm_agent_enumerator HINTS ocm/bin ENV ROCM_PATH)
  if(NOT _rocm_agent_enumerator STREQUAL "_rocm_agent_enumerator-NOTFOUND")
    execute_process(
      COMMAND "${_rocm_agent_enumerator}"
      RESULT_VARIABLE _found_agents
      OUTPUT_VARIABLE _rocm_agents
      ERROR_QUIET
      )
    if (_found_agents EQUAL 0)
      string(REPLACE "\n" ";" _rocm_agents "${_rocm_agents}")
      unset(result)
      foreach (agent IN LISTS _rocm_agents)
        if (NOT agent STREQUAL "gfx000")
          list(APPEND result "${agent}")
        endif()
      endforeach()
      if(result)
        list(REMOVE_DUPLICATES result)
        set(${VARIABLE} "${result}" PARENT_SCOPE)
      endif()
    endif()
  endif()
endfunction()

# Iterate over the "source" list and check if there is a duplicate file name
# NOTE: This is due to compiler bug '--save-temps' and can be removed when fix availabe
function(add_file_unique FILE_LIST FILE)
  get_filename_component(FILE_NAME "${FILE}" NAME)

  # Iterate over whatever is in the list so far
  foreach(curr_file IN LISTS ${FILE_LIST})
    get_filename_component(curr_file_name ${curr_file} NAME)

    # Check if duplicate
    if(${FILE_NAME} STREQUAL ${curr_file_name})
      get_filename_component(DIR_PATH "${FILE}" DIRECTORY)
      get_filename_component(FILE_NAME_WE "${FILE}" NAME_WE)
      get_filename_component(FILE_EXT "${FILE}" EXT)

      # Construct a new file name by adding _tmp
      set(HIP_FILE "${DIR_PATH}/${FILE_NAME_WE}_tmp${FILE_EXT}" PARENT_SCOPE)
    endif()
  endforeach()
endfunction()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMClients)
include(ROCMHeaderWrapper)
