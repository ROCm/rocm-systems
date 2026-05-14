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

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

include(FetchContent)

if(NOT INSTALL_DEPENDENCIES)
    find_package(GTest 1.11)
endif()

if(NOT GTest_FOUND AND BUILD_TESTS OR INSTALL_DEPENDENCIES)
    if(CMAKE_CXX_COMPILER MATCHES ".*/hipcc$")
        # hip-clang cannot compile googlebenchmark for some reason
        set(COMPILER_OVERRIDE "-DCMAKE_CXX_COMPILER=g++")
    endif()

#       unset(GTEST_INCLUDE_DIR CACHE)
#	unset(GTEST_INCLUDE_DIRS CACHE)
    message(STATUS "GTest not found. Downloading and building GTest.")
    # Download, build and install googletest library
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")

    download_project(PROJ                googletest
                     GIT_REPOSITORY      https://github.com/google/googletest.git
                     GIT_TAG             release-1.12.0
                     INSTALL_DIR         ${GTEST_ROOT}
                     CMAKE_ARGS          -DBUILD_GTEST=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${COMPILER_OVERRIDE} -DBUILD_SHARED_LIBS=OFF
                     LOG_DOWNLOAD        TRUE
                     LOG_CONFIGURE       TRUE
                     LOG_BUILD           TRUE
                     LOG_INSTALL         TRUE
                     UPDATE_DISCONNECTED TRUE
    )
    set(GTEST_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/gtest/include CACHE PATH "")
    set(GMOCK_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/gmock/include CACHE PATH "")
    if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib)
        set(GTEST_BOTH_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib/libgtest.a;${CMAKE_CURRENT_BINARY_DIR}/gtest/lib/libgtest_main.a CACHE PATH "")
        set(GMOCK_BOTH_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib/libgmock.a;${CMAKE_CURRENT_BINARY_DIR}/gtest/lib/libgmock_main.a CACHE PATH "")
    elseif(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib64)
        set(GTEST_BOTH_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib64/libgtest.a;${CMAKE_CURRENT_BINARY_DIR}/gtest/lib64/libgtest_main.a CACHE PATH "")
        set(GMOCK_BOTH_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/gtest/lib64/libgmock.a;${CMAKE_CURRENT_BINARY_DIR}/gtest/lib64/libgmock_main.a CACHE PATH "")
    else()
        message(FATAL_ERROR "Cannot find gtest library installation path.")
    find_package(GTest REQUIRED CONFIG PATHS ${GTEST_ROOT})
    find_package(GMock REQUIRED CONFIG PATHS ${GTEST_ROOT})
    endif()
elseif(GTest_FOUND AND BUILD_TESTS)
    set(GTEST_BOTH_LIBRARIES "GTest::gtest;GTest::gtest_main")
    set(GMOCK_BOTH_LIBRARIES "GTest::gmock;GTest::gmock_main")
endif()

# Find or download/install rocm-cmake project
set( PROJECT_EXTERN_DIR ${CMAKE_CURRENT_BINARY_DIR}/extern )
find_package(ROCM 0.7.3 QUIET CONFIG PATHS /opt/rocm)
if(NOT ROCM_FOUND)
    set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
    file(
        DOWNLOAD https://github.com/ROCm/rocm-cmake/archive/${rocm_cmake_tag}.zip
        ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
        STATUS rocm_cmake_download_status LOG rocm_cmake_download_log
    )
    list(GET rocm_cmake_download_status 0 rocm_cmake_download_error_code)
    if(rocm_cmake_download_error_code)
        message(FATAL_ERROR "Error: downloading "
            "https://github.com/ROCm/rocm-cmake/archive/${rocm_cmake_tag}.zip failed "
            "error_code: ${rocm_cmake_download_error_code} "
            "log: ${rocm_cmake_download_log} "
        )
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
        WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}
        RESULT_VARIABLE rocm_cmake_unpack_error_code
    )
    execute_process( COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${PROJECT_EXTERN_DIR}/rocm-cmake .
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} )
    execute_process( COMMAND ${CMAKE_COMMAND} --build rocm-cmake-${rocm_cmake_tag} --target install
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR})

    if(rocm_cmake_unpack_error_code)
        message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip failed")
    endif()
    find_package( ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR}/rocm-cmake )
endif()

set(CMAKE_INSTALL_LIBDIR lib CACHE STRING "Define install directory for libraries" FORCE)

# Find or download/install fmt
find_package(fmt QUIET)
if(NOT fmt_FOUND)
    set(FMT_INSTALL OFF)
    message(STATUS "fmt not found, fetching from source...")
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt
        GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281 # 10.2.1
    )
    FetchContent_MakeAvailable(fmt)
else()
    message(STATUS "Using system fmt")
    get_target_property(FMT_INCLUDE_DIRS fmt::fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "fmt include directories: ${FMT_INCLUDE_DIRS}")
endif()

# Find available local ROCM targets
# NOTE: This will eventually be part of ROCm-CMake and should be removed at that time
function(rocm_local_targets VARIABLE)
  set(${VARIABLE} "NOTFOUND" PARENT_SCOPE)
  find_program(_rocm_agent_enumerator rocm_agent_enumerator HINTS /opt/rocm/bin ENV ROCM_PATH)
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
