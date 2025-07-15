# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#
# rocprofiler-systems tests
#
if(NOT "${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}")
    # ----------------------------------------------------------------------------------------#
    #
    # Ran when building whole project
    #
    # ----------------------------------------------------------------------------------------#
    include_guard(GLOBAL)

    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-testing.cmake)

    # test groups
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-config-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-instrument-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-pthread-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rocm-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-user-api-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-mpi-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-kokkos-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-openmp-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-code-coverage-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-fork-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-time-window-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-attach-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rccl-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-overflow-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-annotate-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-causal-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-python-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-decode-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-nic-perf.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-roctx-tests.cmake)

    add_subdirectory(source)
else()
    # ----------------------------------------------------------------------------------------#
    #
    # Ran when building test exclusive build
    #
    # ----------------------------------------------------------------------------------------#

    cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

    # Set support for Fortran tests
    find_program(GFORTRAN_EXECUTABLE NAMES gfortran)
    if(GFORTRAN_EXECUTABLE)
        set(ENABLE_FORTRAN_TESTS TRUE CACHE BOOL "Enable Fortran CTest support")
        project(rocprofiler-systems-tests LANGUAGES C CXX Fortran)
    else()
        set(ENABLE_FORTRAN_TESTS FALSE CACHE BOOL "Enable Fortran CTest support")
        project(rocprofiler-systems-tests LANGUAGES CXX)
    endif()

    if("${CMAKE_BUILD_TYPE}" STREQUAL "")
        set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
    endif()

    set(CTEST_ONLY_BUILD TRUE CACHE BOOL "CTest only build")

    # Paths for CTests and cmake files
    set(CMAKE_D "${PROJECT_SOURCE_DIR}/../cmake")
    get_filename_component(CMAKE_D "${CMAKE_D}" REALPATH)
    set(CTEST_SOURCE_D "${PROJECT_SOURCE_DIR}/../examples")
    get_filename_component(CTEST_SOURCE_D "${CTEST_SOURCE_D}" REALPATH)
    set(CTEST_CONFIG_D "${PROJECT_SOURCE_DIR}")
    get_filename_component(CTEST_CONFIG_D "${CTEST_CONFIG_D}" REALPATH)
    set(SCRIPTS_D "${PROJECT_SOURCE_DIR}/../scripts")
    get_filename_component(SCRIPTS_D "${SCRIPTS_D}" REALPATH)
    set(CLI_CTEST_D "${PROJECT_SOURCE_DIR}/../source/bin/tests")
    get_filename_component(CLI_CTEST_D "${CLI_CTEST_D}" REALPATH)

    set(CMAKE_MODULE_PATH ${CMAKE_D} ${CMAKE_D}/Modules ${CMAKE_MODULE_PATH})

    find_package(rocprofiler-systems REQUIRED) # System
    find_package(ROCmVersion REQUIRED) # Module path

    set(CMAKE_INSTALL_DEFAULT_COMPONENT_NAME "tests")

    enable_testing()
    include(CTest)

    # Store python bin for causal in this build file
    set(CMAKE_INSTALL_BINDIR "causal" CACHE PATH "Default CTest binaries install path")

    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH} CACHE PATH "Default ROCm installation path")
    elseif(ROCM_PATH)
        message("-- INFO:ROCM_PATH Set -- ${ROCM_PATH}")
    else()
        set(ROCM_PATH /opt/rocm CACHE PATH "Default ROCm installation path")
    endif()

    # Find the binaries used for testing
    set(ROCPROFILER_EXECUTABLES
        rocprof-sys-sample
        rocprof-sys-instrument
        rocprof-sys-run
        rocprof-sys-causal
        rocprof-sys-avail
    )

    foreach(EXEC ${ROCPROFILER_EXECUTABLES})
        find_program(${EXEC}_PATH NAMES ${EXEC} PATHS ${ROCM_PATH}/bin NO_DEFAULT_PATH)
        if(NOT ${EXEC}_PATH)
            message(
                FATAL_ERROR
                "Could not find ${EXEC} in ROCm search paths: ${ROCM_PATH}/bin"
            )
        endif()
        string(REPLACE "rocprof-sys-" "rocprofiler-systems-" TARGET_NAME ${EXEC})
        add_executable(${TARGET_NAME} IMPORTED)
        set_target_properties(${TARGET_NAME} PROPERTIES IMPORTED_LOCATION ${${EXEC}_PATH})
    endforeach()

    # Set compile launcher
    set(ROCPROFSYS_COMPILE_LAUNCHER "${SCRIPTS_D}/rocprof-sys-launch-compiler")
    include(MacroUtilities) #TODO: Determine if the whole thing is actually needed
    include(CTestSettings)

    # Test source code dir
    set(ROCPROFSYS_DISABLE_EXAMPLES python CACHE STRING "Disable these examples") # Blocked for now
    add_subdirectory(${CTEST_SOURCE_D} ${CMAKE_BINARY_DIR}/examples)

    # CTests
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-testing.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-config-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-instrument-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-pthread-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rocm-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-user-api-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-mpi-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-kokkos-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-openmp-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-code-coverage-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-fork-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-time-window-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-attach-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-rccl-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-overflow-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-annotate-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-causal-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-python-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-decode-tests.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-nic-perf.cmake)
    include(${CMAKE_CURRENT_LIST_DIR}/rocprof-sys-roctx-tests.cmake)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/source)
    add_subdirectory(${CLI_CTEST_D} ${CMAKE_BINARY_DIR}/cli-tests)
endif()

# =========================================================================================#
# Important information:
#   - ROCPROFSYS_USE_PYTHON is set to OFF by default in CTestSettings.cmake until rocprofsys module is ready
#   - Must check if the CI portion in CTestSettings.cmake is required
#   - Must check if we want to use rocprofiler_systems_message() after MacroUtilities is included
#   - Investigate how ${COMMAND_PREFIX} is used in tests (to avoid requiring MPI)
#   - Investifate RCCL
#   - Investigate if find_package(ROCmVersion) needs the REQUIRED flag
