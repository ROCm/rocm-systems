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

cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-rccl-example LANGUAGES CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

function(rccl_message _MSG_TYPE)
    if(
        "${CMAKE_PROJECT_NAME}" STREQUAL "rocprofiler-systems"
        AND "$ENV{ROCPROFSYS_CI}"
        AND "${_MSG_TYPE}" MATCHES "WARNING"
    )
        set(_MSG_TYPE STATUS) # don't generate warnings during CI
    endif()
    if("${CMAKE_PROJECT_NAME}" STREQUAL "rocprofiler-systems")
        rocprofiler_systems_message(${_MSG_TYPE} ${ARGN})
    else()
        message(${_MSG_TYPE} ${ARGN})
    endif()
endfunction()

find_package(hip HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm)

if(NOT hip_FOUND)
    rccl_message(AUTHOR_WARNING "${PROJECT_NAME} skipped. Missing HIP...")
    return()
endif()

if(
    "${CMAKE_PROJECT_NAME}" STREQUAL "rocprofiler-systems"
    AND ("$ENV{ROCPROFSYS_CI}" OR ROCPROFSYS_CI OR ROCPROFSYS_BUILD_CI)
)
    # avoid generating warning in CI
    find_package(rccl HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm QUIET)
else()
    find_package(rccl HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm)
endif()

if(NOT rccl_FOUND)
    rccl_message(AUTHOR_WARNING "${PROJECT_NAME} skipped. Missing RCCL...")
    return()
endif()

if(hip_FOUND AND rccl_FOUND)
    # Use existing rccl-tests source in the project binary directory
    set(rccl-tests_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rccl-tests")
    set(rccl-tests_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/rccl-tests")
    file(COPY ${rccl-tests_SOURCE_DIR}/ DESTINATION ${rccl-tests_BUILD_DIR})

    if(NOT EXISTS "${rccl-tests_SOURCE_DIR}")
        rccl_message(AUTHOR_WARNING "${PROJECT_NAME} skipped. Missing rccl-tests source at ${rccl-tests_SOURCE_DIR}")
        return()
    endif()

    get_filename_component(rccl_ROOT_DIR "${rccl_INCLUDE_DIR}" DIRECTORY)

    rccl_message(STATUS "Building rccl-tests...")
    execute_process(
        COMMAND make HIP_HOME=${ROCM_PATH} RCCL_HOME=${rccl_ROOT_DIR}
        WORKING_DIRECTORY ${rccl-tests_BUILD_DIR}
        RESULT_VARIABLE _RCCL_BUILD_RET
        ERROR_VARIABLE _RCCL_BUILD_ERR
        OUTPUT_VARIABLE _RCCL_BUILD_OUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
    )

    if(NOT _RCCL_BUILD_RET EQUAL 0)
        rccl_message(${_RCCL_BUILD_OUT})
        rccl_message(AUTHOR_WARNING "Failed to build rccl-tests: ${_RCCL_BUILD_ERR}")
    else()
        file(GLOB RCCL_TEST_EXECUTABLES ${rccl-tests_BUILD_DIR}/build/*_perf)
        set(_RCCL_TEST_TARGETS)

        foreach(_EXE ${RCCL_TEST_EXECUTABLES})
            get_filename_component(_EXE_NAME "${_EXE}" NAME)
            execute_process(
                COMMAND
                    ${CMAKE_COMMAND} -E copy ${_EXE}
                    ${CMAKE_CURRENT_BINARY_DIR}/${_EXE_NAME}
            )
            add_executable(rccl-tests::${_EXE_NAME} IMPORTED GLOBAL)
            set_property(
                TARGET rccl-tests::${_EXE_NAME}
                PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/${_EXE_NAME}
            )
            list(APPEND _RCCL_TEST_TARGETS "rccl-tests::${_EXE_NAME}")
        endforeach()

        set(RCCL_TEST_TARGETS "${_RCCL_TEST_TARGETS}" CACHE INTERNAL "rccl-test targets")
    endif()
endif()

# Only add the finalize_wrap if not disabled
if(NOT DEFINED DISABLE_AMDSMI_WRAP)
    add_library(finalize_wrap SHARED libamdsmi_wrap.cpp)
    target_link_libraries(finalize_wrap PRIVATE dl)
    set_target_properties(
        finalize_wrap
        PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            OUTPUT_NAME finalize_wrap
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
endif()
