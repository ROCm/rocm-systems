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

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
list(APPEND CMAKE_MESSAGE_CONTEXT "rccl-tests")

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

# Find HIP
find_package(hip HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm QUIET)
if(NOT hip_FOUND)
    message(AUTHOR_WARNING "${PROJECT_NAME} skipped. Missing HIP...")
    return()
endif()

# Find RCCL
find_package(rccl HINTS ${ROCmVersion_DIR} ${ROCM_PATH} /opt/rocm QUIET)
if(NOT rccl_FOUND)
    message(AUTHOR_WARNING "rccl-tests skipped. Missing RCCL...")
    return()
else()
    message(STATUS "RCCL found: ${RCCL_INCLUDE_DIRS}")
    message(STATUS "RCCL library: ${RCCL_LIBRARIES}")
    message(STATUS "RCCL version: ${RCCL_VERSION}")
endif()

# Build rccl-tests
if(hip_FOUND AND rccl_FOUND)
    # Setup source and build directories
    set(rccl-tests_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rccl-tests")
    set(rccl-tests_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/rccl-tests")

    if(NOT EXISTS "${rccl-tests_SOURCE_DIR}")
        message(
            AUTHOR_WARNING
            "rccl-tests skipped. Missing rccl-tests source at ${rccl-tests_SOURCE_DIR}"
        )
        return()
    endif()

    # Copy source to build directory
    file(COPY ${rccl-tests_SOURCE_DIR}/ DESTINATION ${rccl-tests_BUILD_DIR})

    # Get RCCL root directory
    get_filename_component(rccl_ROOT_DIR "${rccl_INCLUDE_DIRS}" DIRECTORY)

    # Create a custom target that builds rccl-tests at build-time
    add_custom_target(
        build-rccl-tests
        ALL
        COMMAND make HIP_HOME=${ROCM_PATH} RCCL_HOME=${rccl_ROOT_DIR} -j
        WORKING_DIRECTORY ${rccl-tests_BUILD_DIR}
        COMMENT
            "Building rccl-tests with HIP_HOME=${ROCM_PATH} RCCL_HOME=${rccl_ROOT_DIR} ..."
        BYPRODUCTS
            ${rccl-tests_BUILD_DIR}/build/all_gather_perf
            ${rccl-tests_BUILD_DIR}/build/all_reduce_perf
            ${rccl-tests_BUILD_DIR}/build/alltoall_perf
            ${rccl-tests_BUILD_DIR}/build/alltoallv_perf
            ${rccl-tests_BUILD_DIR}/build/broadcast_perf
            ${rccl-tests_BUILD_DIR}/build/gather_perf
            ${rccl-tests_BUILD_DIR}/build/reduce_perf
            ${rccl-tests_BUILD_DIR}/build/reduce_scatter_perf
            ${rccl-tests_BUILD_DIR}/build/scatter_perf
            ${rccl-tests_BUILD_DIR}/build/sendrecv_perf
    )

    # Define the expected test executables
    set(RCCL_TEST_NAMES
        all_gather_perf
        all_reduce_perf
        alltoall_perf
        alltoallv_perf
        broadcast_perf
        gather_perf
        reduce_perf
        reduce_scatter_perf
        scatter_perf
        sendrecv_perf
    )

    set(_RCCL_TEST_TARGETS)
    foreach(_EXE_NAME ${RCCL_TEST_NAMES})
        set(_EXE_SRC_PATH "${rccl-tests_BUILD_DIR}/build/${_EXE_NAME}")
        set(_EXE_DEST_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_EXE_NAME}")

        # Copy executable from build dir to binary dir at build-time
        add_custom_command(
            OUTPUT ${_EXE_DEST_PATH}
            COMMAND
                ${CMAKE_COMMAND} -E copy_if_different ${_EXE_SRC_PATH} ${_EXE_DEST_PATH}
            DEPENDS build-rccl-tests
            COMMENT "Copying ${_EXE_NAME} to ${CMAKE_CURRENT_BINARY_DIR}"
        )

        # Create a target for this specific executable that gets built with ALL
        add_custom_target(copy-${_EXE_NAME} ALL DEPENDS ${_EXE_DEST_PATH})
        add_dependencies(copy-${_EXE_NAME} build-rccl-tests)

        # Create imported executable target
        add_executable(rccl-tests::${_EXE_NAME} IMPORTED GLOBAL)
        set_property(
            TARGET rccl-tests::${_EXE_NAME}
            PROPERTY IMPORTED_LOCATION ${_EXE_DEST_PATH}
        )

        # Make sure the imported target depends on the copy operation
        add_dependencies(rccl-tests::${_EXE_NAME} copy-${_EXE_NAME})

        list(APPEND _RCCL_TEST_TARGETS "rccl-tests::${_EXE_NAME}")

        if(ROCPROFSYS_INSTALL_EXAMPLES)
            install(
                PROGRAMS ${_EXE_DEST_PATH}
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
                COMPONENT rocprofiler-systems-examples
            )
        endif()
    endforeach()

    set(RCCL_TEST_TARGETS "${_RCCL_TEST_TARGETS}" CACHE INTERNAL "rccl-test targets")
endif()
