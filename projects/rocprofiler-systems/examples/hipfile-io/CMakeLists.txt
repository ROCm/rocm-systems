# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(rocprofiler-systems-hipfile-io-example LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

find_package(hip QUIET HINTS ${ROCmVersion_DIR} PATHS ${ROCmVersion_DIR})

# hip::hipfile is provided by the parent build's find_package(hipfile); for
# standalone example builds, try to locate it directly.
if(NOT TARGET hip::hipfile)
    find_package(
        hipfile
        QUIET
        HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
        PATHS ${ROCmVersion_DIR} ${ROCM_PATH}
    )
endif()

if(NOT TARGET hip::host OR NOT TARGET hip::hipfile)
    message(
        AUTHOR_WARNING
        "hipfile-io target could not be built (hip::host or hip::hipfile missing)"
    )
    return()
endif()

add_executable(hipfile-io hipfile-io.cpp)
target_compile_options(hipfile-io PRIVATE -W -Wall)
target_link_libraries(hipfile-io PRIVATE hip::host hip::hipfile)

if("${CMAKE_BUILD_TYPE}" MATCHES "Release")
    target_compile_options(hipfile-io PRIVATE -g1)
endif()

if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET hipfile-io)
    install(
        TARGETS hipfile-io
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
    )
endif()
