# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(rocprofiler-systems-minimal LANGUAGES CXX)

# Support standalone builds
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(${CMAKE_CURRENT_LIST_DIR}/../cmake/standalone-helpers.cmake OPTIONAL)
endif()

add_executable(minimal-recursion ${CMAKE_CURRENT_SOURCE_DIR}/recursion.cpp)
# Keep recursive calls as real calls so the instrumenter sees every frame
# (noinline alone does not prevent tail-call / sibling-call optimization).
target_compile_options(minimal-recursion PRIVATE -fno-optimize-sibling-calls)

set(MINIMAL_EXAMPLES minimal-recursion)

foreach(MINIMAL_EXAMPLE IN LISTS MINIMAL_EXAMPLES)
    if(ROCPROFSYS_INSTALL_EXAMPLES AND TARGET ${MINIMAL_EXAMPLE})
        install(
            TARGETS ${MINIMAL_EXAMPLE}
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
            COMPONENT rocprofiler-systems-examples
        )
    endif()
endforeach()
