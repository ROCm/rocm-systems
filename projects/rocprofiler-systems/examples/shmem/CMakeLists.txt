# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)
    if(
        "rocprofiler-systems-shmem-examples" IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

# oshcc is the OpenSHMEM C compiler wrapper (e.g. from Open MPI); it provides
# the correct include paths and libraries for shmem.h and OpenSHMEM runtime.
find_program(OSHCC_EXECUTABLE NAMES oshcc)
if(NOT OSHCC_EXECUTABLE)
    if("${CMAKE_PROJECT_NAME}" STREQUAL "rocprofiler-systems" AND "$ENV{ROCPROFSYS_CI}")
        set(_MSG_TYPE STATUS) # don't generate warnings during CI
    else()
        set(_MSG_TYPE AUTHOR_WARNING)
    endif()
    message(
        ${_MSG_TYPE}
        "oshcc not found. Skipping SHMEM examples (shmem_hello, shmem_pingpong)."
    )
    return()
endif()

# Build with oshcc so shmem.h and OpenSHMEM libs are found. Output to project build root.
set(_SHMEM_BINDIR ${CMAKE_BINARY_DIR})
set(_SHMEM_SRCDIR ${CMAKE_CURRENT_SOURCE_DIR})
set(_SHMEM_HELLO_EXE ${_SHMEM_BINDIR}/shmem_hello${CMAKE_EXECUTABLE_SUFFIX})
set(_SHMEM_PINGPONG_EXE ${_SHMEM_BINDIR}/shmem_pingpong${CMAKE_EXECUTABLE_SUFFIX})

add_custom_command(
    OUTPUT ${_SHMEM_HELLO_EXE}
    COMMAND ${OSHCC_EXECUTABLE} ${_SHMEM_SRCDIR}/shmem_hello.c -o ${_SHMEM_HELLO_EXE}
    DEPENDS ${_SHMEM_SRCDIR}/shmem_hello.c
    COMMENT "Building shmem_hello with oshcc"
    VERBATIM
)
add_custom_target(shmem_hello ALL DEPENDS ${_SHMEM_HELLO_EXE})

add_custom_command(
    OUTPUT ${_SHMEM_PINGPONG_EXE}
    COMMAND
        ${OSHCC_EXECUTABLE} ${_SHMEM_SRCDIR}/shmem_pingpong.c -o ${_SHMEM_PINGPONG_EXE}
    DEPENDS ${_SHMEM_SRCDIR}/shmem_pingpong.c
    COMMENT "Building shmem_pingpong with oshcc"
    VERBATIM
)
add_custom_target(shmem_pingpong ALL DEPENDS ${_SHMEM_PINGPONG_EXE})

# Export paths for rocprof-sys-shmem-tests.cmake (same project)
set(ROCPROFSYS_SHMEM_HELLO_EXE
    "${_SHMEM_HELLO_EXE}"
    CACHE INTERNAL
    "Path to shmem_hello executable"
)
set(ROCPROFSYS_SHMEM_PINGPONG_EXE
    "${_SHMEM_PINGPONG_EXE}"
    CACHE INTERNAL
    "Path to shmem_pingpong executable"
)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(
        PROGRAMS ${_SHMEM_HELLO_EXE} ${_SHMEM_PINGPONG_EXE}
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
        COMPONENT rocprofiler-systems-examples
        OPTIONAL
    )
endif()
