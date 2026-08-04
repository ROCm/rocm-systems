# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(rocprofiler-systems-pytorch LANGUAGES NONE)

set(PYTHON_FILES pytorch_function_fit.py)

# Copied to the build root, where the test suite resolves example targets from.
foreach(_FILE ${PYTHON_FILES})
    configure_file(${PROJECT_SOURCE_DIR}/${_FILE} ${CMAKE_BINARY_DIR}/${_FILE} COPYONLY)

    if(ROCPROFSYS_INSTALL_EXAMPLES)
        install(
            PROGRAMS ${CMAKE_BINARY_DIR}/${_FILE}
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/rocprofiler-systems/examples
            COMPONENT rocprofiler-systems-examples
        )
    endif()
endforeach()
