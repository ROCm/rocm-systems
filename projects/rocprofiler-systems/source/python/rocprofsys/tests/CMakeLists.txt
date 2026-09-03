# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

find_package(Python3 COMPONENTS Interpreter)

if(NOT Python3_FOUND)
    rocprofiler_systems_message(
        STATUS
        "Python3 interpreter not found: skipping rocprofsys-python-unit-tests"
    )
    return()
endif()

# pytest is not guaranteed to be present in the build environment.
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pytest"
    RESULT_VARIABLE _pytest_result
    OUTPUT_QUIET
    ERROR_QUIET
)

if(NOT _pytest_result EQUAL 0)
    rocprofiler_systems_message(
        STATUS
        "pytest not available: skipping rocprofsys-python-unit-tests"
    )
    return()
endif()

# Disabling the cache plugin and bytecode generation, and running from the
# binary directory, keeps test artifacts out of the source tree.
add_test(
    NAME rocprofsys-python-unit-tests
    COMMAND
        ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR} -v -p no:cacheprovider
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

set_tests_properties(
    rocprofsys-python-unit-tests
    PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
)
