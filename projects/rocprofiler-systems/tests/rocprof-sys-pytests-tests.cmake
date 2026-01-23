# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# Discover pytests and wrap them in ctests

# TODO: Enable only if ROCPROFSYS_USE_PYTHON...
if(NOT ROCPROFSYS_USE_ROCM)
    rocprofiler_systems_message(AUTHOR_WARNING
        "Pytest suite requires ROCm to be enabled."
    )
    return()
endif()

set(PYTEST_SOURCE_LOCATION "${CMAKE_CURRENT_LIST_DIR}/pytest/")
set(PYTEST_BUILD_LOCATION "${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/pytest/")

set(_pytest_hints "")
if(ROCPROFSYS_PYTHON_ROOT_DIRS)
    foreach(_pyroot ${ROCPROFSYS_PYTHON_ROOT_DIRS})
        list(APPEND _pytest_hints "${_pyroot}/bin")
    endforeach()
endif()

find_program(PYTEST_EXECUTABLE NAMES pytest HINTS ${_pytest_hints})
mark_as_advanced(PYTEST_EXECUTABLE)

if(PYTEST_EXECUTABLE)
    set(pytest_COMMAND "${PYTEST_EXECUTABLE}")
    execute_process(
        COMMAND "${PYTEST_EXECUTABLE}" --version
        OUTPUT_VARIABLE _version
        ERROR_VARIABLE _version
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_version MATCHES "pytest (version )?([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(PYTEST_VERSION "${CMAKE_MATCH_2}")
        rocprofiler_systems_message(STATUS "Found pytest ${PYTEST_VERSION} at: ${PYTEST_EXECUTABLE}")
    else()
        rocprofiler_systems_message(FATAL_ERROR
            "Could not determine pytest version. Output: ${_version}"
        )
    endif()
else()
    rocprofiler_systems_message(FATAL_ERROR
        "Could not find pytest. Searched PATH and hints: ${_pytest_hints}\n"
        "Install with: python -m pip install -r requirements.txt"
    )
endif()

if(PYTEST_VERSION VERSION_LESS 7.4.0)
    rocprofiler_systems_message(FATAL_ERROR
        "pytest version ${PYTEST_VERSION} is less than 7.4.0. Please upgrade to 7.4.0 or later."
    )
endif()

# Build marker exclusion string from ROCPROFSYS_DISABLE_EXAMPLES
set(_pytest_marker_exclusions "")
if(ROCPROFSYS_DISABLE_EXAMPLES)
    foreach(_marker ${ROCPROFSYS_DISABLE_EXAMPLES})
        # Convert hyphens to underscores for pytest marker names
        string(REPLACE "-" "_" _marker "${_marker}")
        if(_pytest_marker_exclusions)
            string(APPEND _pytest_marker_exclusions " and not ${_marker}")
        else()
            set(_pytest_marker_exclusions "not ${_marker}")
        endif()
    endforeach()
endif()

# -------------------------------------------------------------------------------------- #
# Collect and create tests
# -------------------------------------------------------------------------------------- #

# Collect pytest tests
execute_process(
    COMMAND
        ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
        ROCPROFSYS_BUILD_DIR=${CMAKE_BINARY_DIR} ${pytest_COMMAND}
        ${PYTEST_SOURCE_LOCATION} --collect-only -q -m "${_pytest_marker_exclusions}"
    OUTPUT_VARIABLE _full_pytest_list
    ERROR_VARIABLE _pytest_errors
    RESULT_VARIABLE _pytest_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Debug: show collection results
if(NOT _pytest_result EQUAL 0)
    rocprofiler_systems_message(WARNING
        "pytest collection returned non-zero: ${_pytest_result}\n"
        "Errors: ${_pytest_errors}"
    )
endif()

rocprofiler_systems_message(STATUS "pytest collection output:\n${_full_pytest_list}")

string(REPLACE "\n" ";" PYTEST_TESTS "${_full_pytest_list}")

# Debug: show raw list before filtering
list(LENGTH PYTEST_TESTS _raw_count)
rocprofiler_systems_message(STATUS "Raw pytest entries: ${_raw_count}")

list(FILTER PYTEST_TESTS INCLUDE REGEX "^tests/pytest/test_.*\\.py::.*")

# Debug: show filtered list
list(LENGTH PYTEST_TESTS _filtered_count)
rocprofiler_systems_message(STATUS "Filtered pytest tests: ${_filtered_count}")

if(_filtered_count EQUAL 0)
    rocprofiler_systems_message(FATAL_ERROR "No pytest tests were discovered!")
endif()

# Add to the test suite
foreach(PYTEST_TEST ${PYTEST_TESTS})
    # Strip "tests/pytest/" prefix since PYTEST_BUILD_LOCATION already includes it
    string(REGEX REPLACE "^tests/pytest/" "" PYTEST_TEST_FILE "${PYTEST_TEST}")

    # Reformat test name from the part after .py::
    string(REGEX REPLACE "^.*\\.py::" "" TEST_NAME "${PYTEST_TEST}")
    string(REGEX REPLACE "::" "-" TEST_NAME "${TEST_NAME}")
    string(REGEX REPLACE "_" "-" TEST_NAME "${TEST_NAME}")
    string(REGEX REPLACE "-test-" "-" TEST_NAME "${TEST_NAME}")
    string(REPLACE "[" "-" TEST_NAME "${TEST_NAME}")
    string(REPLACE "]" "" TEST_NAME "${TEST_NAME}")
    string(REGEX REPLACE "^Test" "" TEST_NAME "${TEST_NAME}")

    add_test(
        NAME ${TEST_NAME}
        COMMAND
            ${pytest_COMMAND} ${PYTEST_BUILD_LOCATION}/${PYTEST_TEST_FILE}
            --show-output-on-subtest-fail --ctest-integration
    )
    # Ensure cleanup test runs after all pytest tests
    set_tests_properties(
        ${TEST_NAME}
        PROPERTIES FIXTURES_REQUIRED rocprofsys-global-tmp-files LABELS "pytest"
    )
endforeach()
