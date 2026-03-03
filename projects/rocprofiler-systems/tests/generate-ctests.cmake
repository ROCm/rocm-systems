# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Validate that required vars are set
foreach(_var PYTEST_COMMAND PYTEST_BUILD_LOCATION OUTPUT_FILE CMAKE_BINARY_DIR)
    if(NOT DEFINED ${_var})
        message(FATAL_ERROR "Required variable ${_var} is not defined")
    endif()
endforeach()

# Run pytest --ctest-mode=collect to discover tests
#   PYTHONDONTWRITEBYTECODE=1  : Prevent __pycache__ creation
#   ROCPROFSYS_BUILD_DIR       : Tell conftest.py where the build is
message(STATUS "Collecting PyTest tests from ${PYTEST_BUILD_LOCATION}")
execute_process(
    COMMAND
        ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1
        ROCPROFSYS_BUILD_DIR=${CMAKE_BINARY_DIR} ${PYTEST_COMMAND}
        ${PYTEST_BUILD_LOCATION} --rootdir=${CMAKE_BINARY_DIR} --ctest-mode=collect -m
        "${PYTEST_MARKER_EXCLUSIONS}"
    OUTPUT_VARIABLE _full_pytest_list
    ERROR_VARIABLE _pytest_errors
    RESULT_VARIABLE _pytest_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT _pytest_result EQUAL 0)
    message(
        FATAL_ERROR
        "pytest collection returned non-zero: ${_pytest_result}\n"
        "Error msg: ${_pytest_errors}"
    )
endif()

# Pytest output generates some extra lines, we remove them to parse the JSON array
string(FIND "${_full_pytest_list}" "[" _json_start)
string(FIND "${_full_pytest_list}" "]" _json_end REVERSE)
if(_json_start LESS 0 OR _json_end LESS 0)
    message(
        FATAL_ERROR
        "Could not find JSON array in pytest output:\n${_full_pytest_list}"
    )
endif()
math(EXPR _json_len "${_json_end} - ${_json_start} + 1")
string(SUBSTRING "${_full_pytest_list}" ${_json_start} ${_json_len} _full_pytest_list)

string(JSON _test_count LENGTH "${_full_pytest_list}")

message(STATUS "Discovered ${_test_count} pytest tests")

if(_test_count EQUAL 0)
    message(FATAL_ERROR "No pytest tests were discovered!")
endif()

# Generate the output .cmake file header
file(
    WRITE "${OUTPUT_FILE}"
    "# Auto-generated pytest tests - DO NOT EDIT
# Generated at build time by generate-ctests.cmake
# Discovered ${_test_count} tests
#
# This file is included by CTest via the TEST_INCLUDE_FILES directory property.
# Regenerate by rebuilding the 'discover-pytests' target.

"
)

# Add Display-pytest-config (Always runs first)
# This runs pytest with --show-config-only to display the pytest configuration
file(
    APPEND "${OUTPUT_FILE}"
    "add_test(
    Display-pytest-config
    \"${PYTEST_COMMAND}\"
    \"${PYTEST_BUILD_LOCATION}\"
    \"--show-config-only\"
)
set_tests_properties(
    Display-pytest-config
    PROPERTIES
        LABELS \"pytest\"
        FIXTURES_SETUP \"pytest-header\"
)

"
)

# Generate add_test() for each discovered test
# Each JSON element has:
#   "id":         nodeid, e.g. "path/test_file.py::TestClass::test_method[param]"
#   "name":       CTest-friendly name, e.g. "InstrumentBinary-help"
#   "markers":    list of label strings, e.g. ["binary_rewrite", "user_api"]
#   "depends_on": (optional) list of CTest names this test depends on
math(EXPR _last_index "${_test_count} - 1")
foreach(_idx RANGE 0 ${_last_index})
    # Extract fields from the JSON object
    string(JSON _test_id GET "${_full_pytest_list}" ${_idx} "id")
    string(JSON _test_name GET "${_full_pytest_list}" ${_idx} "name")
    string(JSON _markers_json GET "${_full_pytest_list}" ${_idx} "markers")
    string(
        JSON _deps_json
        ERROR_VARIABLE _deps_err
        GET "${_full_pytest_list}"
        ${_idx}
        "depends_on"
    )
    # Build the absolute test path from the nodeid.
    # Nodeids are relative to pytest's rootdir (forced to CMAKE_BINARY_DIR).
    if(IS_ABSOLUTE "${_test_id}")
        set(_test_path "${_test_id}")
    else()
        set(_test_path "${CMAKE_BINARY_DIR}/${_test_id}")
    endif()

    # Build semicolon-separated labels list from the markers JSON array
    string(JSON _marker_count LENGTH "${_markers_json}")
    set(_labels "pytest")
    if(_marker_count GREATER 0)
        math(EXPR _last_marker "${_marker_count} - 1")
        foreach(_midx RANGE 0 ${_last_marker})
            string(JSON _marker GET "${_markers_json}" ${_midx})
            # For compatibility with old CTest label naming
            string(REPLACE "_" "-" _marker "${_marker}")
            string(APPEND _labels "\;${_marker}")
        endforeach()
    endif()

    # Build FIXTURES_REQUIRED from depends_on (if present) + global fixtures
    set(_fixtures_required "rocprofsys-global-tmp-files\;pytest-header")
    if(NOT _deps_err)
        string(JSON _dep_count LENGTH "${_deps_json}")
        if(_dep_count GREATER 0)
            math(EXPR _last_dep "${_dep_count} - 1")
            foreach(_didx RANGE 0 ${_last_dep})
                string(JSON _dep GET "${_deps_json}" ${_didx})
                string(APPEND _fixtures_required "\;${_dep}")
            endforeach()
        endif()
    endif()

    file(
        APPEND "${OUTPUT_FILE}"
        "add_test(
    [=[${_test_name}]=]
    \"${PYTEST_COMMAND}\"
    \"${_test_path}\"
    \"--ctest-mode=run\"
)
set_tests_properties(
    [=[${_test_name}]=]
    PROPERTIES
        FIXTURES_SETUP \"${_test_name}\"
        FIXTURES_REQUIRED \"${_fixtures_required}\"
        LABELS \"${_labels}\"
        DEPENDS \"pytest-generate-header\"
)

"
    )
endforeach()

# Global cleanup test for temporary files
# Runs once after ALL tests complete to clean up trace cache temporary files
file(
    APPEND "${OUTPUT_FILE}"
    "add_test(
    rocprofsys-cleanup-tmp-files
    sh -c
    \"find /tmp -maxdepth 1 -user \\$(whoami) \\\\( -name 'buffered_storage*.bin' -o -name 'metadata*.json' \\\\) -delete 2>/dev/null || true\"
)
set_tests_properties(
    rocprofsys-cleanup-tmp-files
    PROPERTIES
        FIXTURES_CLEANUP \"rocprofsys-global-tmp-files\"
        LABELS \"cleanup\;global\"
        TIMEOUT 30
)

"
)

message(STATUS "Generated ${_test_count} pytest tests in ${OUTPUT_FILE}")
