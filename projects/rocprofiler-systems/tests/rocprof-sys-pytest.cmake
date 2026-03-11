# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

if(NOT ROCPROFSYS_USE_PYTHON)
    rocprofiler_systems_message(FATAL_ERROR "CTest suite requires Python to be enabled")
endif()

# Find pytest executable
set(_pytest_hints "")
if(ROCPROFSYS_PYTHON_ROOT_DIRS)
    foreach(_pyroot ${ROCPROFSYS_PYTHON_ROOT_DIRS})
        list(APPEND _pytest_hints "${_pyroot}/bin")
    endforeach()
endif()

find_program(PYTEST_EXECUTABLE NAMES pytest HINTS ${_pytest_hints})
mark_as_advanced(PYTEST_EXECUTABLE)

if(NOT PYTEST_EXECUTABLE)
    rocprofiler_systems_message(FATAL_ERROR
        "pytest executable not found, cannot generate CTest suite. "
        "Install dependencies with: pip install -r requirements.txt"
    )
endif()

# Ensure proper pytest version
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

# Read requirements.txt for the minimum pytest version
set(_requirements_file "${CMAKE_SOURCE_DIR}/requirements.txt")
if(EXISTS "${_requirements_file}")
    file(STRINGS "${_requirements_file}" _req_lines REGEX "^pytest>=")
    if(_req_lines)
        list(GET _req_lines 0 _pytest_req)
        string(REGEX REPLACE "^pytest>=" "" PYTEST_MIN_VERSION "${_pytest_req}")
    endif()
endif()
if(NOT DEFINED PYTEST_MIN_VERSION)
    set(PYTEST_MIN_VERSION "7.4.0")
    rocprofiler_systems_message(FATAL_ERROR "DONT MERGE THIS")
endif()

if(PYTEST_VERSION VERSION_LESS "${PYTEST_MIN_VERSION}")
    rocprofiler_systems_message(FATAL_ERROR
        "pytest version ${PYTEST_VERSION} is less than the minimum "
        "required version of ${PYTEST_MIN_VERSION} (from requirements.txt)."
    )
endif()

# Set up marker exclusions
# This prevents certain CTests from being generated
set(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS_STRING "")
set(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS_LIST "")
macro(ROCPROFILER_SYSTEMS_ADD_PYTEST_MARKER_EXCLUSION MARKER_NAME)
    # PyTest requires markers to use "_" and not "-"
    # This is done to prevent having these tests "skipped" as it will not find
    #   the target executable.
    string(REPLACE "-" "_" _marker "${MARKER_NAME}")
    list(APPEND ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS_LIST "${_marker}")
    if(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS)
        string(APPEND ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS " and not ${_marker}")
    else()
        set(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS "not ${_marker}")
    endif()
endmacro()

set(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS "")
if(ROCPROFSYS_DISABLE_EXAMPLES)
    foreach(_marker ${ROCPROFSYS_DISABLE_EXAMPLES})
        rocprofiler_systems_add_pytest_marker_exclusion(${_marker})
    endforeach()
endif()

if(NOT ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS_LIST STREQUAL "")
    rocprofiler_systems_message(STATUS
        "PyTest tests with the following markers will be excluded from the CTest suite: ${ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS_LIST}"
    )
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# Pytest config requires that the base executables be found
set(PYTEST_DEPENDENCIES
    copy-pytest-files
    rocprofiler-systems-instrument
    rocprofiler-systems-sample
    rocprofiler-systems-run
    rocprofiler-systems-causal
    rocprofiler-systems-avail
)

# Filter out targets that don't exist (e.g. when certain components are disabled)
set(_valid_deps "")
foreach(_dep ${PYTEST_DEPENDENCIES})
    if(TARGET ${_dep})
        list(APPEND _valid_deps ${_dep})
    else()
        rocprofiler_systems_message(STATUS "Pytest dependency target '${_dep}' not found, skipping")
    endif()
endforeach()

# ---------------------------------------------------------------------------
# Build the pytest arguments
# ---------------------------------------------------------------------------
set(ROCPROFSYS_PYTEST_CTEST_FILE "${CMAKE_BINARY_DIR}/tests/CTestTestfile.cmake")
set(ROCPROFSYS_PYTEST_BUILD_DIR
    ${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/pytest
)

set(_generate_args
    "${PYTEST_EXECUTABLE}"
    "${ROCPROFSYS_PYTEST_BUILD_DIR}"
    --ctest-mode=generate
    --ctest-output-path
    "${ROCPROFSYS_PYTEST_CTEST_FILE}"
    --no-header
    --output-log=none
    -q
    -p
    no:xdist
    -p
    no:cacheprovider
)

# Note: ROCPROFSYS_PYTHON_VERSIONS will inherit the value of ROCPROFSYS_PYTHON_VERSION
if(ROCPROFSYS_PYTHON_VERSIONS)
    list(JOIN ROCPROFSYS_PYTHON_VERSIONS "\\;" _py_versions_escaped)
    list(APPEND _generate_args "--python-versions=${_py_versions_escaped}")
endif()

if(ROCPROFSYS_PYTHON_ROOT_DIRS)
    list(JOIN ROCPROFSYS_PYTHON_ROOT_DIRS "\\;" _py_roots_escaped)
    list(APPEND _generate_args "--python-root-dirs=${_py_roots_escaped}")
endif()

if(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS)
    list(APPEND _generate_args "-m" "${ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS}")
endif()

# ---------------------------------------------------------------------------
# Generate CTestTestfile.cmake
# ---------------------------------------------------------------------------
add_custom_command(
    OUTPUT "${ROCPROFSYS_PYTEST_CTEST_FILE}"
    COMMAND ${CMAKE_COMMAND} -E env PYTHONDONTWRITEBYTECODE=1 ${_generate_args}
    DEPENDS ${_valid_deps} ${ROCPROFSYS_PYTEST_PACKAGE_FILES} ${ROCPROFSYS_PYTEST_FILES}
    WORKING_DIRECTORY "${ROCPROFSYS_PYTEST_BUILD_DIR}"
    COMMENT "Generating CTest definitions from pytest suite"
    VERBATIM
)

add_custom_target(generate-pytest-ctests ALL DEPENDS "${ROCPROFSYS_PYTEST_CTEST_FILE}")

if(ROCPROFSYS_INSTALL_TESTING)
    install(
        FILES "${ROCPROFSYS_PYTEST_CTEST_FILE}"
        DESTINATION share/rocprofiler-systems/tests
        COMPONENT rocprofiler-systems-tests
    )
endif()
