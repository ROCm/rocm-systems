# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

if(NOT ROCPROFSYS_USE_PYTHON)
    rocprofiler_systems_message(FATAL_ERROR "CTest suite requires Python to be enabled")
endif()

# Find pytest executable
set(_pytest_hints "")
if(ROCPROFSYS_TEST_PYTHON_VERSIONS AND ROCPROFSYS_PYTHON_ROOT_DIRS)
    # When both ROCPROFSYS_PYTHON_VERSIONS and ROCPROFSYS_PYTHON_PREFIX are set,
    # ROCPROFSYS_PYTHON_ROOT_DIRS will contain the list of versioned python prefixes
    foreach(_pyroot ${ROCPROFSYS_PYTHON_ROOT_DIRS})
        foreach(_ver ${ROCPROFSYS_TEST_PYTHON_VERSIONS})
            if(EXISTS "${_pyroot}/bin/python${_ver}")
                list(APPEND _pytest_hints "${_pyroot}/bin")
                break()
            endif()
        endforeach()
    endforeach()
elseif(ROCPROFSYS_PYTHON_ROOT_DIRS)
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

# Configure test marker and keyword inclusions/exclusions

set(_ROCPROFSYS_PYTEST_TEST_MARKERS "")
set(_ROCPROFSYS_PYTEST_TEST_KEYWORDS "")

function(ROCPROFILER_SYSTEMS_CONFIGURE_TEST_VAR PYTEST_VAR LIST_TO_ADD TO_INCLUDE)
    set(_current "${${PYTEST_VAR}}")
    foreach(_to_add ${LIST_TO_ADD})
        string(REPLACE "-" "_" _to_add "${_to_add}")
        if(_current)
            string(APPEND _current " and ")
        endif()
        if(${TO_INCLUDE})
            string(APPEND _current "${_to_add}")
        else()
            string(APPEND _current "not ${_to_add}")
        endif()
    endforeach()
    set(${PYTEST_VAR} "${_current}" PARENT_SCOPE)
endfunction()

rocprofiler_systems_configure_test_var(_ROCPROFSYS_PYTEST_TEST_MARKERS "${ROCPROFSYS_DISABLE_EXAMPLES}" FALSE)
rocprofiler_systems_configure_test_var(_ROCPROFSYS_PYTEST_TEST_MARKERS "${ROCPROFSYS_TEST_LABELS_INCLUDE}" TRUE)
rocprofiler_systems_configure_test_var(_ROCPROFSYS_PYTEST_TEST_MARKERS "${ROCPROFSYS_TEST_LABELS_EXCLUDE}" FALSE)
rocprofiler_systems_configure_test_var(_ROCPROFSYS_PYTEST_TEST_KEYWORDS "${ROCPROFSYS_TEST_KEYWORDS_INCLUDE}" TRUE)
rocprofiler_systems_configure_test_var(_ROCPROFSYS_PYTEST_TEST_KEYWORDS "${ROCPROFSYS_TEST_KEYWORDS_EXCLUDE}" FALSE)

if(NOT "${_ROCPROFSYS_PYTEST_TEST_MARKERS}" STREQUAL "")
    rocprofiler_systems_message(STATUS
        "PyTest marker command: ${_ROCPROFSYS_PYTEST_TEST_MARKERS}"
    )
endif()

if(NOT "${_ROCPROFSYS_PYTEST_TEST_KEYWORDS}" STREQUAL "")
    rocprofiler_systems_message(STATUS
        "PyTest keyword command: ${_ROCPROFSYS_PYTEST_TEST_KEYWORDS}"
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

# Python versioned tests look for the versioned libpyrocprofsys to be found
if(TARGET libpyrocprofsys)
    list(APPEND PYTEST_DEPENDENCIES libpyrocprofsys)
endif()

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
    -q
    -p
    no:cacheprovider
)

# Note: ROCPROFSYS_PYTHON_VERSIONS will inherit the value of ROCPROFSYS_PYTHON_VERSION
if(ROCPROFSYS_PYTHON_VERSIONS AND NOT ROCPROFSYS_TEST_PYTHON_VERSIONS)
    list(JOIN ROCPROFSYS_PYTHON_VERSIONS "\\;" _py_versions_escaped)
    list(APPEND _generate_args "--python-versions=${_py_versions_escaped}")
elseif(ROCPROFSYS_TEST_PYTHON_VERSIONS)
    # Rhel 8.10 has python 3.6 by default, which we need to exclude
    list(JOIN ROCPROFSYS_TEST_PYTHON_VERSIONS "\\;" _py_versions_escaped)
    list(APPEND _generate_args "--python-versions=${_py_versions_escaped}")
endif()

if(ROCPROFSYS_PYTHON_ROOT_DIRS)
    list(JOIN ROCPROFSYS_PYTHON_ROOT_DIRS "\\;" _py_roots_escaped)
    list(APPEND _generate_args "--python-root-dirs=${_py_roots_escaped}")
endif()

if(NOT "${_ROCPROFSYS_PYTEST_TEST_MARKERS}" STREQUAL "")
    list(APPEND _generate_args "-m" "${_ROCPROFSYS_PYTEST_TEST_MARKERS}")
endif()

if(NOT "${_ROCPROFSYS_PYTEST_TEST_KEYWORDS}" STREQUAL "")
    list(APPEND _generate_args "-k" "${_ROCPROFSYS_PYTEST_TEST_KEYWORDS}")
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
