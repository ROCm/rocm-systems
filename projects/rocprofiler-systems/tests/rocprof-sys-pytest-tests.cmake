# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -------------------------------------------------------------------------------------- #
#
# Generates a CTest for every PyTest collected
#
# -------------------------------------------------------------------------------------- #

if(NOT ROCPROFSYS_USE_PYTHON)
    rocprofiler_systems_message(AUTHOR_WARNING
            "CTest suite requires Python to be enabled"
    )
    return()
endif()

set(ROCPROFSYS_PYTEST_BUILD_DIR
    "${CMAKE_BINARY_DIR}/share/rocprofiler-systems/tests/pytest"
)

set(_pytest_hints "")
if(ROCPROFSYS_PYTHON_ROOT_DIRS)
    foreach(_pyroot ${ROCPROFSYS_PYTHON_ROOT_DIRS})
        list(APPEND _pytest_hints "${_pyroot}/bin")
    endforeach()
endif()

find_program(PYTEST_EXECUTABLE NAMES pytest HINTS ${_pytest_hints})
mark_as_advanced(PYTEST_EXECUTABLE)

if(NOT PYTEST_EXECUTABLE)
    rocprofiler_systems_message(AUTHOR_WARNING
        "pytest executable not found, cannot generate CTest suite"
    )
    return()
endif()

# Get pytest version
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

if(PYTEST_VERSION VERSION_LESS 7.4.0)
    rocprofiler_systems_message(FATAL_ERROR
        "pytest version ${PYTEST_VERSION} is less than the minimum required version of 7.4.0."
    )
endif()

# Build marker exclusion string from ROCPROFSYS_DISABLE_EXAMPLES
# PyTest requires markers to use "_" and not "-"
# This is done to prevent having these tests "skipped" as it will not find
#   the target executable.
set(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS "")
set(_excluded_markers "")
if(ROCPROFSYS_DISABLE_EXAMPLES)
    foreach(_marker ${ROCPROFSYS_DISABLE_EXAMPLES})
        # Convert hyphens to underscores for pytest marker names
        string(REPLACE "-" "_" _marker "${_marker}")
        list(APPEND _excluded_markers "${_marker}")
        if(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS)
            string(APPEND ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS " and not ${_marker}")
        else()
            set(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS "not ${_marker}")
        endif()
    endforeach()
endif()

rocprofiler_systems_message(STATUS
    "PyTest tests with the following markers will be excluded from the CTest suite due to ROCPROFSYS_DISABLE_EXAMPLES: ${_excluded_markers}"
)

# -------------------------------------------------------------------------------------- #
# Generate CTests (via generate-ctests.cmake)
# -------------------------------------------------------------------------------------- #

set(ROCPROFSYS_DEFAULT_PYTEST_OUTPUT_DIR "${CMAKE_BINARY_DIR}/rocprof-sys-pytest-output") # Default used by pytest
set(ROCPROFSYS_GENERATED_TESTS_FILE
    "${CMAKE_BINARY_DIR}/tests/pytest_tests_generated.cmake"
)

# Discover tests at build time
add_custom_command(
    OUTPUT "${ROCPROFSYS_GENERATED_TESTS_FILE}"
    COMMAND
        ${CMAKE_COMMAND} -D "PYTEST_COMMAND=${PYTEST_EXECUTABLE}" -D
        "PYTEST_BUILD_LOCATION=${ROCPROFSYS_PYTEST_BUILD_DIR}" -D
        "PYTEST_MARKER_EXCLUSIONS=${ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS}" -D
        "OUTPUT_FILE=${ROCPROFSYS_GENERATED_TESTS_FILE}" -D
        "CMAKE_BINARY_DIR=${CMAKE_BINARY_DIR}" -P
        "${CMAKE_CURRENT_LIST_DIR}/generate-ctests.cmake"
    DEPENDS "${CMAKE_CURRENT_LIST_DIR}/generate-ctests.cmake"
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    COMMENT "Discovering pytest tests at build time"
    VERBATIM
)

add_custom_target(discover-pytests ALL DEPENDS "${ROCPROFSYS_GENERATED_TESTS_FILE}")
# Make sure pytest files are copied first
add_dependencies(discover-pytests copy-pytest-files)

# Tell CTest to include the generated tests file
set_property(
    DIRECTORY
    APPEND
    PROPERTY TEST_INCLUDE_FILES "${ROCPROFSYS_GENERATED_TESTS_FILE}"
)
