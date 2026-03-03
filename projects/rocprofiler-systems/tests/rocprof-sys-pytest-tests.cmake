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

# Helper macro to add to marker exclusion
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

# Build marker exclusion string from ROCPROFSYS_DISABLE_EXAMPLES
set(ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS "")
set(_excluded_markers "")
if(ROCPROFSYS_DISABLE_EXAMPLES)
    foreach(_marker ${ROCPROFSYS_DISABLE_EXAMPLES})
        rocprofiler_systems_add_pytest_marker_exclusion(${_marker})
    endforeach()
endif()

# Exclude GPU tests if no GPU is available
set(GPU_TARGETS "")
rocprofiler_systems_get_gfx_archs(GPU_TARGETS)
if(NOT ROCPROFSYS_CI_GPU OR GPU_TARGETS STREQUAL "")
    rocprofiler_systems_add_pytest_marker_exclusion("gpu")
endif()

rocprofiler_systems_message(STATUS
    "PyTest tests with the following markers will be excluded from the CTest suite: ${ROCPROFSYS_PYTEST_MARKER_EXCLUSIONS_LIST}"
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
# Pytest collection needs the built executables to discover the build config,
# and the copied pytest files.
set(PYTEST_DEPENDENCIES
    copy-pytest-files
    rocprofiler-systems-instrument
    rocprofiler-systems-sample
    rocprofiler-systems-run
    rocprofiler-systems-causal
    rocprofiler-systems-avail
)
add_dependencies(discover-pytests ${PYTEST_DEPENDENCIES})

# Tell CTest to include the generated tests file
set_property(
    DIRECTORY
    APPEND
    PROPERTY TEST_INCLUDE_FILES "${ROCPROFSYS_GENERATED_TESTS_FILE}"
)

# Display pytest configuration before any tests run.
# This way output is not hidden when the test passes.
file(
    WRITE
    "${CMAKE_BINARY_DIR}/CTestCustom.cmake"
    "set(CTEST_CUSTOM_PRE_TEST \"${PYTEST_EXECUTABLE} ${ROCPROFSYS_PYTEST_BUILD_DIR} --show-config-only\")\n"
)
