# Vendor pytest-cmake module.
# Discover required rocprofiler_sdk_pytest target.
#
# This module defines the following imported targets:
#     rocprofiler_sdk_pytest::Pytest
#
# It also exposes the 'rocprofiler_sdk_pytest_discover_tests' function, which adds CTest
# test for each Pytest test. The "BUNDLE_PYTHON_TESTS" environment variable
# can be used to run all discovered tests together.
#
# Usage:
#     find_package(rocprofiler_sdk_pytest)
#     find_package(rocprofiler_sdk_pytest REQUIRED)
#
# Note:
#     The Pytest_ROOT environment variable or CMake variable can be used to
#     prepend a custom search path.
#     (https://cmake.org/cmake/help/latest/policy/CMP0074.html)

cmake_minimum_required(VERSION 3.20...4.1)

include(FindPackageHandleStandardArgs)

find_program(ROCPSDK_PYTEST_EXECUTABLE NAMES pytest)
mark_as_advanced(ROCPSDK_PYTEST_EXECUTABLE)

if(ROCPSDK_PYTEST_EXECUTABLE)
    execute_process(
        COMMAND "${ROCPSDK_PYTEST_EXECUTABLE}" --version
        OUTPUT_VARIABLE _version
        ERROR_VARIABLE _version
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(_version MATCHES "pytest (version )?([0-9]+\\.[0-9]+\\.[0-9]+)")
        set(PYTEST_VERSION "${CMAKE_MATCH_2}")
    endif()
endif()

find_package_handle_standard_args(
    rocprofiler_sdk_pytest
    REQUIRED_VARS ROCPSDK_PYTEST_EXECUTABLE
    VERSION_VAR PYTEST_VERSION
    HANDLE_COMPONENTS HANDLE_VERSION_RANGE)

if(rocprofiler_sdk_pytest_FOUND AND NOT TARGET rocprofiler_sdk_pytest::Pytest)
    add_executable(rocprofiler_sdk_pytest::Pytest IMPORTED)
    set_target_properties(rocprofiler_sdk_pytest::Pytest
                          PROPERTIES IMPORTED_LOCATION "${ROCPSDK_PYTEST_EXECUTABLE}")

    # Function to discover pytest tests and add them to CTest.
    function(rocprofiler_sdk_pytest_discover_tests NAME)
        set(_BOOL_ARGS STRIP_PARAM_BRACKETS INCLUDE_FILE_PATH BUNDLE_TESTS)

        set(_SINGLE_VALUE_ARGS WORKING_DIRECTORY TRIM_FROM_NAME TRIM_FROM_FULL_NAME)

        set(_MULTI_VALUE_ARGS
            TEST_PATHS LIBRARY_PATH_PREPEND PYTHON_PATH_PREPEND ENVIRONMENT PROPERTIES
            DEPENDS EXTRA_ARGS DISCOVERY_EXTRA_ARGS)

        cmake_parse_arguments(PARSE_ARGV 1 "" "${_BOOL_ARGS}" "${_SINGLE_VALUE_ARGS}"
                              "${_MULTI_VALUE_ARGS}")

        # Set platform-specific library path environment variable.
        if(CMAKE_SYSTEM_NAME STREQUAL Windows)
            set(LIBRARY_ENV_NAME PATH)
        elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
            set(LIBRARY_ENV_NAME DYLD_LIBRARY_PATH)
        else()
            set(LIBRARY_ENV_NAME LD_LIBRARY_PATH)
        endif()

        # Convert paths to CMake-friendly format.
        if(DEFINED ENV{${LIBRARY_ENV_NAME}})
            cmake_path(CONVERT "$ENV{${LIBRARY_ENV_NAME}}" TO_CMAKE_PATH_LIST
                       LIBRARY_PATH)
        else()
            set(LIBRARY_PATH "")
        endif()
        if(DEFINED ENV{PYTHONPATH})
            cmake_path(CONVERT "$ENV{PYTHONPATH}" TO_CMAKE_PATH_LIST PYTHON_PATH)
        else()
            set(PYTHON_PATH "")
        endif()

        # Prepend specified paths to the library and Python paths.
        if(_LIBRARY_PATH_PREPEND)
            list(REVERSE _LIBRARY_PATH_PREPEND)
            foreach(_path ${_LIBRARY_PATH_PREPEND})
                set(LIBRARY_PATH "${_path}" "${LIBRARY_PATH}")
            endforeach()
        endif()

        if(_PYTHON_PATH_PREPEND)
            list(REVERSE _PYTHON_PATH_PREPEND)
            foreach(_path ${_PYTHON_PATH_PREPEND})
                set(PYTHON_PATH "${_path}" "${PYTHON_PATH}")
            endforeach()
        endif()

        # Set default working directory if none is specified.
        if(NOT _WORKING_DIRECTORY)
            set(_WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        endif()

        get_filename_component(_WORKING_DIRECTORY "${_WORKING_DIRECTORY}" REALPATH)

        # Override option by environment variable if available.
        if(DEFINED ENV{BUNDLE_PYTHON_TESTS})
            set(_BUNDLE_TESTS $ENV{BUNDLE_PYTHON_TESTS})
        endif()

        # Define file paths for generated CMake include files.
        set(_include_file "${CMAKE_CURRENT_BINARY_DIR}/${NAME}_include.cmake")
        set(_tests_file "${CMAKE_CURRENT_BINARY_DIR}/${NAME}_tests.cmake")

        add_custom_command(
            VERBATIM
            OUTPUT "${_tests_file}"
            DEPENDS ${_DEPENDS}
            COMMAND
                ${CMAKE_COMMAND} -D "PYTEST_EXECUTABLE=${ROCPSDK_PYTEST_EXECUTABLE}" -D
                "TEST_PATHS=${_TEST_PATHS}" -D "TEST_GROUP_NAME=${NAME}" -D
                "BUNDLE_TESTS=${_BUNDLE_TESTS}" -D "LIBRARY_ENV_NAME=${LIBRARY_ENV_NAME}"
                -D "LIBRARY_PATH=${LIBRARY_PATH}" -D "PYTHON_PATH=${PYTHON_PATH}" -D
                "TRIM_FROM_NAME=${_TRIM_FROM_NAME}" -D
                "TRIM_FROM_FULL_NAME=${_TRIM_FROM_FULL_NAME}" -D
                "STRIP_PARAM_BRACKETS=${_STRIP_PARAM_BRACKETS}" -D
                "INCLUDE_FILE_PATH=${_INCLUDE_FILE_PATH}" -D
                "WORKING_DIRECTORY=${_WORKING_DIRECTORY}" -D
                "ENVIRONMENT=${_ENVIRONMENT}" -D "TEST_PROPERTIES=${_PROPERTIES}" -D
                "CTEST_FILE=${_tests_file}" -D "EXTRA_ARGS=${_EXTRA_ARGS}" -D
                "DISCOVERY_EXTRA_ARGS=${_DISCOVERY_EXTRA_ARGS}" -P
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/rocprofiler_sdk_PytestAddTests.cmake")

        # Create a custom target to run the tests.
        add_custom_target(${NAME} ALL DEPENDS ${_tests_file})

        file(WRITE "${_include_file}"
             "if(EXISTS \"${_tests_file}\")\n" "    include(\"${_tests_file}\")\n"
             "else()\n" "    add_test(${NAME}_NOT_BUILT ${NAME}_NOT_BUILT)\n" "endif()\n")

        # Register the include file to be processed for tests.
        set_property(
            DIRECTORY
            APPEND
            PROPERTY TEST_INCLUDE_FILES "${_include_file}")

    endfunction()

endif()
