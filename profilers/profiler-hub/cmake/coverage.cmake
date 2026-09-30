# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# Code Coverage Support
#
# This module provides code coverage functionality using gcov/lcov or gcovr.
#
# Usage:
#   cmake -DPROFILER_HUB_ENABLE_COVERAGE=ON -DCMAKE_BUILD_TYPE=Debug ..
#   make
#   make coverage          # Run tests and generate HTML report
#   make coverage-xml      # Run tests and generate Cobertura XML (for CI)
#   make coverage-clean    # Clean coverage data
#
# ----------------------------------------------------------------------------------------#

include_guard(DIRECTORY)

if(NOT PROFILER_HUB_ENABLE_COVERAGE)
    return()
endif()

# Coverage requires Debug build for accurate line information
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(
        WARNING
        "[profiler-hub] Code coverage works best with CMAKE_BUILD_TYPE=Debug. "
        "Current build type: ${CMAKE_BUILD_TYPE}"
    )
endif()

# Only GCC and Clang support gcov-compatible coverage
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(
        FATAL_ERROR
        "[profiler-hub] Code coverage requires GCC or Clang. "
        "Current compiler: ${CMAKE_CXX_COMPILER_ID}"
    )
endif()

message(STATUS "[profiler-hub] Code coverage enabled")

# ----------------------------------------------------------------------------------------#
# Find coverage tools
# ----------------------------------------------------------------------------------------#

find_program(GCOV_PATH gcov)
find_program(LCOV_PATH lcov)
find_program(GENHTML_PATH genhtml)
find_program(GCOVR_PATH gcovr)

if(NOT GCOV_PATH)
    message(FATAL_ERROR "[profiler-hub] gcov not found, required for coverage")
endif()

# Prefer lcov/genhtml, fall back to gcovr
set(PROFILER_HUB_COVERAGE_USE_LCOV FALSE)
set(PROFILER_HUB_COVERAGE_USE_GCOVR FALSE)

if(LCOV_PATH AND GENHTML_PATH)
    set(PROFILER_HUB_COVERAGE_USE_LCOV TRUE)
    message(
        STATUS
        "[profiler-hub] Using lcov for coverage reports: ${LCOV_PATH}"
    )
elseif(GCOVR_PATH)
    set(PROFILER_HUB_COVERAGE_USE_GCOVR TRUE)
    message(
        STATUS
        "[profiler-hub] Using gcovr for coverage reports: ${GCOVR_PATH}"
    )
else()
    message(
        FATAL_ERROR
        "[profiler-hub] Coverage report generation requires either:\n"
        "  - lcov and genhtml (apt install lcov)\n"
        "  - gcovr (pip install gcovr)"
    )
endif()

# ----------------------------------------------------------------------------------------#
# Coverage compiler and linker flags
# ----------------------------------------------------------------------------------------#

set(PROFILER_HUB_COVERAGE_COMPILE_FLAGS -fprofile-arcs -ftest-coverage)
set(PROFILER_HUB_COVERAGE_LINK_FLAGS --coverage)

# For Clang, we may need to specify the gcov tool
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    list(
        APPEND PROFILER_HUB_COVERAGE_COMPILE_FLAGS
        -fprofile-instr-generate
        -fcoverage-mapping
    )
    set(PROFILER_HUB_COVERAGE_LINK_FLAGS -fprofile-instr-generate)
endif()

# ----------------------------------------------------------------------------------------#
# Function to add coverage flags to a target
# ----------------------------------------------------------------------------------------#

function(profiler_hub_add_coverage_flags target)
    target_compile_options(
        ${target}
        PRIVATE ${PROFILER_HUB_COVERAGE_COMPILE_FLAGS}
    )
    target_link_options(${target} PRIVATE ${PROFILER_HUB_COVERAGE_LINK_FLAGS})
endfunction()

# ----------------------------------------------------------------------------------------#
# Coverage output directory
# ----------------------------------------------------------------------------------------#

set(PROFILER_HUB_COVERAGE_DIR ${CMAKE_BINARY_DIR}/coverage)
file(MAKE_DIRECTORY ${PROFILER_HUB_COVERAGE_DIR})

# ----------------------------------------------------------------------------------------#
# Coverage targets using lcov
# ----------------------------------------------------------------------------------------#

if(PROFILER_HUB_COVERAGE_USE_LCOV)
    # Clean coverage data
    add_custom_target(
        coverage-clean
        COMMAND ${LCOV_PATH} --directory ${CMAKE_BINARY_DIR} --zerocounters
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${PROFILER_HUB_COVERAGE_DIR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Cleaning coverage data..."
    )

    # Generate HTML coverage report
    add_custom_target(
        coverage
        # Reset counters
        COMMAND ${LCOV_PATH} --directory ${CMAKE_BINARY_DIR} --zerocounters
        # Run tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        # Capture coverage data
        COMMAND
            ${LCOV_PATH} --directory ${CMAKE_BINARY_DIR} --capture --output-file
            ${PROFILER_HUB_COVERAGE_DIR}/coverage.info --ignore-errors mismatch
        # Remove external dependencies from coverage
        COMMAND
            ${LCOV_PATH} --remove ${PROFILER_HUB_COVERAGE_DIR}/coverage.info
            '/usr/*' '${CMAKE_BINARY_DIR}/_deps/*' '${CMAKE_SOURCE_DIR}/tests/*'
            --output-file ${PROFILER_HUB_COVERAGE_DIR}/coverage.info
            --ignore-errors unused
        # Generate HTML report
        COMMAND
            ${GENHTML_PATH} ${PROFILER_HUB_COVERAGE_DIR}/coverage.info
            --output-directory ${PROFILER_HUB_COVERAGE_DIR}/html --title
            "profiler-hub Code Coverage" --legend --show-details
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating code coverage report..."
    )

    add_custom_command(
        TARGET coverage
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E echo
            "Coverage report: file://${PROFILER_HUB_COVERAGE_DIR}/html/index.html"
    )
endif()

# ----------------------------------------------------------------------------------------#
# Coverage targets using gcovr
# ----------------------------------------------------------------------------------------#

if(PROFILER_HUB_COVERAGE_USE_GCOVR)
    # Clean coverage data
    add_custom_target(
        coverage-clean
        COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${PROFILER_HUB_COVERAGE_DIR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Cleaning coverage data..."
    )

    # Generate HTML coverage report
    add_custom_target(
        coverage
        # Clean old data
        COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
        # Run tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        # Generate HTML report
        COMMAND
            ${GCOVR_PATH} --root ${CMAKE_SOURCE_DIR} --filter
            ${CMAKE_SOURCE_DIR}/source --filter ${CMAKE_SOURCE_DIR}/include
            --exclude ${CMAKE_SOURCE_DIR}/tests --html-details
            ${PROFILER_HUB_COVERAGE_DIR}/html/index.html --print-summary
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating code coverage report..."
    )

    add_custom_command(
        TARGET coverage
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E echo
            "Coverage report: file://${PROFILER_HUB_COVERAGE_DIR}/html/index.html"
    )

    # Generate Cobertura XML report (for CI integration)
    add_custom_target(
        coverage-xml
        # Clean old data
        COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
        # Run tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        # Generate XML report
        COMMAND
            ${GCOVR_PATH} --root ${CMAKE_SOURCE_DIR} --filter
            ${CMAKE_SOURCE_DIR}/source --filter ${CMAKE_SOURCE_DIR}/include
            --exclude ${CMAKE_SOURCE_DIR}/tests --xml
            ${PROFILER_HUB_COVERAGE_DIR}/coverage.xml --print-summary
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating Cobertura XML coverage report..."
    )

    add_custom_command(
        TARGET coverage-xml
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E echo
            "Coverage XML: ${PROFILER_HUB_COVERAGE_DIR}/coverage.xml"
    )
endif()

# XML target for lcov (using lcov's cobertura output if available)
if(PROFILER_HUB_COVERAGE_USE_LCOV AND GCOVR_PATH)
    add_custom_target(
        coverage-xml
        # Reset counters
        COMMAND ${LCOV_PATH} --directory ${CMAKE_BINARY_DIR} --zerocounters
        # Run tests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        # Generate XML using gcovr
        COMMAND
            ${GCOVR_PATH} --root ${CMAKE_SOURCE_DIR} --filter
            ${CMAKE_SOURCE_DIR}/source --filter ${CMAKE_SOURCE_DIR}/include
            --exclude ${CMAKE_SOURCE_DIR}/tests --xml
            ${PROFILER_HUB_COVERAGE_DIR}/coverage.xml --print-summary
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating Cobertura XML coverage report..."
    )
endif()
