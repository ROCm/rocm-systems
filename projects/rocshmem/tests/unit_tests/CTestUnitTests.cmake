###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# CTest-based unit test definitions
# This replaces the shell-based driver.sh with native CTest integration

# Find MPI runtime
find_program(MPIRUN_EXECUTABLE NAMES mpirun mpiexec)
if(NOT MPIRUN_EXECUTABLE)
    message(WARNING "mpirun not found - unit tests will not be added")
    return()
endif()

# Helper function to add a unit test with MPI
# Usage:
#   add_rocshmem_unit_test(
#     NAME <test_name>
#     RANKS <num_ranks>
#     GTEST_FILTER <filter_pattern>
#     [TIMEOUT <seconds>]
#     [LABELS <label1> <label2> ...]
#   )
function(add_rocshmem_unit_test)
    set(options "")
    set(oneValueArgs NAME RANKS GTEST_FILTER TIMEOUT)
    set(multiValueArgs LABELS)
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT DEFINED TEST_NAME OR NOT DEFINED TEST_RANKS OR NOT DEFINED TEST_GTEST_FILTER)
        message(FATAL_ERROR "add_rocshmem_unit_test: NAME, RANKS, and GTEST_FILTER are required")
    endif()

    # Build test name
    set(FULL_TEST_NAME "${TEST_NAME}_n${TEST_RANKS}")

    # Default timeout (20 minutes, matching driver.sh)
    if(NOT DEFINED TEST_TIMEOUT)
        set(TEST_TIMEOUT 1200)  # 20 minutes
    endif()

    # Build test command with wrapper (handles GPU count checking and skip logic)
    set(TEST_COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_wrapper.sh
        ${FULL_TEST_NAME}
        ${TEST_RANKS}
        ${MPIRUN_EXECUTABLE}
        -np ${TEST_RANKS}
        --timeout ${TEST_TIMEOUT}
        $<TARGET_FILE:rocshmem_unit_tests>
        --gtest_filter=${TEST_GTEST_FILTER}
    )

    # Add the test
    add_test(
        NAME ${FULL_TEST_NAME}
        COMMAND ${TEST_COMMAND}
    )

    # Build labels list
    set(ALL_LABELS "unit")
    if(DEFINED TEST_LABELS)
        list(APPEND ALL_LABELS ${TEST_LABELS})
    endif()

    # Set test properties
    set_tests_properties(${FULL_TEST_NAME} PROPERTIES
        TIMEOUT ${TEST_TIMEOUT}
        LABELS "${ALL_LABELS}"
        PROCESSORS ${TEST_RANKS}
        SKIP_RETURN_CODE 125  # CTest skip code for insufficient GPUs
    )
endfunction()

###############################################################################
# Test Definitions (matching driver.sh test suites)
###############################################################################

function(register_all_unit_tests)
    # Check for gfx1201 - unit tests disabled (AIROCSHMEM-393)
    execute_process(
        COMMAND rocminfo
        OUTPUT_VARIABLE ROCMINFO_OUTPUT
        ERROR_QUIET
    )
    if(ROCMINFO_OUTPUT MATCHES "gfx1201")
        message(STATUS "Unit tests disabled for gfx1201 (AIROCSHMEM-393)")
        return()
    endif()

    # Define test filters from driver.sh
    set(TEST_WITH_TWO_PES "IPCImplSimpleCoarseTestFixture/*:IPCImplSimpleFineTestFixture/*:IPCImplTiledFineTestFixture/*:DegenerateTiledFine.*")
    set(TEST_WITH_TWO_PES "${TEST_WITH_TWO_PES}:SdmaSimpleCoarse/*:SdmaSimpleFine/*:SdmaTiledFine/*")

    # All unit tests (4 ranks) - runs everything EXCEPT the 2-PE specific tests
    # NOTE: Tier labels (quick/standard/comprehensive/full) are now applied
    # via YAML-based categorization in test_categories.yaml
    add_rocshmem_unit_test(
        NAME unit_tests
        RANKS 4
        GTEST_FILTER "-${TEST_WITH_TWO_PES}"
        LABELS "ALL;IPC;SDMA"
    )

    # Note: 2-rank tests are commented out in driver.sh
    # If needed, uncomment this:
    # add_rocshmem_unit_test(
    #     NAME two_pe_tests
    #     RANKS 2
    #     GTEST_FILTER "${TEST_WITH_TWO_PES}"
    #     LABELS "TWO_PE;IPC;SDMA"
    # )
endfunction()
