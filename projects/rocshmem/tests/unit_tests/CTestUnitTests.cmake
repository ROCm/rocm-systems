###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# CTest-based unit test definitions
# This replaces the shell-based driver.sh with native CTest integration

# MPI should already be found by the parent CMakeLists.txt
# Use standard CMake MPI variables set by find_package(MPI)
if(NOT MPIEXEC_EXECUTABLE)
    message(WARNING "MPIEXEC_EXECUTABLE not found - unit tests will not be added")
    return()
endif()

# MCA parameters can be overridden via environment variables at CMake configure time
# or via CMake cache variables. Defaults to ucx for both.
set(OMPI_MCA_PML "$ENV{OMPI_MCA_pml}" CACHE STRING "OpenMPI MCA pml parameter")
set(OMPI_MCA_OSC "$ENV{OMPI_MCA_osc}" CACHE STRING "OpenMPI MCA osc parameter")

# Use ucx as default if not specified
if(NOT OMPI_MCA_PML)
    set(OMPI_MCA_PML "ucx")
endif()
if(NOT OMPI_MCA_OSC)
    set(OMPI_MCA_OSC "ucx")
endif()

message(STATUS "MPI executable: ${MPIEXEC_EXECUTABLE}")
message(STATUS "MPI numproc flag: ${MPIEXEC_NUMPROC_FLAG}")
message(STATUS "MPI MCA parameters: pml=${OMPI_MCA_PML}, osc=${OMPI_MCA_OSC}")

###############################################################################
# Install-Time CTest Generation Support
###############################################################################

# Global variable to store install-time test definitions
set(INSTALL_CTEST_FILE "" CACHE INTERNAL "Install-time CTest file path")

# Function to initialize install-time test generation
function(init_install_ctest_generation output_file)
    set(INSTALL_CTEST_FILE "${output_file}" CACHE INTERNAL "Install-time CTest file path")
    file(WRITE "${INSTALL_CTEST_FILE}" "")
endfunction()

# Helper function to write install-time unit test definition
function(write_install_unit_test_definition TEST_NAME TEST_COMMAND TEST_LABELS TEST_TIMEOUT TEST_RANKS)
    if(NOT INSTALL_CTEST_FILE)
        return()
    endif()

    # Build install-time command by iterating through the list
    # Working directory for tests is: <install>/bin/rocshmem/tests/unit
    set(INSTALL_CMD_PARTS "")
    foreach(part IN LISTS TEST_COMMAND)
        # Replace build-time paths with install-time relative paths
        if("${part}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/unit_test_wrapper.sh")
            # Wrapper is at: <install>/share/rocshmem/unit_test_wrapper.sh
            # Relative from working dir: ../../../../share/rocshmem/unit_test_wrapper.sh
            list(APPEND INSTALL_CMD_PARTS "../../../../share/rocshmem/unit_test_wrapper.sh")
        elseif("${part}" STREQUAL "$<TARGET_FILE:rocshmem_unit_tests>")
            # Executable is at: <install>/share/rocshmem/rocshmem_unit_tests
            # Relative from working dir: ../../../../share/rocshmem/rocshmem_unit_tests
            list(APPEND INSTALL_CMD_PARTS "../../../../share/rocshmem/rocshmem_unit_tests")
        elseif("${part}" STREQUAL "${MPIEXEC_EXECUTABLE}")
            # Use the MPI executable found at configure time (absolute path)
            list(APPEND INSTALL_CMD_PARTS "${MPIEXEC_EXECUTABLE}")
        elseif("${part}" STREQUAL "${MPIEXEC_NUMPROC_FLAG}")
            list(APPEND INSTALL_CMD_PARTS "${MPIEXEC_NUMPROC_FLAG}")
        elseif("${part}" STREQUAL "${OMPI_MCA_PML}")
            list(APPEND INSTALL_CMD_PARTS "${OMPI_MCA_PML}")
        elseif("${part}" STREQUAL "${OMPI_MCA_OSC}")
            list(APPEND INSTALL_CMD_PARTS "${OMPI_MCA_OSC}")
        else()
            list(APPEND INSTALL_CMD_PARTS "${part}")
        endif()
    endforeach()

    # Write test definition using legacy syntax (no NAME/COMMAND keywords)
    # This matches the format CMake generates in build-time CTestTestfile.cmake files
    file(APPEND "${INSTALL_CTEST_FILE}" "\nadd_test(${TEST_NAME}")
    foreach(arg IN LISTS INSTALL_CMD_PARTS)
        # Escape quotes and write each argument
        string(REPLACE "\"" "\\\"" escaped_arg "${arg}")
        file(APPEND "${INSTALL_CTEST_FILE}" " \"${escaped_arg}\"")
    endforeach()
    file(APPEND "${INSTALL_CTEST_FILE}" ")\n")
    file(APPEND "${INSTALL_CTEST_FILE}" "set_tests_properties(${TEST_NAME} PROPERTIES\n")
    file(APPEND "${INSTALL_CTEST_FILE}" "    TIMEOUT ${TEST_TIMEOUT}\n")
    file(APPEND "${INSTALL_CTEST_FILE}" "    SKIP_RETURN_CODE 125\n")
    file(APPEND "${INSTALL_CTEST_FILE}" "    LABELS \"${TEST_LABELS}\"\n")
    file(APPEND "${INSTALL_CTEST_FILE}" "    PROCESSORS ${TEST_RANKS})\n")
endfunction()

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
    # Use standard CMake MPI variables
    set(TEST_COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_wrapper.sh
        ${FULL_TEST_NAME}
        ${TEST_RANKS}
        ${MPIEXEC_EXECUTABLE}
        ${MPIEXEC_NUMPROC_FLAG} ${TEST_RANKS}
        ${MPIEXEC_PREFLAGS}
        -mca pml ${OMPI_MCA_PML}
        -mca osc ${OMPI_MCA_OSC}
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

    # Also write install-time test definition if generating install tests
    write_install_unit_test_definition("${FULL_TEST_NAME}" "${TEST_COMMAND}" "${ALL_LABELS}" "${TEST_TIMEOUT}" "${TEST_RANKS}")
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
