###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# CTest-based functional test definitions with multi-dimensional label system
# This replaces the shell-based driver.sh with native CTest integration
#
# Multi-Dimensional Label System:
# 1. Category: RMA, PUT, GET, AMO, COLLECTIVE, STREAM, etc.
# 2. CI Tiers: quick, standard, comprehensive, full
# 3. Backend Compatibility: backend_all, backend_ipc, backend_gda, backend_ro
# 4. GPU Architecture: gpu_all, gpu_gfx942, gpu_gfx90a, etc.
# 5. Variants: variant_base, variant_uuid, variant_default_stream

# Test number mappings (must match TestType enum in tester.hpp)
set(TEST_get 0)
set(TEST_getnbi 1)
set(TEST_put 2)
set(TEST_putnbi 3)
set(TEST_amo_fadd 4)
set(TEST_amo_finc 5)
set(TEST_amo_fetch 6)
set(TEST_amo_fcswap 7)
set(TEST_amo_add 8)
set(TEST_amo_inc 9)
set(TEST_amo_cswap 10)
set(TEST_init 11)
set(TEST_pingpong 12)
set(TEST_randomaccess 13)
set(TEST_barrierall 14)
set(TEST_syncall 15)
set(TEST_teamsync 16)
set(TEST_collect 17)
set(TEST_fcollect 18)
set(TEST_alltoall 19)
set(TEST_alltoallv 20)
set(TEST_shmemptr 21)
set(TEST_p 22)
set(TEST_g 23)
set(TEST_wgget 24)
set(TEST_wggetnbi 25)
set(TEST_wgput 26)
set(TEST_wgputnbi 27)
set(TEST_waveget 28)
set(TEST_wavegetnbi 29)
set(TEST_waveput 30)
set(TEST_waveputnbi 31)
set(TEST_teambroadcast 32)
set(TEST_teamreduction 33)
set(TEST_teamctxget 34)
set(TEST_teamctxgetnbi 35)
set(TEST_teamctxput 36)
set(TEST_teamctxputnbi 37)
set(TEST_teamctxinfra 38)
set(TEST_putnbimr 39)
set(TEST_amo_set 40)
set(TEST_amo_swap 41)
set(TEST_amo_fetchand 42)
set(TEST_amo_fetchor 43)
set(TEST_amo_fetchxor 44)
set(TEST_amo_and 45)
set(TEST_amo_or 46)
set(TEST_amo_xor 47)
set(TEST_pingall 48)
set(TEST_putsignal 49)
set(TEST_wgputsignal 50)
set(TEST_waveputsignal 51)
set(TEST_putsignalnbi 52)
set(TEST_wgputsignalnbi 53)
set(TEST_waveputsignalnbi 54)
set(TEST_signalfetch 55)
set(TEST_wgsignalfetch 56)
set(TEST_wavesignalfetch 57)
set(TEST_teamwgbarrier 58)
set(TEST_defaultctxget 59)
set(TEST_defaultctxgetnbi 60)
set(TEST_defaultctxput 61)
set(TEST_defaultctxputnbi 62)
set(TEST_defaultctxp 63)
set(TEST_defaultctxg 64)
set(TEST_wavebarrierall 65)
set(TEST_wgbarrierall 66)
set(TEST_wavesyncall 67)
set(TEST_wgsyncall 68)
set(TEST_teambarrier 69)
set(TEST_teamwavebarrier 70)
set(TEST_teamwavesync 71)
set(TEST_teamwgsync 72)
set(TEST_teamctxsingleinfra 73)
set(TEST_teamctxblockinfra 74)
set(TEST_teamctxoddeveninfra 75)
set(TEST_alltoallmem_on_stream 76)
set(TEST_barrier_all_on_stream 77)
set(TEST_broadcastmem_on_stream 78)
set(TEST_getmem_on_stream 79)
set(TEST_putmem_on_stream 80)
set(TEST_putmem_signal_on_stream 81)
set(TEST_signal_wait_until_on_stream 82)
set(TEST_flood_put 83)
set(TEST_flood_putnbi 84)
set(TEST_flood_p 85)
set(TEST_flood_get 86)
set(TEST_flood_getnbi 87)
set(TEST_flood_g 88)
set(TEST_hipmodule_init 89)
set(TEST_flood_add 90)
set(TEST_flood_fadd 91)
set(TEST_flood_waitadd 92)
set(TEST_device_bitcode 93)
set(TEST_library_info 94)
set(TEST_teamctxsharedinfra 95)
set(TEST_quiet_on_stream 96)
set(TEST_sync_all_on_stream 97)
set(TEST_teamctxsubsetparentinfra 98)
set(TEST_fence_putwavesignal 99)
set(TEST_fence_putlargesmall 100)
set(TEST_fence_fanout 101)
set(TEST_fence_putwavenbichunks 102)
set(TEST_tile_put_contiguous 103)
set(TEST_tile_put_rowmajor 104)
set(TEST_tile_put_colmajor 105)
set(TEST_tile_put_arbitrary 106)
set(TEST_tile_put_wave_contiguous 107)
set(TEST_tile_put_wg_contiguous 108)
set(TEST_tile_get_contiguous 109)
set(TEST_tile_get_wg_contiguous 110)
set(TEST_tile_put_1d 111)
set(TEST_tile_get_1d 112)
set(TEST_tile_get_wave_contiguous 113)
set(TEST_tile_get_rowmajor 114)
set(TEST_tile_get_colmajor 115)
set(TEST_tile_get_arbitrary 116)
set(TEST_reduce_on_stream 117)
set(TEST_host_ctx_create 118)
set(TEST_teamsplit2d 119)
set(TEST_hostteamsyncbarrier 120)
set(TEST_host_putmem 121)
set(TEST_host_getmem 122)
set(TEST_host_amo_fadd 123)
set(TEST_host_amo_fcswap 124)
set(TEST_host_ctx_putmem 125)
set(TEST_host_ctx_getmem 126)
set(TEST_host_int_amo_fadd 127)
set(TEST_host_int_amo_fcswap 128)
set(TEST_host_amo_all_pes 129)
set(TEST_host_amo_self 130)
set(TEST_host_amo_add 131)
set(TEST_tile_broadcast 132)
set(TEST_tile_broadcast_wave 133)
set(TEST_tile_broadcast_wg 134)
set(TEST_tile_allgather 135)
set(TEST_tile_allgather_wave 136)
set(TEST_tile_allgather_wg 137)
set(TEST_host_wait_until 138)
set(TEST_host_test 139)
set(TEST_host_wait_until_all 140)
set(TEST_host_wait_until_any 141)
set(TEST_host_wait_until_some 142)
set(TEST_host_wait_until_all_vector 143)
set(TEST_host_wait_until_any_vector 144)
set(TEST_host_wait_until_some_vector 145)
set(TEST_host_wait_until_all_status 146)
set(TEST_host_wait_until_any_status 147)
set(TEST_host_wait_until_some_status 148)

# MPI should already be found by the parent CMakeLists.txt
# Use standard CMake MPI variables set by find_package(MPI)
# If MPI is not found, automatically fall back to SLR (Simple Local Runtime)
if(NOT MPIEXEC_EXECUTABLE)
    message(STATUS "MPI not found - using SLR (Simple Local Runtime) for functional tests")
    set(USE_SLR_LAUNCHER ON CACHE BOOL "Use SLR launcher instead of MPI" FORCE)
else()
    message(STATUS "MPI found - using MPI launcher for functional tests")
    set(USE_SLR_LAUNCHER OFF CACHE BOOL "Use SLR launcher instead of MPI" FORCE)

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
endif()

###############################################################################
# Install-Time CTest Generation Support
###############################################################################

# Global variable to store install-time test definitions
set(INSTALL_CTEST_FILE "" CACHE INTERNAL "Install-time CTest file path")

# Function to initialize install-time test generation
function(init_install_ctest_generation output_file)
    set(INSTALL_CTEST_FILE "${output_file}" CACHE INTERNAL "Install-time CTest file path")
    # Clear the file
    file(WRITE "${INSTALL_CTEST_FILE}" "")
endfunction()

# Helper function to write install-time test definition
function(write_install_test_definition TEST_NAME TEST_COMMAND TEST_LABELS TEST_TIMEOUT TEST_RANKS)
    if(NOT INSTALL_CTEST_FILE)
        return()  # Not generating install tests
    endif()

    # Build install-time command by iterating through the list
    # and replacing build-time paths with install-time relative paths
    # Working directory for tests is: <install>/bin/rocshmem/tests/functional
    set(INSTALL_CMD_PARTS "")
    foreach(part IN LISTS TEST_COMMAND)
        # Replace build-time paths with install-time relative paths
        if("${part}" STREQUAL "${CMAKE_CURRENT_SOURCE_DIR}/test_wrapper.sh")
            # Wrapper is at: <install>/share/rocshmem/test_wrapper.sh
            # Relative from working dir: ../../../../share/rocshmem/test_wrapper.sh
            list(APPEND INSTALL_CMD_PARTS "../../../../share/rocshmem/test_wrapper.sh")
        elseif("${part}" STREQUAL "$<TARGET_FILE:rocshmem_functional_tests>")
            # Executable is at: <install>/share/rocshmem/rocshmem_functional_tests
            # Relative from working dir: ../../../../share/rocshmem/rocshmem_functional_tests
            list(APPEND INSTALL_CMD_PARTS "../../../../share/rocshmem/rocshmem_functional_tests")
        elseif("${part}" STREQUAL "${CMAKE_COMMAND}")
            # Replace build-host cmake path with portable system env
            # This prevents build-host absolute paths from leaking into install files
            list(APPEND INSTALL_CMD_PARTS "env")
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
            # Keep the part as-is
            list(APPEND INSTALL_CMD_PARTS "${part}")
        endif()
    endforeach()

    # Write add_test() command using legacy syntax (no NAME/COMMAND keywords)
    # This matches the format CMake generates in build-time CTestTestfile.cmake files
    file(APPEND "${INSTALL_CTEST_FILE}" "\nadd_test(${TEST_NAME}")
    foreach(arg IN LISTS INSTALL_CMD_PARTS)
        # Escape quotes and write each argument
        string(REPLACE "\"" "\\\"" escaped_arg "${arg}")
        file(APPEND "${INSTALL_CTEST_FILE}" " \"${escaped_arg}\"")
    endforeach()
    file(APPEND "${INSTALL_CTEST_FILE}" ")\n")

    # Write test properties
    if(TEST_TIMEOUT GREATER 0)
        file(APPEND "${INSTALL_CTEST_FILE}" "set_tests_properties(${TEST_NAME} PROPERTIES\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    TIMEOUT ${TEST_TIMEOUT}\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    SKIP_RETURN_CODE 125\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    LABELS \"${TEST_LABELS}\"\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    PROCESSORS ${TEST_RANKS}\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    WILL_FAIL FALSE\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    FAIL_REGULAR_EXPRESSION \"FAILED\")\n")
    else()
        # Heatmap tests - no timeout
        file(APPEND "${INSTALL_CTEST_FILE}" "set_tests_properties(${TEST_NAME} PROPERTIES\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    SKIP_RETURN_CODE 125\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    LABELS \"${TEST_LABELS}\"\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    PROCESSORS ${TEST_RANKS}\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    WILL_FAIL FALSE\n")
        file(APPEND "${INSTALL_CTEST_FILE}" "    FAIL_REGULAR_EXPRESSION \"FAILED\")\n")
    endif()
endfunction()

###############################################################################
# Variant Definitions
###############################################################################

# Global variants: Apply to ALL tests (via variant multiplication)
# Format: variant_name|env_var|label
set(GLOBAL_VARIANT_uuid "uuid|ROCSHMEM_TEST_UUID=1|variant_uuid")

# Test-specific variants: Apply only when explicitly requested
# Format: variant_name|env_var|label
set(TEST_VARIANT_default_stream "default_stream|ROCSHMEM_TEST_USE_DEFAULT_STREAM=1|variant_default_stream;STREAM")

# List of all global variant names
set(ALL_GLOBAL_VARIANT_NAMES uuid)

# List of all test-specific variant names
set(ALL_TEST_VARIANT_NAMES default_stream)

###############################################################################
# Helper Functions for Multi-Dimensional Labels
###############################################################################

# Generate all combinations of global and test variants
function(generate_variant_combinations global_variants test_variants output_var)
    set(ALL_COMBINATIONS "")

    # When using SLR launcher, skip base variant (no difference from uuid variant in SLR mode)
    # Only include base variant for MPI mode
    if(NOT USE_SLR_LAUNCHER)
        list(APPEND ALL_COMBINATIONS "base")
    endif()

    # Add global-only variants
    foreach(global_var ${global_variants})
        list(APPEND ALL_COMBINATIONS "${global_var}")
    endforeach()

    # Add test-only variants
    foreach(test_var ${test_variants})
        list(APPEND ALL_COMBINATIONS "${test_var}")
    endforeach()

    # Add combinations of global + test variants
    foreach(global_var ${global_variants})
        foreach(test_var ${test_variants})
            list(APPEND ALL_COMBINATIONS "${global_var}+${test_var}")
        endforeach()
    endforeach()

    set(${output_var} "${ALL_COMBINATIONS}" PARENT_SCOPE)
endfunction()

# Parse a variant combination string and get env vars + labels + suffix
function(parse_variant_combination combo global_variants test_variants env_vars_var labels_var suffix_var)
    set(ENV_VARS "")
    set(LABELS "variant_base")
    set(SUFFIX "")

    if(combo STREQUAL "base")
        set(${env_vars_var} "" PARENT_SCOPE)
        set(${labels_var} "variant_base" PARENT_SCOPE)
        set(${suffix_var} "" PARENT_SCOPE)
        return()
    endif()

    # Split combination by '+'
    string(REPLACE "+" ";" combo_parts "${combo}")

    set(LABELS "")
    set(suffix_parts "")

    foreach(variant_part ${combo_parts})
        # Check if it's a global variant
        list(FIND global_variants "${variant_part}" global_idx)
        if(NOT global_idx EQUAL -1)
            string(REPLACE "|" ";" variant_config "${GLOBAL_VARIANT_${variant_part}}")
            list(GET variant_config 1 env_var)
            list(GET variant_config 2 label)

            if(env_var)
                list(APPEND ENV_VARS "${env_var}")
            endif()
            if(label)
                string(REPLACE ";" "," label_temp "${label}")
                string(REPLACE "," ";" label_list "${label_temp}")
                list(APPEND LABELS ${label_list})
            endif()
            list(APPEND suffix_parts "${variant_part}")
        else()
            # Check if it's a test variant
            list(FIND test_variants "${variant_part}" test_idx)
            if(NOT test_idx EQUAL -1)
                string(REPLACE "|" ";" variant_config "${TEST_VARIANT_${variant_part}}")
                list(GET variant_config 1 env_var)
                list(GET variant_config 2 label)

                if(env_var)
                    list(APPEND ENV_VARS "${env_var}")
                endif()
                if(label)
                    string(REPLACE ";" "," label_temp "${label}")
                    string(REPLACE "," ";" label_list "${label_temp}")
                    list(APPEND LABELS ${label_list})
                endif()
                list(APPEND suffix_parts "${variant_part}")
            endif()
        endif()
    endforeach()

    # Build suffix from parts
    string(REPLACE ";" "_" SUFFIX "${suffix_parts}")

    set(${env_vars_var} "${ENV_VARS}" PARENT_SCOPE)
    set(${labels_var} "${LABELS}" PARENT_SCOPE)
    set(${suffix_var} "${SUFFIX}" PARENT_SCOPE)
endfunction()

# Compute backend labels
function(compute_backend_labels backends output_var)
    if(NOT backends OR backends STREQUAL "all")
        set(${output_var} "backend_all" PARENT_SCOPE)
    else()
        set(BACKEND_LABELS "")
        foreach(backend ${backends})
            list(APPEND BACKEND_LABELS "backend_${backend}")
        endforeach()
        set(${output_var} "${BACKEND_LABELS}" PARENT_SCOPE)
    endif()
endfunction()

# Compute GPU labels
function(compute_gpu_labels gpus output_var)
    if(NOT gpus OR gpus STREQUAL "all")
        set(${output_var} "gpu_all" PARENT_SCOPE)
    else()
        set(GPU_LABELS "")
        foreach(gpu ${gpus})
            list(APPEND GPU_LABELS "gpu_${gpu}")
        endforeach()
        set(${output_var} "${GPU_LABELS}" PARENT_SCOPE)
    endif()
endfunction()

###############################################################################
# Group Context Management
###############################################################################

function(begin_test_group)
    set(oneValueArgs CATEGORY TIER)  # TIER kept for backward compatibility but ignored
    set(multiValueArgs BACKENDS GPUS EXTRA_LABELS)
    cmake_parse_arguments(GROUP "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Parse CATEGORY into labels
    set(CATEGORY_LABELS "")
    if(DEFINED GROUP_CATEGORY)
        string(REPLACE ";" "," TEMP "${GROUP_CATEGORY}")
        string(REPLACE "," ";" CATEGORY_LABELS "${TEMP}")
    endif()

    # Combine category and extra labels
    set(ALL_GROUP_LABELS ${CATEGORY_LABELS})
    if(DEFINED GROUP_EXTRA_LABELS)
        list(APPEND ALL_GROUP_LABELS ${GROUP_EXTRA_LABELS})
    endif()

    # Store in parent scope (TIER is ignored but kept for backward compatibility)
    set(CURRENT_GROUP_BACKENDS "${GROUP_BACKENDS}" PARENT_SCOPE)
    set(CURRENT_GROUP_GPUS "${GROUP_GPUS}" PARENT_SCOPE)
    set(CURRENT_GROUP_LABELS "${ALL_GROUP_LABELS}" PARENT_SCOPE)
endfunction()

function(end_test_group)
    unset(CURRENT_GROUP_BACKENDS PARENT_SCOPE)
    unset(CURRENT_GROUP_GPUS PARENT_SCOPE)
    unset(CURRENT_GROUP_LABELS PARENT_SCOPE)
endfunction()

###############################################################################
# Main Test Registration Function with Automatic Variant Multiplication
###############################################################################

function(add_rocshmem_functional_test)
    set(options NO_VERIFY)
    set(oneValueArgs NAME RANKS WORKGROUPS THREADS MAX_MSG_SIZE VOLUME_SIZE
                     LOCALBUFTYPE TIMEOUT TIER)  # TIER kept for backward compatibility but ignored
    set(multiValueArgs ENV_VARS EXTRA_LABELS BACKENDS GPUS TEST_VARIANTS)
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Use group defaults if not specified
    if(NOT DEFINED TEST_BACKENDS AND DEFINED CURRENT_GROUP_BACKENDS)
        set(TEST_BACKENDS "${CURRENT_GROUP_BACKENDS}")
    endif()
    if(NOT DEFINED TEST_GPUS AND DEFINED CURRENT_GROUP_GPUS)
        set(TEST_GPUS "${CURRENT_GROUP_GPUS}")
    endif()
    if(NOT DEFINED TEST_EXTRA_LABELS AND DEFINED CURRENT_GROUP_LABELS)
        set(TEST_EXTRA_LABELS "${CURRENT_GROUP_LABELS}")
    endif()

    # Set defaults
    if(NOT DEFINED TEST_BACKENDS)
        set(TEST_BACKENDS "all")
    endif()
    if(NOT DEFINED TEST_GPUS)
        set(TEST_GPUS "all")
    endif()

    # Global variants are always applied to all tests
    set(global_variants ${ALL_GLOBAL_VARIANT_NAMES})

    # Get test-specific variants (if any)
    if(DEFINED TEST_TEST_VARIANTS)
        set(test_variants ${TEST_TEST_VARIANTS})
    else()
        set(test_variants "")
    endif()

    # Generate all variant combinations
    generate_variant_combinations("${global_variants}" "${test_variants}" variant_combinations)

    # Create a CTest test for each variant combination
    foreach(combo ${variant_combinations})
        # Parse combination to get env vars and labels
        parse_variant_combination(
            "${combo}"
            "${global_variants}"
            "${test_variants}"
            variant_env_vars
            variant_labels
            variant_suffix
        )

        # Combine with test-specific env vars
        set(combined_env_vars ${variant_env_vars})
        if(DEFINED TEST_ENV_VARS)
            list(APPEND combined_env_vars ${TEST_ENV_VARS})
        endif()

        # Build multi-dimensional labels
        # NOTE: Tier labels (quick/standard/comprehensive/full) are now applied
        # via YAML-based categorization in test_categories.yaml
        compute_backend_labels("${TEST_BACKENDS}" BACKEND_LABELS)
        compute_gpu_labels("${TEST_GPUS}" GPU_LABELS)

        set(all_labels "functional")
        list(APPEND all_labels ${BACKEND_LABELS})
        list(APPEND all_labels ${GPU_LABELS})
        list(APPEND all_labels ${variant_labels})

        # Add launcher label
        if(USE_SLR_LAUNCHER)
            list(APPEND all_labels "launcher_slr")
        else()
            list(APPEND all_labels "launcher_mpi")
        endif()

        if(DEFINED TEST_EXTRA_LABELS)
            list(APPEND all_labels ${TEST_EXTRA_LABELS})
        endif()

        # Call internal function to create actual CTest test
        # Only pass NO_VERIFY flag if it was explicitly set
        if(TEST_NO_VERIFY)
            _add_single_rocshmem_test(
                NAME ${TEST_NAME}
                RANKS ${TEST_RANKS}
                WORKGROUPS ${TEST_WORKGROUPS}
                THREADS ${TEST_THREADS}
                MAX_MSG_SIZE ${TEST_MAX_MSG_SIZE}
                VOLUME_SIZE ${TEST_VOLUME_SIZE}
                LOCALBUFTYPE ${TEST_LOCALBUFTYPE}
                TIMEOUT ${TEST_TIMEOUT}
                SUFFIX "${variant_suffix}"
                ENV_VARS ${combined_env_vars}
                LABELS "${all_labels}"
                NO_VERIFY
            )
        else()
            _add_single_rocshmem_test(
                NAME ${TEST_NAME}
                RANKS ${TEST_RANKS}
                WORKGROUPS ${TEST_WORKGROUPS}
                THREADS ${TEST_THREADS}
                MAX_MSG_SIZE ${TEST_MAX_MSG_SIZE}
                VOLUME_SIZE ${TEST_VOLUME_SIZE}
                LOCALBUFTYPE ${TEST_LOCALBUFTYPE}
                TIMEOUT ${TEST_TIMEOUT}
                SUFFIX "${variant_suffix}"
                ENV_VARS ${combined_env_vars}
                LABELS "${all_labels}"
            )
        endif()
    endforeach()
endfunction()

###############################################################################
# Internal Function - Creates Single CTest Test
###############################################################################

function(_add_single_rocshmem_test)
    set(options NO_VERIFY)
    set(oneValueArgs NAME RANKS WORKGROUPS THREADS MAX_MSG_SIZE VOLUME_SIZE
                     LOCALBUFTYPE TIMEOUT SUFFIX)
    set(multiValueArgs ENV_VARS LABELS)
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT DEFINED TEST_NAME OR NOT DEFINED TEST_RANKS OR NOT DEFINED TEST_WORKGROUPS OR NOT DEFINED TEST_THREADS)
        message(FATAL_ERROR "_add_single_rocshmem_test: NAME, RANKS, WORKGROUPS, and THREADS are required")
    endif()

    # Get test number from mapping
    if(NOT DEFINED TEST_${TEST_NAME})
        message(FATAL_ERROR "Unknown test name: ${TEST_NAME}")
    endif()
    set(TEST_NUM ${TEST_${TEST_NAME}})

    # Build test name following driver.sh convention
    set(FULL_TEST_NAME "${TEST_NAME}_n${TEST_RANKS}_w${TEST_WORKGROUPS}_z${TEST_THREADS}")

    # Add size suffix
    if(DEFINED TEST_MAX_MSG_SIZE)
        set(FULL_TEST_NAME "${FULL_TEST_NAME}_${TEST_MAX_MSG_SIZE}B")
    elseif(DEFINED TEST_VOLUME_SIZE)
        set(FULL_TEST_NAME "${FULL_TEST_NAME}_v${TEST_VOLUME_SIZE}B")
    endif()

    # Add buffer type suffix if specified
    if(DEFINED TEST_LOCALBUFTYPE)
        set(FULL_TEST_NAME "${FULL_TEST_NAME}_${TEST_LOCALBUFTYPE}")
    endif()

    # Add variant suffix if specified
    if(DEFINED TEST_SUFFIX AND NOT TEST_SUFFIX STREQUAL "")
        set(FULL_TEST_NAME "${FULL_TEST_NAME}_${TEST_SUFFIX}")
    endif()

    # Environment variables for the test wrapper (NOT for MPI ranks)
    set(DEFAULT_ENV)

    # Set default log directory
    set(DEFAULT_LOG_DIR "${CMAKE_CURRENT_BINARY_DIR}/test_logs")
    list(APPEND DEFAULT_ENV "LOG_DIR=${DEFAULT_LOG_DIR}")

    # Combine with user-specified env vars (for wrapper script)
    set(ALL_ENV ${DEFAULT_ENV})

    # Default timeout
    if(NOT DEFINED TEST_TIMEOUT)
        set(TEST_TIMEOUT 300)  # 5 minutes
    endif()

    # Build test command - choose launcher based on MPI availability
    if(USE_SLR_LAUNCHER)
        # SLR mode: Direct execution with ROCSHMEM_SLR_NP environment variable
        # Use system env instead of cmake -E env for portability (build-host cmake path doesn't leak into install)
        set(TEST_COMMAND
            env "ROCSHMEM_SLR_NP=${TEST_RANKS}"
            "ROCSHMEM_MAX_NUM_CONTEXTS=${TEST_WORKGROUPS}"
            "ROCSHMEM_HEAP_SIZE=6442450944"
        )

        # Add LOCALBUFTYPE if specified
        if(DEFINED TEST_LOCALBUFTYPE)
            list(APPEND TEST_COMMAND "LOCALBUFTYPE=${TEST_LOCALBUFTYPE}")
        endif()

        # Add variant-specific and user-specified environment variables
        foreach(ENV_VAR ${TEST_ENV_VARS})
            list(APPEND TEST_COMMAND "${ENV_VAR}")
        endforeach()

        # Add wrapper script and test executable
        list(APPEND TEST_COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/test_wrapper.sh
            ${FULL_TEST_NAME}
            $<TARGET_FILE:rocshmem_functional_tests>
        )
    else()
        # MPI mode: Build test command using standard CMake MPI variables
        set(TEST_COMMAND
            ${CMAKE_CURRENT_SOURCE_DIR}/test_wrapper.sh
            ${FULL_TEST_NAME}
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} ${TEST_RANKS}
            ${MPIEXEC_PREFLAGS}
            -mca pml ${OMPI_MCA_PML}
            -mca osc ${OMPI_MCA_OSC}
        )

        # Export environment variables to MPI ranks via -x flags
        list(APPEND TEST_COMMAND
            -x "ROCSHMEM_MAX_NUM_CONTEXTS=${TEST_WORKGROUPS}"
            -x "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384"
            -x "ROCSHMEM_HEAP_SIZE=6442450944"
        )

        # Export LOCALBUFTYPE if specified
        if(DEFINED TEST_LOCALBUFTYPE)
            list(APPEND TEST_COMMAND -x "LOCALBUFTYPE=${TEST_LOCALBUFTYPE}")
        endif()

        # Export variant-specific and user-specified environment variables
        foreach(ENV_VAR ${TEST_ENV_VARS})
            list(APPEND TEST_COMMAND -x "${ENV_VAR}")
        endforeach()

        # Add timeout if non-zero
        if(TEST_TIMEOUT GREATER 0)
            list(APPEND TEST_COMMAND --timeout ${TEST_TIMEOUT})
        endif()

        list(APPEND TEST_COMMAND --map-by numa)

        # Add hostfile if provided via environment
        if(DEFINED ENV{HOSTFILE})
            list(APPEND TEST_COMMAND --hostfile $ENV{HOSTFILE})
        endif()

        # Add the actual test executable
        list(APPEND TEST_COMMAND $<TARGET_FILE:rocshmem_functional_tests>)
    endif()

    # Add test arguments (common to both MPI and SLR)
    list(APPEND TEST_COMMAND
        -a ${TEST_NUM}
        -w ${TEST_WORKGROUPS}
        -z ${TEST_THREADS}
    )

    # Add size argument
    if(DEFINED TEST_MAX_MSG_SIZE)
        list(APPEND TEST_COMMAND -s ${TEST_MAX_MSG_SIZE})
    elseif(DEFINED TEST_VOLUME_SIZE)
        list(APPEND TEST_COMMAND -v ${TEST_VOLUME_SIZE})
    endif()

    # Add verification flag
    if(TEST_NO_VERIFY)
        list(APPEND TEST_COMMAND -noverif)
    endif()

    # Add buffer type
    if(DEFINED TEST_LOCALBUFTYPE)
        list(APPEND TEST_COMMAND -localbuftype ${TEST_LOCALBUFTYPE})
    else()
        list(APPEND TEST_COMMAND -localbuftype heap)
    endif()

    # Add the test
    add_test(
        NAME ${FULL_TEST_NAME}
        COMMAND ${TEST_COMMAND}
    )

    # Set test properties
    if(TEST_TIMEOUT GREATER 0)
        set_tests_properties(${FULL_TEST_NAME} PROPERTIES
            ENVIRONMENT "${ALL_ENV}"
            TIMEOUT ${TEST_TIMEOUT}
            SKIP_RETURN_CODE 125
            LABELS "${TEST_LABELS}"
            PROCESSORS ${TEST_RANKS}
            WILL_FAIL FALSE
            FAIL_REGULAR_EXPRESSION "FAILED"
        )
    else()
        # Heatmap tests - no timeout
        set_tests_properties(${FULL_TEST_NAME} PROPERTIES
            ENVIRONMENT "${ALL_ENV}"
            SKIP_RETURN_CODE 125
            LABELS "${TEST_LABELS}"
            PROCESSORS ${TEST_RANKS}
            WILL_FAIL FALSE
            FAIL_REGULAR_EXPRESSION "FAILED"
        )
    endif()

    # Also write install-time test definition if generating install tests
    write_install_test_definition("${FULL_TEST_NAME}" "${TEST_COMMAND}" "${TEST_LABELS}" "${TEST_TIMEOUT}" "${TEST_RANKS}")
endfunction()

###############################################################################
# Test Definitions (using multi-dimensional label system)
###############################################################################

# RMA Put Tests
function(add_rma_put_tests)
    # Quick tier PUT tests - all backends, all GPUs
    begin_test_group(CATEGORY "RMA;PUT" TIER quick BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576)
    end_test_group()

    # Standard tier PUT tests
    begin_test_group(CATEGORY "RMA;PUT" TIER standard BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 512)
        add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 1048576)
    end_test_group()

    # Comprehensive tier PUT tests
    begin_test_group(CATEGORY "RMA;PUT" TIER comprehensive BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8)
        add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 32 THREADS 256 MAX_MSG_SIZE 512)
    end_test_group()

    # Full tier PUT tests
    begin_test_group(CATEGORY "RMA;PUT" TIER full BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 64 THREADS 1024 MAX_MSG_SIZE 8)
    end_test_group()

    # Context PUT tests
    begin_test_group(CATEGORY "RMA;PUT;CTX" TIER quick BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME defaultctxput RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024)
    end_test_group()

    begin_test_group(CATEGORY "RMA;PUT;CTX;TEAM" TIER quick BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME teamctxput RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024)
    end_test_group()

    begin_test_group(CATEGORY "RMA;PUT;CTX;TEAM" TIER standard BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME teamctxput RANKS 2 WORKGROUPS 16 THREADS 256 MAX_MSG_SIZE 1024)
    end_test_group()

    # Workgroup PUT tests
    begin_test_group(CATEGORY "RMA;PUT;WG" TIER comprehensive BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 16 THREADS 64 MAX_MSG_SIZE 8)
    end_test_group()

    # Wave PUT tests
    begin_test_group(CATEGORY "RMA;PUT;WAVE" TIER comprehensive BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 2 THREADS 128 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8)
    end_test_group()

    # Non-blocking PUT tests
    begin_test_group(CATEGORY "RMA;PUT;NBI" TIER comprehensive BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME p RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 128)
        add_rocshmem_functional_test(NAME p RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 2)
        add_rocshmem_functional_test(NAME p RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 32)
        add_rocshmem_functional_test(NAME p RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 4)

        add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 512)
        add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8)
        add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 32 THREADS 256 MAX_MSG_SIZE 512)
        add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 64 THREADS 1024 MAX_MSG_SIZE 8)

        add_rocshmem_functional_test(NAME defaultctxputnbi RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024)
        add_rocshmem_functional_test(NAME teamctxputnbi RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024)
        add_rocshmem_functional_test(NAME teamctxputnbi RANKS 2 WORKGROUPS 16 THREADS 256 MAX_MSG_SIZE 1024)

        add_rocshmem_functional_test(NAME wgputnbi RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wgputnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wgputnbi RANKS 2 WORKGROUPS 16 THREADS 64 MAX_MSG_SIZE 8)

        add_rocshmem_functional_test(NAME waveputnbi RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveputnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveputnbi RANKS 2 WORKGROUPS 2 THREADS 128 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveputnbi RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8)
    end_test_group()
endfunction()

# RMA Get Tests (don't work with RO backend - AIROCSHMEM-120)
function(add_rma_get_tests)
    # Quick tier GET tests - IPC and GDA only (not RO)
    begin_test_group(CATEGORY "RMA;GET" TIER quick BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576)
    end_test_group()

    # Standard tier GET tests
    begin_test_group(CATEGORY "RMA;GET" TIER standard BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 512)
        add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8)
        add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 32 THREADS 256 MAX_MSG_SIZE 512)
        add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 64 THREADS 1024 MAX_MSG_SIZE 8)
    end_test_group()

    # Context GET tests
    begin_test_group(CATEGORY "RMA;GET;CTX" TIER quick BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME defaultctxget RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024)
    end_test_group()

    begin_test_group(CATEGORY "RMA;GET;CTX;TEAM" TIER quick BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME teamctxget RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024)
    end_test_group()

    begin_test_group(CATEGORY "RMA;GET;CTX;TEAM" TIER standard BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME teamctxget RANKS 2 WORKGROUPS 16 THREADS 256 MAX_MSG_SIZE 1024)
    end_test_group()

    # Workgroup GET tests
    begin_test_group(CATEGORY "RMA;GET;WG" TIER comprehensive BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 16 THREADS 64 MAX_MSG_SIZE 8)
    end_test_group()

    # Wave GET tests
    begin_test_group(CATEGORY "RMA;GET;WAVE" TIER comprehensive BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 2 THREADS 128 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8)
    end_test_group()

    # Scalar g test - IPC only (not GDA - AIROCSHMEM-162, not RO - AIROCSHMEM-120)
    begin_test_group(CATEGORY "RMA;GET;NBI" TIER comprehensive BACKENDS "ipc" GPUS "all")
        add_rocshmem_functional_test(NAME g RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 128)
        add_rocshmem_functional_test(NAME g RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 1)
        add_rocshmem_functional_test(NAME g RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 32)
        add_rocshmem_functional_test(NAME g RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 4)
    end_test_group()

    # Non-blocking GET tests
    begin_test_group(CATEGORY "RMA;GET;NBI" TIER comprehensive BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 512)
        add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8)
        add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 32 THREADS 256 MAX_MSG_SIZE 512)
        add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 64 THREADS 1024 MAX_MSG_SIZE 8)

        add_rocshmem_functional_test(NAME defaultctxgetnbi RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024)
        add_rocshmem_functional_test(NAME teamctxgetnbi RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024)
        add_rocshmem_functional_test(NAME teamctxgetnbi RANKS 2 WORKGROUPS 16 THREADS 256 MAX_MSG_SIZE 1024)

        add_rocshmem_functional_test(NAME wggetnbi RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wggetnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wggetnbi RANKS 2 WORKGROUPS 16 THREADS 64 MAX_MSG_SIZE 8)

        add_rocshmem_functional_test(NAME wavegetnbi RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wavegetnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wavegetnbi RANKS 2 WORKGROUPS 2 THREADS 128 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wavegetnbi RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8)
    end_test_group()
endfunction()

# AMO Tests
function(add_amo_tests)
    # AMO add tests - don't work with RO backend (AIROCSHMEM-211)
    begin_test_group(CATEGORY "AMO" TIER quick BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME amo_add RANKS 2 WORKGROUPS 1 THREADS 1)
    end_test_group()

    begin_test_group(CATEGORY "AMO" TIER standard BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME amo_add RANKS 2 WORKGROUPS 1 THREADS 1024)
        add_rocshmem_functional_test(NAME amo_add RANKS 2 WORKGROUPS 8 THREADS 1)
        add_rocshmem_functional_test(NAME amo_add RANKS 2 WORKGROUPS 32 THREADS 128)

        add_rocshmem_functional_test(NAME amo_fadd RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME amo_fadd RANKS 2 WORKGROUPS 1 THREADS 1024)
        add_rocshmem_functional_test(NAME amo_fadd RANKS 2 WORKGROUPS 8 THREADS 1)
        add_rocshmem_functional_test(NAME amo_fadd RANKS 2 WORKGROUPS 32 THREADS 128)

        add_rocshmem_functional_test(NAME amo_inc RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME amo_inc RANKS 2 WORKGROUPS 1 THREADS 1024)
        add_rocshmem_functional_test(NAME amo_inc RANKS 2 WORKGROUPS 8 THREADS 1)
        add_rocshmem_functional_test(NAME amo_inc RANKS 2 WORKGROUPS 32 THREADS 128)

        add_rocshmem_functional_test(NAME amo_finc RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME amo_finc RANKS 2 WORKGROUPS 1 THREADS 1024)
        add_rocshmem_functional_test(NAME amo_finc RANKS 2 WORKGROUPS 8 THREADS 1)
        add_rocshmem_functional_test(NAME amo_finc RANKS 2 WORKGROUPS 32 THREADS 128)
    end_test_group()

    # Other AMO tests - work with all backends
    begin_test_group(CATEGORY "AMO" TIER standard BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME amo_set RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME amo_set RANKS 2 WORKGROUPS 8 THREADS 1)
        add_rocshmem_functional_test(NAME amo_set RANKS 2 WORKGROUPS 32 THREADS 1)

        add_rocshmem_functional_test(NAME amo_fetch RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME amo_fetch RANKS 2 WORKGROUPS 1 THREADS 1024)
        add_rocshmem_functional_test(NAME amo_fetch RANKS 2 WORKGROUPS 8 THREADS 1)
        add_rocshmem_functional_test(NAME amo_fetch RANKS 2 WORKGROUPS 32 THREADS 128)

        add_rocshmem_functional_test(NAME amo_fcswap RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME amo_fcswap RANKS 2 WORKGROUPS 32 THREADS 1)
        add_rocshmem_functional_test(NAME amo_fcswap RANKS 2 WORKGROUPS 8 THREADS 1)

        add_rocshmem_functional_test(NAME amo_and RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME amo_fetchand RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME amo_xor RANKS 2 WORKGROUPS 1 THREADS 1)
    end_test_group()
endfunction()

# Signal Operations
function(add_sigops_tests)
    begin_test_group(CATEGORY "SIGOPS" TIER standard BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME putsignal RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME putsignal RANKS 2 WORKGROUPS 2 THREADS 32 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wgputsignal RANKS 2 WORKGROUPS 2 THREADS 32 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveputsignal RANKS 2 WORKGROUPS 1 THREADS 32 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveputsignal RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)

        add_rocshmem_functional_test(NAME putsignalnbi RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME putsignalnbi RANKS 2 WORKGROUPS 2 THREADS 32 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME wgputsignalnbi RANKS 2 WORKGROUPS 2 THREADS 32 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveputsignalnbi RANKS 2 WORKGROUPS 1 THREADS 32 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME waveputsignalnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576)

        add_rocshmem_functional_test(NAME signalfetch RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME wgsignalfetch RANKS 2 WORKGROUPS 2 THREADS 32)
        add_rocshmem_functional_test(NAME wavesignalfetch RANKS 2 WORKGROUPS 1 THREADS 32)
        add_rocshmem_functional_test(NAME wavesignalfetch RANKS 2 WORKGROUPS 1 THREADS 64)
    end_test_group()
endfunction()

# Collective Operations
function(add_coll_tests)
    begin_test_group(CATEGORY "COLLECTIVE" TIER quick BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME init RANKS 2 WORKGROUPS 1 THREADS 1)
    end_test_group()

    begin_test_group(CATEGORY "COLLECTIVE;SYNC" TIER standard BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME syncall RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME wavesyncall RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME wgsyncall RANKS 2 WORKGROUPS 1 THREADS 1)

        add_rocshmem_functional_test(NAME teamsync RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME teamsync RANKS 2 WORKGROUPS 16 THREADS 64)
        add_rocshmem_functional_test(NAME teamsync RANKS 2 WORKGROUPS 32 THREADS 256)
        add_rocshmem_functional_test(NAME teamsync RANKS 2 WORKGROUPS 39 THREADS 1024)

        add_rocshmem_functional_test(NAME teamwavesync RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME teamwavesync RANKS 2 WORKGROUPS 16 THREADS 64)
        add_rocshmem_functional_test(NAME teamwavesync RANKS 2 WORKGROUPS 32 THREADS 256)
        add_rocshmem_functional_test(NAME teamwavesync RANKS 2 WORKGROUPS 39 THREADS 1024)

        add_rocshmem_functional_test(NAME teamwgsync RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME teamwgsync RANKS 2 WORKGROUPS 16 THREADS 64)
        add_rocshmem_functional_test(NAME teamwgsync RANKS 2 WORKGROUPS 32 THREADS 256)
        add_rocshmem_functional_test(NAME teamwgsync RANKS 2 WORKGROUPS 39 THREADS 1024)
    end_test_group()

    begin_test_group(CATEGORY "COLLECTIVE;BARRIER" TIER standard BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME barrierall RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME wavebarrierall RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME wgbarrierall RANKS 2 WORKGROUPS 1 THREADS 1)

        add_rocshmem_functional_test(NAME teambarrier RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME teambarrier RANKS 2 WORKGROUPS 16 THREADS 64)
        add_rocshmem_functional_test(NAME teambarrier RANKS 2 WORKGROUPS 32 THREADS 256)
        add_rocshmem_functional_test(NAME teambarrier RANKS 2 WORKGROUPS 39 THREADS 1024)

        add_rocshmem_functional_test(NAME teamwavebarrier RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME teamwavebarrier RANKS 2 WORKGROUPS 16 THREADS 64)
        add_rocshmem_functional_test(NAME teamwavebarrier RANKS 2 WORKGROUPS 32 THREADS 256)
        add_rocshmem_functional_test(NAME teamwavebarrier RANKS 2 WORKGROUPS 39 THREADS 1024)

        add_rocshmem_functional_test(NAME teamwgbarrier RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME teamwgbarrier RANKS 2 WORKGROUPS 16 THREADS 64)
        add_rocshmem_functional_test(NAME teamwgbarrier RANKS 2 WORKGROUPS 32 THREADS 256)
        add_rocshmem_functional_test(NAME teamwgbarrier RANKS 2 WORKGROUPS 39 THREADS 1024)
    end_test_group()

    begin_test_group(CATEGORY "COLLECTIVE" TIER standard BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME alltoall RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 512)
        add_rocshmem_functional_test(NAME teambroadcast RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 32768)
        add_rocshmem_functional_test(NAME fcollect RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 32768)
        add_rocshmem_functional_test(NAME teamreduction RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 32768)
    end_test_group()

    # Team split 2D test - requires exactly 4 PEs
    begin_test_group(CATEGORY "COLLECTIVE;TEAM" TIER comprehensive BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME teamsplit2d RANKS 4 WORKGROUPS 1 THREADS 1)
    end_test_group()
endfunction()

# Stream Tests
function(add_stream_tests)
    # putmem_on_stream with test-specific variant - standard tier for stream coverage
    begin_test_group(CATEGORY "RMA;PUT;STREAM" TIER standard BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(
            NAME putmem_on_stream
            RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576
            TEST_VARIANTS "default_stream"
        )
    end_test_group()

    # Other stream tests (no test-specific variant)
    begin_test_group(CATEGORY "RMA;GET;STREAM" TIER full BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME getmem_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576)
    end_test_group()

    begin_test_group(CATEGORY "STREAM;SIGOPS" TIER full BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME signal_wait_until_on_stream RANKS 2 WORKGROUPS 1 THREADS 1)
    end_test_group()

    # putmem_signal_on_stream - doesn't work with RO (AIROCSHMEM-217)
    begin_test_group(CATEGORY "RMA;PUT;STREAM;SIGOPS" TIER full BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME putmem_signal_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576)
    end_test_group()

    # barrier_all_on_stream - standard tier for collective stream coverage
    begin_test_group(CATEGORY "COLLECTIVE;STREAM" TIER standard BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME barrier_all_on_stream RANKS 2 WORKGROUPS 1 THREADS 1)
    end_test_group()

    # Other collective stream tests - full tier
    begin_test_group(CATEGORY "COLLECTIVE;STREAM" TIER full BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME quiet_on_stream RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME sync_all_on_stream RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME reduce_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME alltoallmem_on_stream RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME broadcastmem_on_stream RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
    end_test_group()
endfunction()

# Other Tests
function(add_other_tests)
    # Device bitcode tests - quick tier for early validation
    begin_test_group(CATEGORY "OTHER" TIER quick BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME device_bitcode RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME device_bitcode RANKS 2 WORKGROUPS 32 THREADS 1024)
        add_rocshmem_functional_test(NAME device_bitcode RANKS 4 WORKGROUPS 16 THREADS 256)
        add_rocshmem_functional_test(NAME device_bitcode RANKS 8 WORKGROUPS 16 THREADS 128)
    end_test_group()

    # Other miscellaneous tests - full tier
    begin_test_group(CATEGORY "OTHER" TIER full BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME library_info RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME hipmodule_init RANKS 2 WORKGROUPS 1 THREADS 1)

        add_rocshmem_functional_test(NAME pingpong RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME pingpong RANKS 2 WORKGROUPS 8 THREADS 1)
        add_rocshmem_functional_test(NAME pingpong RANKS 2 WORKGROUPS 32 THREADS 1)

        add_rocshmem_functional_test(NAME pingall RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME pingall RANKS 2 WORKGROUPS 8 THREADS 1)
        add_rocshmem_functional_test(NAME pingall RANKS 2 WORKGROUPS 32 THREADS 1)

        add_rocshmem_functional_test(NAME shmemptr RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 8)
        add_rocshmem_functional_test(NAME shmemptr RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 8)
        add_rocshmem_functional_test(NAME shmemptr RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 8)
        add_rocshmem_functional_test(NAME shmemptr RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8)
    end_test_group()

    # Flood tests - don't work with RO backend (AIROCSHMEM-324)
    begin_test_group(CATEGORY "FLOOD;RMA;PUT" TIER full BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME flood_put RANKS 2 WORKGROUPS 64 THREADS 1024)
        add_rocshmem_functional_test(NAME flood_put RANKS 8 WORKGROUPS 64 THREADS 1024)
        add_rocshmem_functional_test(NAME flood_putnbi RANKS 8 WORKGROUPS 64 THREADS 1024)
        add_rocshmem_functional_test(NAME flood_p RANKS 8 WORKGROUPS 64 THREADS 1024)
    end_test_group()

    # Temporarily disabled flood_get tests
    # begin_test_group(CATEGORY "FLOOD;RMA;GET" TIER full BACKENDS "ipc;gda" GPUS "all")
    #     add_rocshmem_functional_test(NAME flood_get RANKS 2 WORKGROUPS 64 THREADS 1024)
    #     add_rocshmem_functional_test(NAME flood_get RANKS 8 WORKGROUPS 64 THREADS 1024)
    #     add_rocshmem_functional_test(NAME flood_getnbi RANKS 8 WORKGROUPS 64 THREADS 1024)
    # end_test_group()

    # Temporarily disabled flood_g test
    # flood_g - only works with IPC (not GDA, not RO)
    # begin_test_group(CATEGORY "FLOOD;RMA;GET" TIER full BACKENDS "ipc" GPUS "all")
    #     add_rocshmem_functional_test(NAME flood_g RANKS 8 WORKGROUPS 64 THREADS 1024)
    # end_test_group()

    begin_test_group(CATEGORY "FLOOD;AMO" TIER full BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME flood_add RANKS 2 WORKGROUPS 64 THREADS 1024)
        add_rocshmem_functional_test(NAME flood_add RANKS 8 WORKGROUPS 64 THREADS 1024)
        add_rocshmem_functional_test(NAME flood_fadd RANKS 8 WORKGROUPS 64 THREADS 1024)
        add_rocshmem_functional_test(NAME flood_waitadd RANKS 8 WORKGROUPS 64 THREADS 1024)
    end_test_group()

    # Host context creation test
    begin_test_group(CATEGORY "CTX;INFRA" TIER full BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME host_ctx_create RANKS 2 WORKGROUPS 1 THREADS 1)
    end_test_group()

    # Team context infrastructure tests - need ROCSHMEM_MAX_NUM_CONTEXTS=1024
    begin_test_group(CATEGORY "TEAM;CTX;INFRA" TIER full BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME teamctxinfra RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
        add_rocshmem_functional_test(NAME teamctxsingleinfra RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
        add_rocshmem_functional_test(NAME teamctxblockinfra RANKS 4 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
        add_rocshmem_functional_test(NAME teamctxblockinfra RANKS 5 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
        add_rocshmem_functional_test(NAME teamctxoddeveninfra RANKS 4 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
        add_rocshmem_functional_test(NAME teamctxoddeveninfra RANKS 5 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
        add_rocshmem_functional_test(NAME teamctxsharedinfra RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
        add_rocshmem_functional_test(NAME teamctxsharedinfra RANKS 5 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
        add_rocshmem_functional_test(NAME teamctxsubsetparentinfra RANKS 4 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
        add_rocshmem_functional_test(NAME teamctxsubsetparentinfra RANKS 5 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024")
    end_test_group()

    # Fence tests - don't work with RO backend (AIROCSHMEM-418)
    begin_test_group(CATEGORY "FENCE" TIER full BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME fence_putwavesignal RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME fence_putwavesignal RANKS 2 WORKGROUPS 8 THREADS 256 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME fence_putwavesignal RANKS 2 WORKGROUPS 32 THREADS 1024 MAX_MSG_SIZE 65536)
        add_rocshmem_functional_test(NAME fence_putlargesmall RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 4096)
        add_rocshmem_functional_test(NAME fence_putlargesmall RANKS 2 WORKGROUPS 8 THREADS 256 MAX_MSG_SIZE 65536)
        add_rocshmem_functional_test(NAME fence_fanout RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME fence_fanout RANKS 4 WORKGROUPS 4 THREADS 256 MAX_MSG_SIZE 65536)
        add_rocshmem_functional_test(NAME fence_fanout RANKS 8 WORKGROUPS 8 THREADS 256 MAX_MSG_SIZE 65536)
        add_rocshmem_functional_test(NAME fence_putwavenbichunks RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576)
        add_rocshmem_functional_test(NAME fence_putwavenbichunks RANKS 2 WORKGROUPS 8 THREADS 256 MAX_MSG_SIZE 65536)
    end_test_group()
endfunction()

# Heatmap Tests
function(add_heatmap_tests)
    # Heatmap RMA GET tests - no timeout, no verification
    begin_test_group(CATEGORY "HEATMAP;RMA;GET" TIER full BACKENDS "ipc;gda" GPUS "all")
        add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 1 THREADS 1 VOLUME_SIZE 1048576 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 32 THREADS 1024 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 1 THREADS 64 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 2 THREADS 64 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 16 THREADS 1024 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 1 THREADS 1024 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 16 THREADS 1024 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
    end_test_group()

    # Heatmap RMA PUT tests
    begin_test_group(CATEGORY "HEATMAP;RMA;PUT" TIER full BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 1 THREADS 1 VOLUME_SIZE 1048576 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 32 THREADS 1024 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 1 THREADS 64 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 2 THREADS 64 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 16 THREADS 1024 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 1 THREADS 1024 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 16 THREADS 1024 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
    end_test_group()

    # Heatmap collective tests
    begin_test_group(CATEGORY "HEATMAP;COLLECTIVE" TIER full BACKENDS "all" GPUS "all")
        add_rocshmem_functional_test(NAME alltoall RANKS 2 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME alltoall RANKS 4 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME alltoall RANKS 8 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME alltoall RANKS 16 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME alltoall RANKS 32 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
        add_rocshmem_functional_test(NAME alltoall RANKS 64 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824 TIMEOUT 0 NO_VERIFY)
    end_test_group()
endfunction()

###############################################################################
# Tile RMA Tests (2D/strided operations)
###############################################################################

function(add_tile_tests)
    # Tile tests are only supported on IPC backend
    # These tests use 2D strided memory access patterns
    begin_test_group(CATEGORY "TILE;RMA;PUT" TIER comprehensive BACKENDS "ipc" GPUS "all")
        add_rocshmem_functional_test(NAME tile_put_contiguous RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_put_rowmajor RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_put_colmajor RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_put_arbitrary RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_put_1d RANKS 2 WORKGROUPS 1 THREADS 1)
    end_test_group()

    # Wavefront and workgroup tile put tests
    begin_test_group(CATEGORY "TILE;RMA;PUT" TIER comprehensive BACKENDS "ipc" GPUS "all")
        # Note: WAVE_SIZE is typically 64 on most AMD GPUs, 32 on gfx11xx
        add_rocshmem_functional_test(NAME tile_put_wave_contiguous RANKS 2 WORKGROUPS 1 THREADS 64)
        add_rocshmem_functional_test(NAME tile_put_wg_contiguous RANKS 2 WORKGROUPS 1 THREADS 1024)
    end_test_group()

    begin_test_group(CATEGORY "TILE;RMA;GET" TIER comprehensive BACKENDS "ipc" GPUS "all")
        add_rocshmem_functional_test(NAME tile_get_contiguous RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_get_rowmajor RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_get_colmajor RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_get_arbitrary RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_get_1d RANKS 2 WORKGROUPS 1 THREADS 1)
    end_test_group()

    # Wavefront and workgroup tile get tests
    begin_test_group(CATEGORY "TILE;RMA;GET" TIER comprehensive BACKENDS "ipc" GPUS "all")
        add_rocshmem_functional_test(NAME tile_get_wave_contiguous RANKS 2 WORKGROUPS 1 THREADS 64)
        add_rocshmem_functional_test(NAME tile_get_wg_contiguous RANKS 2 WORKGROUPS 1 THREADS 1024)
        add_rocshmem_functional_test(NAME tile_get_wg_contiguous RANKS 2 WORKGROUPS 4 THREADS 1024)
    end_test_group()

    # Tile collective tests (broadcast and allgather)
    begin_test_group(CATEGORY "TILE;COLLECTIVE;BROADCAST" TIER comprehensive BACKENDS "ipc" GPUS "all")
        # Thread-level broadcast - test with 2 and 4 PEs
        add_rocshmem_functional_test(NAME tile_broadcast RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_broadcast RANKS 4 WORKGROUPS 1 THREADS 1)
        # Wave-level broadcast
        add_rocshmem_functional_test(NAME tile_broadcast_wave RANKS 2 WORKGROUPS 1 THREADS 64)
        add_rocshmem_functional_test(NAME tile_broadcast_wave RANKS 4 WORKGROUPS 1 THREADS 64)
        # Workgroup-level broadcast
        add_rocshmem_functional_test(NAME tile_broadcast_wg RANKS 2 WORKGROUPS 1 THREADS 1024)
        add_rocshmem_functional_test(NAME tile_broadcast_wg RANKS 4 WORKGROUPS 1 THREADS 1024)
    end_test_group()

    begin_test_group(CATEGORY "TILE;COLLECTIVE;ALLGATHER" TIER comprehensive BACKENDS "ipc" GPUS "all")
        # Thread-level allgather - test with 2 and 4 PEs
        add_rocshmem_functional_test(NAME tile_allgather RANKS 2 WORKGROUPS 1 THREADS 1)
        add_rocshmem_functional_test(NAME tile_allgather RANKS 4 WORKGROUPS 1 THREADS 1)
        # Wave-level allgather
        add_rocshmem_functional_test(NAME tile_allgather_wave RANKS 2 WORKGROUPS 1 THREADS 64)
        add_rocshmem_functional_test(NAME tile_allgather_wave RANKS 4 WORKGROUPS 1 THREADS 64)
        # Workgroup-level allgather
        add_rocshmem_functional_test(NAME tile_allgather_wg RANKS 2 WORKGROUPS 1 THREADS 1024)
        add_rocshmem_functional_test(NAME tile_allgather_wg RANKS 4 WORKGROUPS 1 THREADS 1024)
    end_test_group()
endfunction()

###############################################################################
# Host RMA/AMO Tests (non-MPI IPC TcpBootstrap path, AIROCSHMEM-419)
###############################################################################

function(add_host_tests)
    # Default-context put/get and AMOs - IPC only, always use UUID path
    begin_test_group(CATEGORY "HOST;RMA" TIER comprehensive BACKENDS "ipc" GPUS "all")
        add_rocshmem_functional_test(NAME host_putmem     RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 65536
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_getmem     RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 65536
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
    end_test_group()

    begin_test_group(CATEGORY "HOST;AMO" TIER comprehensive BACKENDS "ipc" GPUS "all")
        add_rocshmem_functional_test(NAME host_amo_fadd   RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_amo_fcswap RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_int_amo_fadd   RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_int_amo_fcswap RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_amo_add    RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1;ROCSHMEM_MAX_NUM_HOST_CONTEXTS=2")
    end_test_group()

    # Explicit-context put/get - need slot 1 available for the explicit ctx
    begin_test_group(CATEGORY "HOST;RMA;CTX" TIER comprehensive BACKENDS "ipc" GPUS "all")
        add_rocshmem_functional_test(NAME host_ctx_putmem RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 65536
            ENV_VARS "ROCSHMEM_TEST_UUID=1;ROCSHMEM_MAX_NUM_HOST_CONTEXTS=2")
        add_rocshmem_functional_test(NAME host_ctx_getmem RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 65536
            ENV_VARS "ROCSHMEM_TEST_UUID=1;ROCSHMEM_MAX_NUM_HOST_CONTEXTS=2")
    end_test_group()

    # Multi-PE concurrency tests (default 4 ranks, mirrors IPC_HOST_NPES=4 in driver.sh)
    begin_test_group(CATEGORY "HOST;AMO" TIER comprehensive BACKENDS "ipc" GPUS "all")
        add_rocshmem_functional_test(NAME host_amo_all_pes RANKS 4 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_amo_self    RANKS 4 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
    end_test_group()

    # P2P sync tests: wait_until / test variants (AIROCSHMEM-419)
    begin_test_group(CATEGORY "HOST;P2P" TIER comprehensive BACKENDS "ipc" GPUS "all")
        add_rocshmem_functional_test(NAME host_wait_until            RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_test                  RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_wait_until_all        RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_wait_until_any        RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_wait_until_some       RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_wait_until_all_vector RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_wait_until_any_vector RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_wait_until_some_vector RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_wait_until_all_status RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_wait_until_any_status RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
        add_rocshmem_functional_test(NAME host_wait_until_some_status RANKS 2 WORKGROUPS 1 THREADS 1
            ENV_VARS "ROCSHMEM_TEST_UUID=1")
    end_test_group()
endfunction()

###############################################################################
# Register all tests
###############################################################################

function(register_all_functional_tests)
    add_rma_put_tests()
    add_rma_get_tests()
    add_amo_tests()
    add_sigops_tests()
    add_coll_tests()
    add_stream_tests()
    add_other_tests()
    add_heatmap_tests()
    add_tile_tests()
    add_host_tests()
endfunction()
