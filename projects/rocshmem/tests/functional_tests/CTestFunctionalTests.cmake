###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################

# CTest-based functional test definitions
# This replaces the shell-based driver.sh with native CTest integration

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

# Find MPI runtime
find_program(MPIRUN_EXECUTABLE NAMES mpirun mpiexec)
if(NOT MPIRUN_EXECUTABLE)
    message(WARNING "mpirun not found - functional tests will not be added")
    return()
endif()

# Helper function to add a single functional test
# Usage:
#   add_rocshmem_functional_test(
#     NAME <test_name>
#     RANKS <num_ranks>
#     WORKGROUPS <num_workgroups>
#     THREADS <num_threads>
#     [MAX_MSG_SIZE <size>]
#     [VOLUME_SIZE <size>]  # For volume mode tests (heatmap)
#     [LOCALBUFTYPE <type>]  # host, device, fine, uncached, managed
#     [SUFFIX <suffix>]  # Optional suffix to distinguish tests with same params
#     [ENV_VARS <var1> <var2> ...]
#     [TIMEOUT <seconds>]
#     [LABELS <label1> <label2> ...]
#     [NO_VERIFY]  # Disable verification (for heatmap tests)
#   )
function(add_rocshmem_functional_test)
    set(options NO_VERIFY)
    set(oneValueArgs NAME RANKS WORKGROUPS THREADS MAX_MSG_SIZE VOLUME_SIZE LOCALBUFTYPE TIMEOUT SUFFIX)
    set(multiValueArgs ENV_VARS LABELS)
    cmake_parse_arguments(TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    # Validate required arguments
    if(NOT DEFINED TEST_NAME OR NOT DEFINED TEST_RANKS OR NOT DEFINED TEST_WORKGROUPS OR NOT DEFINED TEST_THREADS)
        message(FATAL_ERROR "add_rocshmem_functional_test: NAME, RANKS, WORKGROUPS, and THREADS are required")
    endif()

    # Get test number from mapping
    if(NOT DEFINED TEST_${TEST_NAME})
        message(FATAL_ERROR "Unknown test name: ${TEST_NAME}")
    endif()
    set(TEST_NUM ${TEST_${TEST_NAME}})

    # Build test name following driver.sh convention
    set(FULL_TEST_NAME "${TEST_NAME}_n${TEST_RANKS}_w${TEST_WORKGROUPS}_z${TEST_THREADS}")

    # Add size suffix if specified
    if(DEFINED TEST_MAX_MSG_SIZE)
        set(FULL_TEST_NAME "${FULL_TEST_NAME}_${TEST_MAX_MSG_SIZE}B")
    elseif(DEFINED TEST_VOLUME_SIZE)
        set(FULL_TEST_NAME "${FULL_TEST_NAME}_v${TEST_VOLUME_SIZE}B")
    endif()

    # Add buffer type suffix if specified
    if(DEFINED TEST_LOCALBUFTYPE)
        set(FULL_TEST_NAME "${FULL_TEST_NAME}_${TEST_LOCALBUFTYPE}")
    endif()

    # Add custom suffix if specified (for tests with same params but different env)
    if(DEFINED TEST_SUFFIX)
        set(FULL_TEST_NAME "${FULL_TEST_NAME}_${TEST_SUFFIX}")
    endif()

    # Default environment variables (matching driver.sh)
    set(DEFAULT_ENV
        "ROCSHMEM_MAX_NUM_CONTEXTS=${TEST_WORKGROUPS}"
        "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384"
        "ROCSHMEM_HEAP_SIZE=6442450944"  # 6GB
    )

    # Add LOCALBUFTYPE if specified
    if(DEFINED TEST_LOCALBUFTYPE)
        list(APPEND DEFAULT_ENV "LOCALBUFTYPE=${TEST_LOCALBUFTYPE}")
    endif()

    # Add log directory (can be overridden with ROCSHMEM_TEST_LOG_DIR env var)
    # Default to build/test_logs
    if(NOT DEFINED ENV{ROCSHMEM_TEST_LOG_DIR})
        set(DEFAULT_LOG_DIR "${CMAKE_CURRENT_BINARY_DIR}/test_logs")
    else()
        set(DEFAULT_LOG_DIR "$ENV{ROCSHMEM_TEST_LOG_DIR}")
    endif()
    list(APPEND DEFAULT_ENV "ROCSHMEM_TEST_LOG_DIR=${DEFAULT_LOG_DIR}")

    # Combine with user-specified env vars
    set(ALL_ENV ${DEFAULT_ENV} ${TEST_ENV_VARS})

    # Default timeout (matching driver.sh)
    if(NOT DEFINED TEST_TIMEOUT)
        set(TEST_TIMEOUT 300)  # 5 minutes
    endif()

    # Build test command
    set(TEST_COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/test_wrapper.sh
        ${FULL_TEST_NAME}
        ${MPIRUN_EXECUTABLE}
        -n ${TEST_RANKS}
        -mca pml ucx
        -mca osc ucx
    )

    # Add timeout if non-zero (0 = no timeout, for heatmap tests)
    if(TEST_TIMEOUT GREATER 0)
        list(APPEND TEST_COMMAND --timeout ${TEST_TIMEOUT})
    endif()

    list(APPEND TEST_COMMAND --map-by numa)

    # Add hostfile if provided via environment
    if(DEFINED ENV{HOSTFILE})
        list(APPEND TEST_COMMAND --hostfile $ENV{HOSTFILE})
    endif()

    # Add the actual test executable and arguments
    list(APPEND TEST_COMMAND
        $<TARGET_FILE:rocshmem_functional_tests>
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

    # Build labels list
    set(ALL_LABELS "functional")
    if(DEFINED TEST_LABELS)
        list(APPEND ALL_LABELS ${TEST_LABELS})
    endif()

    # Set test properties directly (don't build as list variable to avoid expansion issues)
    if(TEST_TIMEOUT GREATER 0)
        # Normal tests with timeout
        set_tests_properties(${FULL_TEST_NAME} PROPERTIES
            ENVIRONMENT "${ALL_ENV}"
            TIMEOUT ${TEST_TIMEOUT}
            SKIP_RETURN_CODE 125
            LABELS "${ALL_LABELS}"
            PROCESSORS ${TEST_RANKS}
            WILL_FAIL FALSE
            FAIL_REGULAR_EXPRESSION "FAILED"
        )
    else()
        # Heatmap tests - no timeout
        set_tests_properties(${FULL_TEST_NAME} PROPERTIES
            ENVIRONMENT "${ALL_ENV}"
            SKIP_RETURN_CODE 125
            LABELS "${ALL_LABELS}"
            PROCESSORS ${TEST_RANKS}
            WILL_FAIL FALSE
            FAIL_REGULAR_EXPRESSION "FAILED"
        )
    endif()
endfunction()

###############################################################################
# Test Definitions (matching driver.sh test suites)
###############################################################################

# RMA Put Tests
function(add_rma_put_tests)
    add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT")
    add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 512 LABELS "RMA;PUT")
    add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT")
    add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8 LABELS "RMA;PUT")
    add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 32 THREADS 256 MAX_MSG_SIZE 512 LABELS "RMA;PUT")
    add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 64 THREADS 1024 MAX_MSG_SIZE 8 LABELS "RMA;PUT")

    add_rocshmem_functional_test(NAME defaultctxput RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024 LABELS "RMA;PUT;CTX")
    add_rocshmem_functional_test(NAME teamctxput RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024 LABELS "RMA;PUT;CTX;TEAM")
    add_rocshmem_functional_test(NAME teamctxput RANKS 2 WORKGROUPS 16 THREADS 256 MAX_MSG_SIZE 1024 LABELS "RMA;PUT;CTX;TEAM")

    add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;WG")
    add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;WG")
    add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 16 THREADS 64 MAX_MSG_SIZE 8 LABELS "RMA;PUT;WG")

    add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;WAVE")
    add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;WAVE")
    add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 2 THREADS 128 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;WAVE")
    add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8 LABELS "RMA;PUT;WAVE")

    # Non-blocking
    add_rocshmem_functional_test(NAME p RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 128 LABELS "RMA;PUT;NBI")
    add_rocshmem_functional_test(NAME p RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 2 LABELS "RMA;PUT;NBI")
    add_rocshmem_functional_test(NAME p RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 32 LABELS "RMA;PUT;NBI")
    add_rocshmem_functional_test(NAME p RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 4 LABELS "RMA;PUT;NBI")

    add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;NBI")
    add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 512 LABELS "RMA;PUT;NBI")
    add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;NBI")
    add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8 LABELS "RMA;PUT;NBI")
    add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 32 THREADS 256 MAX_MSG_SIZE 512 LABELS "RMA;PUT;NBI")
    add_rocshmem_functional_test(NAME putnbi RANKS 2 WORKGROUPS 64 THREADS 1024 MAX_MSG_SIZE 8 LABELS "RMA;PUT;NBI")

    add_rocshmem_functional_test(NAME defaultctxputnbi RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024 LABELS "RMA;PUT;NBI;CTX")
    add_rocshmem_functional_test(NAME teamctxputnbi RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024 LABELS "RMA;PUT;NBI;CTX;TEAM")
    add_rocshmem_functional_test(NAME teamctxputnbi RANKS 2 WORKGROUPS 16 THREADS 256 MAX_MSG_SIZE 1024 LABELS "RMA;PUT;NBI;CTX;TEAM")

    add_rocshmem_functional_test(NAME wgputnbi RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;NBI;WG")
    add_rocshmem_functional_test(NAME wgputnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;NBI;WG")
    add_rocshmem_functional_test(NAME wgputnbi RANKS 2 WORKGROUPS 16 THREADS 64 MAX_MSG_SIZE 8 LABELS "RMA;PUT;NBI;WG")

    add_rocshmem_functional_test(NAME waveputnbi RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;NBI;WAVE")
    add_rocshmem_functional_test(NAME waveputnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;NBI;WAVE")
    add_rocshmem_functional_test(NAME waveputnbi RANKS 2 WORKGROUPS 2 THREADS 128 MAX_MSG_SIZE 1048576 LABELS "RMA;PUT;NBI;WAVE")
    add_rocshmem_functional_test(NAME waveputnbi RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8 LABELS "RMA;PUT;NBI;WAVE")
endfunction()

# RMA Get Tests
function(add_rma_get_tests)
    add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "RMA;GET")
    add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 512 LABELS "RMA;GET")
    add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "RMA;GET")
    add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8 LABELS "RMA;GET")
    add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 32 THREADS 256 MAX_MSG_SIZE 512 LABELS "RMA;GET")
    add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 64 THREADS 1024 MAX_MSG_SIZE 8 LABELS "RMA;GET")

    add_rocshmem_functional_test(NAME defaultctxget RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024 LABELS "RMA;GET;CTX")
    add_rocshmem_functional_test(NAME teamctxget RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024 LABELS "RMA;GET;CTX;TEAM")
    add_rocshmem_functional_test(NAME teamctxget RANKS 2 WORKGROUPS 16 THREADS 256 MAX_MSG_SIZE 1024 LABELS "RMA;GET;CTX;TEAM")

    add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;WG")
    add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;WG")
    add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 16 THREADS 64 MAX_MSG_SIZE 8 LABELS "RMA;GET;WG")

    add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;WAVE")
    add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;WAVE")
    add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 2 THREADS 128 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;WAVE")
    add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8 LABELS "RMA;GET;WAVE")

    add_rocshmem_functional_test(NAME g RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 128 LABELS "RMA;GET;NBI")
    add_rocshmem_functional_test(NAME g RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 1 LABELS "RMA;GET;NBI")
    add_rocshmem_functional_test(NAME g RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 32 LABELS "RMA;GET;NBI")
    add_rocshmem_functional_test(NAME g RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 4 LABELS "RMA;GET;NBI")

    # Non-blocking
    add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;NBI")
    add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 512 LABELS "RMA;GET;NBI")
    add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;NBI")
    add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8 LABELS "RMA;GET;NBI")
    add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 32 THREADS 256 MAX_MSG_SIZE 512 LABELS "RMA;GET;NBI")
    add_rocshmem_functional_test(NAME getnbi RANKS 2 WORKGROUPS 64 THREADS 1024 MAX_MSG_SIZE 8 LABELS "RMA;GET;NBI")

    add_rocshmem_functional_test(NAME defaultctxgetnbi RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024 LABELS "RMA;GET;NBI;CTX")
    add_rocshmem_functional_test(NAME teamctxgetnbi RANKS 2 WORKGROUPS 4 THREADS 128 MAX_MSG_SIZE 1024 LABELS "RMA;GET;NBI;CTX;TEAM")
    add_rocshmem_functional_test(NAME teamctxgetnbi RANKS 2 WORKGROUPS 16 THREADS 256 MAX_MSG_SIZE 1024 LABELS "RMA;GET;NBI;CTX;TEAM")

    add_rocshmem_functional_test(NAME wggetnbi RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;NBI;WG")
    add_rocshmem_functional_test(NAME wggetnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;NBI;WG")
    add_rocshmem_functional_test(NAME wggetnbi RANKS 2 WORKGROUPS 16 THREADS 64 MAX_MSG_SIZE 8 LABELS "RMA;GET;NBI;WG")

    add_rocshmem_functional_test(NAME wavegetnbi RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;NBI;WAVE")
    add_rocshmem_functional_test(NAME wavegetnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;NBI;WAVE")
    add_rocshmem_functional_test(NAME wavegetnbi RANKS 2 WORKGROUPS 2 THREADS 128 MAX_MSG_SIZE 1048576 LABELS "RMA;GET;NBI;WAVE")
    add_rocshmem_functional_test(NAME wavegetnbi RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8 LABELS "RMA;GET;NBI;WAVE")
endfunction()

# AMO Tests
function(add_amo_tests)
    add_rocshmem_functional_test(NAME amo_add RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_add RANKS 2 WORKGROUPS 1 THREADS 1024 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_add RANKS 2 WORKGROUPS 8 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_add RANKS 2 WORKGROUPS 32 THREADS 128 LABELS "AMO")

    add_rocshmem_functional_test(NAME amo_fadd RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_fadd RANKS 2 WORKGROUPS 1 THREADS 1024 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_fadd RANKS 2 WORKGROUPS 8 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_fadd RANKS 2 WORKGROUPS 32 THREADS 128 LABELS "AMO")

    add_rocshmem_functional_test(NAME amo_inc RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_inc RANKS 2 WORKGROUPS 1 THREADS 1024 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_inc RANKS 2 WORKGROUPS 8 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_inc RANKS 2 WORKGROUPS 32 THREADS 128 LABELS "AMO")

    add_rocshmem_functional_test(NAME amo_finc RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_finc RANKS 2 WORKGROUPS 1 THREADS 1024 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_finc RANKS 2 WORKGROUPS 8 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_finc RANKS 2 WORKGROUPS 32 THREADS 128 LABELS "AMO")

    add_rocshmem_functional_test(NAME amo_set RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_set RANKS 2 WORKGROUPS 8 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_set RANKS 2 WORKGROUPS 32 THREADS 1 LABELS "AMO")

    add_rocshmem_functional_test(NAME amo_fetch RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_fetch RANKS 2 WORKGROUPS 1 THREADS 1024 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_fetch RANKS 2 WORKGROUPS 8 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_fetch RANKS 2 WORKGROUPS 32 THREADS 128 LABELS "AMO")

    add_rocshmem_functional_test(NAME amo_fcswap RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_fcswap RANKS 2 WORKGROUPS 32 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_fcswap RANKS 2 WORKGROUPS 8 THREADS 1 LABELS "AMO")

    add_rocshmem_functional_test(NAME amo_and RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_fetchand RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
    add_rocshmem_functional_test(NAME amo_xor RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "AMO")
endfunction()

# Signaling Operations Tests
function(add_sigops_tests)
    add_rocshmem_functional_test(NAME putsignal RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "SIGOPS")
    add_rocshmem_functional_test(NAME putsignal RANKS 2 WORKGROUPS 2 THREADS 32 MAX_MSG_SIZE 1048576 LABELS "SIGOPS")
    add_rocshmem_functional_test(NAME wgputsignal RANKS 2 WORKGROUPS 2 THREADS 32 MAX_MSG_SIZE 1048576 LABELS "SIGOPS;WG")
    add_rocshmem_functional_test(NAME waveputsignal RANKS 2 WORKGROUPS 1 THREADS 32 MAX_MSG_SIZE 1048576 LABELS "SIGOPS;WAVE")
    add_rocshmem_functional_test(NAME waveputsignal RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "SIGOPS;WAVE")

    add_rocshmem_functional_test(NAME putsignalnbi RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "SIGOPS;NBI")
    add_rocshmem_functional_test(NAME putsignalnbi RANKS 2 WORKGROUPS 2 THREADS 32 MAX_MSG_SIZE 1048576 LABELS "SIGOPS;NBI")
    add_rocshmem_functional_test(NAME wgputsignalnbi RANKS 2 WORKGROUPS 2 THREADS 32 MAX_MSG_SIZE 1048576 LABELS "SIGOPS;NBI;WG")
    add_rocshmem_functional_test(NAME waveputsignalnbi RANKS 2 WORKGROUPS 1 THREADS 32 MAX_MSG_SIZE 1048576 LABELS "SIGOPS;NBI;WAVE")
    add_rocshmem_functional_test(NAME waveputsignalnbi RANKS 2 WORKGROUPS 2 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "SIGOPS;NBI;WAVE")

    add_rocshmem_functional_test(NAME signalfetch RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "SIGOPS")
    add_rocshmem_functional_test(NAME wgsignalfetch RANKS 2 WORKGROUPS 2 THREADS 32 LABELS "SIGOPS;WG")
    add_rocshmem_functional_test(NAME wavesignalfetch RANKS 2 WORKGROUPS 1 THREADS 32 LABELS "SIGOPS;WAVE")
    add_rocshmem_functional_test(NAME wavesignalfetch RANKS 2 WORKGROUPS 1 THREADS 64 LABELS "SIGOPS;WAVE")
endfunction()

# Collective Tests
function(add_collective_tests)
    add_rocshmem_functional_test(NAME syncall RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;SYNC")
    add_rocshmem_functional_test(NAME wavesyncall RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;SYNC;WAVE")
    add_rocshmem_functional_test(NAME wgsyncall RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;SYNC;WG")

    add_rocshmem_functional_test(NAME teamsync RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;SYNC;TEAM")
    add_rocshmem_functional_test(NAME teamsync RANKS 2 WORKGROUPS 16 THREADS 64 LABELS "COLLECTIVE;SYNC;TEAM")
    add_rocshmem_functional_test(NAME teamsync RANKS 2 WORKGROUPS 32 THREADS 256 LABELS "COLLECTIVE;SYNC;TEAM")
    add_rocshmem_functional_test(NAME teamsync RANKS 2 WORKGROUPS 39 THREADS 1024 LABELS "COLLECTIVE;SYNC;TEAM")

    add_rocshmem_functional_test(NAME teamwavesync RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;SYNC;TEAM;WAVE")
    add_rocshmem_functional_test(NAME teamwavesync RANKS 2 WORKGROUPS 16 THREADS 64 LABELS "COLLECTIVE;SYNC;TEAM;WAVE")
    add_rocshmem_functional_test(NAME teamwavesync RANKS 2 WORKGROUPS 32 THREADS 256 LABELS "COLLECTIVE;SYNC;TEAM;WAVE")
    add_rocshmem_functional_test(NAME teamwavesync RANKS 2 WORKGROUPS 39 THREADS 1024 LABELS "COLLECTIVE;SYNC;TEAM;WAVE")

    add_rocshmem_functional_test(NAME teamwgsync RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;SYNC;TEAM;WG")
    add_rocshmem_functional_test(NAME teamwgsync RANKS 2 WORKGROUPS 16 THREADS 64 LABELS "COLLECTIVE;SYNC;TEAM;WG")
    add_rocshmem_functional_test(NAME teamwgsync RANKS 2 WORKGROUPS 32 THREADS 256 LABELS "COLLECTIVE;SYNC;TEAM;WG")
    add_rocshmem_functional_test(NAME teamwgsync RANKS 2 WORKGROUPS 39 THREADS 1024 LABELS "COLLECTIVE;SYNC;TEAM;WG")

    add_rocshmem_functional_test(NAME barrierall RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;BARRIER")
    add_rocshmem_functional_test(NAME wavebarrierall RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;BARRIER;WAVE")
    add_rocshmem_functional_test(NAME wgbarrierall RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;BARRIER;WG")

    add_rocshmem_functional_test(NAME teambarrier RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;BARRIER;TEAM")
    add_rocshmem_functional_test(NAME teambarrier RANKS 2 WORKGROUPS 16 THREADS 64 LABELS "COLLECTIVE;BARRIER;TEAM")
    add_rocshmem_functional_test(NAME teambarrier RANKS 2 WORKGROUPS 32 THREADS 256 LABELS "COLLECTIVE;BARRIER;TEAM")
    add_rocshmem_functional_test(NAME teambarrier RANKS 2 WORKGROUPS 39 THREADS 1024 LABELS "COLLECTIVE;BARRIER;TEAM")

    add_rocshmem_functional_test(NAME teamwavebarrier RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;BARRIER;TEAM;WAVE")
    add_rocshmem_functional_test(NAME teamwavebarrier RANKS 2 WORKGROUPS 16 THREADS 64 LABELS "COLLECTIVE;BARRIER;TEAM;WAVE")
    add_rocshmem_functional_test(NAME teamwavebarrier RANKS 2 WORKGROUPS 32 THREADS 256 LABELS "COLLECTIVE;BARRIER;TEAM;WAVE")
    add_rocshmem_functional_test(NAME teamwavebarrier RANKS 2 WORKGROUPS 39 THREADS 1024 LABELS "COLLECTIVE;BARRIER;TEAM;WAVE")

    add_rocshmem_functional_test(NAME teamwgbarrier RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "COLLECTIVE;BARRIER;TEAM;WG")
    add_rocshmem_functional_test(NAME teamwgbarrier RANKS 2 WORKGROUPS 16 THREADS 64 LABELS "COLLECTIVE;BARRIER;TEAM;WG")
    add_rocshmem_functional_test(NAME teamwgbarrier RANKS 2 WORKGROUPS 32 THREADS 256 LABELS "COLLECTIVE;BARRIER;TEAM;WG")
    add_rocshmem_functional_test(NAME teamwgbarrier RANKS 2 WORKGROUPS 39 THREADS 1024 LABELS "COLLECTIVE;BARRIER;TEAM;WG")

    add_rocshmem_functional_test(NAME alltoall RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 512 LABELS "COLLECTIVE;ALLTOALL")
    add_rocshmem_functional_test(NAME teambroadcast RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 32768 LABELS "COLLECTIVE;BROADCAST;TEAM")
    add_rocshmem_functional_test(NAME fcollect RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 32768 LABELS "COLLECTIVE;FCOLLECT")
    add_rocshmem_functional_test(NAME teamreduction RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 32768 LABELS "COLLECTIVE;REDUCTION;TEAM")
endfunction()

# Stream-based Tests
function(add_stream_tests)
    add_rocshmem_functional_test(NAME putmem_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "STREAM")

    # Test with default stream (add suffix to distinguish from above)
    add_rocshmem_functional_test(NAME putmem_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576
                                 SUFFIX "default_stream"
                                 ENV_VARS "ROCSHMEM_TEST_USE_DEFAULT_STREAM=1" LABELS "STREAM;DEFAULT_STREAM")

    add_rocshmem_functional_test(NAME getmem_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "STREAM")
    add_rocshmem_functional_test(NAME signal_wait_until_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "STREAM")
    add_rocshmem_functional_test(NAME putmem_signal_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 1048576 LABELS "STREAM")
    add_rocshmem_functional_test(NAME barrier_all_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "STREAM")
    add_rocshmem_functional_test(NAME quiet_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "STREAM")
    add_rocshmem_functional_test(NAME sync_all_on_stream RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "STREAM")
    add_rocshmem_functional_test(NAME alltoallmem_on_stream RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "STREAM")
    add_rocshmem_functional_test(NAME broadcastmem_on_stream RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "STREAM")
endfunction()

# Other Tests
function(add_other_tests)
    add_rocshmem_functional_test(NAME init RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "OTHER")
    add_rocshmem_functional_test(NAME library_info RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "OTHER")
    add_rocshmem_functional_test(NAME hipmodule_init RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "OTHER")

    add_rocshmem_functional_test(NAME device_bitcode RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "OTHER")
    add_rocshmem_functional_test(NAME device_bitcode RANKS 2 WORKGROUPS 32 THREADS 1024 LABELS "OTHER")
    add_rocshmem_functional_test(NAME device_bitcode RANKS 4 WORKGROUPS 16 THREADS 256 LABELS "OTHER")
    add_rocshmem_functional_test(NAME device_bitcode RANKS 8 WORKGROUPS 16 THREADS 128 LABELS "OTHER")

    add_rocshmem_functional_test(NAME pingpong RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "OTHER")
    add_rocshmem_functional_test(NAME pingpong RANKS 2 WORKGROUPS 8 THREADS 1 LABELS "OTHER")
    add_rocshmem_functional_test(NAME pingpong RANKS 2 WORKGROUPS 32 THREADS 1 LABELS "OTHER")

    add_rocshmem_functional_test(NAME pingall RANKS 2 WORKGROUPS 1 THREADS 1 LABELS "OTHER")
    add_rocshmem_functional_test(NAME pingall RANKS 2 WORKGROUPS 8 THREADS 1 LABELS "OTHER")
    add_rocshmem_functional_test(NAME pingall RANKS 2 WORKGROUPS 32 THREADS 1 LABELS "OTHER")

    # Flood tests
    add_rocshmem_functional_test(NAME flood_put RANKS 2 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")
    add_rocshmem_functional_test(NAME flood_put RANKS 8 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")
    add_rocshmem_functional_test(NAME flood_putnbi RANKS 8 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")
    add_rocshmem_functional_test(NAME flood_p RANKS 8 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")

    add_rocshmem_functional_test(NAME flood_get RANKS 2 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")
    add_rocshmem_functional_test(NAME flood_get RANKS 8 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")
    add_rocshmem_functional_test(NAME flood_getnbi RANKS 8 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")
    add_rocshmem_functional_test(NAME flood_g RANKS 8 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")

    add_rocshmem_functional_test(NAME flood_add RANKS 2 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")
    add_rocshmem_functional_test(NAME flood_add RANKS 8 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")
    add_rocshmem_functional_test(NAME flood_fadd RANKS 8 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")
    add_rocshmem_functional_test(NAME flood_waitadd RANKS 8 WORKGROUPS 64 THREADS 1024 LABELS "OTHER;FLOOD")

    # Team context infrastructure tests (require more contexts than workgroups)
    add_rocshmem_functional_test(NAME teamctxinfra RANKS 2 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")
    add_rocshmem_functional_test(NAME teamctxsingleinfra RANKS 2 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")
    add_rocshmem_functional_test(NAME teamctxblockinfra RANKS 4 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")
    add_rocshmem_functional_test(NAME teamctxblockinfra RANKS 5 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")
    add_rocshmem_functional_test(NAME teamctxoddeveninfra RANKS 4 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")
    add_rocshmem_functional_test(NAME teamctxoddeveninfra RANKS 5 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")
    add_rocshmem_functional_test(NAME teamctxsharedinfra RANKS 2 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")
    add_rocshmem_functional_test(NAME teamctxsharedinfra RANKS 5 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")
    add_rocshmem_functional_test(NAME teamctxsubsetparentinfra RANKS 4 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")
    add_rocshmem_functional_test(NAME teamctxsubsetparentinfra RANKS 5 WORKGROUPS 1 THREADS 1
                                 ENV_VARS "ROCSHMEM_MAX_NUM_CONTEXTS=1024" LABELS "OTHER;TEAM;CTX")

    add_rocshmem_functional_test(NAME shmemptr RANKS 2 WORKGROUPS 1 THREADS 1 MAX_MSG_SIZE 8 LABELS "OTHER")
    add_rocshmem_functional_test(NAME shmemptr RANKS 2 WORKGROUPS 1 THREADS 1024 MAX_MSG_SIZE 8 LABELS "OTHER")
    add_rocshmem_functional_test(NAME shmemptr RANKS 2 WORKGROUPS 8 THREADS 1 MAX_MSG_SIZE 8 LABELS "OTHER")
    add_rocshmem_functional_test(NAME shmemptr RANKS 2 WORKGROUPS 16 THREADS 128 MAX_MSG_SIZE 8 LABELS "OTHER")

    # Fence ordering tests
    add_rocshmem_functional_test(NAME fence_putwavesignal RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "OTHER;FENCE")
    add_rocshmem_functional_test(NAME fence_putwavesignal RANKS 2 WORKGROUPS 8 THREADS 256 MAX_MSG_SIZE 1048576 LABELS "OTHER;FENCE")
    add_rocshmem_functional_test(NAME fence_putwavesignal RANKS 2 WORKGROUPS 32 THREADS 1024 MAX_MSG_SIZE 65536 LABELS "OTHER;FENCE")
    add_rocshmem_functional_test(NAME fence_putlargesmall RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 4096 LABELS "OTHER;FENCE")
    add_rocshmem_functional_test(NAME fence_putlargesmall RANKS 2 WORKGROUPS 8 THREADS 256 MAX_MSG_SIZE 65536 LABELS "OTHER;FENCE")
    add_rocshmem_functional_test(NAME fence_fanout RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "OTHER;FENCE")
    add_rocshmem_functional_test(NAME fence_fanout RANKS 4 WORKGROUPS 4 THREADS 256 MAX_MSG_SIZE 65536 LABELS "OTHER;FENCE")
    add_rocshmem_functional_test(NAME fence_fanout RANKS 8 WORKGROUPS 8 THREADS 256 MAX_MSG_SIZE 65536 LABELS "OTHER;FENCE")
    add_rocshmem_functional_test(NAME fence_putwavenbichunks RANKS 2 WORKGROUPS 1 THREADS 64 MAX_MSG_SIZE 1048576 LABELS "OTHER;FENCE")
    add_rocshmem_functional_test(NAME fence_putwavenbichunks RANKS 2 WORKGROUPS 8 THREADS 256 MAX_MSG_SIZE 65536 LABELS "OTHER;FENCE")
endfunction()

# Heatmap RMA Tests (volume mode, no verification, no timeout)
function(add_heatmap_rma_tests)
    # GET tests
    add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 1 THREADS 1 VOLUME_SIZE 1048576
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;GET")
    add_rocshmem_functional_test(NAME get RANKS 2 WORKGROUPS 32 THREADS 1024 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;GET")

    add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 1 THREADS 64 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;GET;WAVE")
    add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 2 THREADS 64 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;GET;WAVE")
    add_rocshmem_functional_test(NAME waveget RANKS 2 WORKGROUPS 16 THREADS 1024 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;GET;WAVE")

    add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 1 THREADS 1024 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;GET;WG")
    add_rocshmem_functional_test(NAME wgget RANKS 2 WORKGROUPS 16 THREADS 1024 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;GET;WG")

    # PUT tests
    add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 1 THREADS 1 VOLUME_SIZE 1048576
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;PUT")
    add_rocshmem_functional_test(NAME put RANKS 2 WORKGROUPS 32 THREADS 1024 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;PUT")

    add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 1 THREADS 64 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;PUT;WAVE")
    add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 2 THREADS 64 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;PUT;WAVE")
    add_rocshmem_functional_test(NAME waveput RANKS 2 WORKGROUPS 16 THREADS 1024 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;PUT;WAVE")

    add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 1 THREADS 1024 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;PUT;WG")
    add_rocshmem_functional_test(NAME wgput RANKS 2 WORKGROUPS 16 THREADS 1024 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;RMA;PUT;WG")
endfunction()

# Heatmap Collective Tests (volume mode, no verification, no timeout)
function(add_heatmap_coll_tests)
    add_rocshmem_functional_test(NAME alltoall RANKS 2 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;COLLECTIVE;ALLTOALL")
    add_rocshmem_functional_test(NAME alltoall RANKS 4 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;COLLECTIVE;ALLTOALL")
    add_rocshmem_functional_test(NAME alltoall RANKS 8 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;COLLECTIVE;ALLTOALL")
    add_rocshmem_functional_test(NAME alltoall RANKS 16 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;COLLECTIVE;ALLTOALL")
    add_rocshmem_functional_test(NAME alltoall RANKS 32 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;COLLECTIVE;ALLTOALL")
    add_rocshmem_functional_test(NAME alltoall RANKS 64 WORKGROUPS 1 THREADS 256 VOLUME_SIZE 1073741824
                                 SUFFIX heatmap NO_VERIFY TIMEOUT 0 LABELS "HEATMAP;COLLECTIVE;ALLTOALL")
endfunction()

# Register all test suites
function(register_all_functional_tests)
    add_rma_put_tests()
    add_rma_get_tests()
    add_amo_tests()
    add_sigops_tests()
    add_collective_tests()
    add_stream_tests()
    add_other_tests()
    add_heatmap_rma_tests()
    add_heatmap_coll_tests()
endfunction()
