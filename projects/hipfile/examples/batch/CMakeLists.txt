# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Batch examples for the hipFile C API. examples_common supplies shared file
# and verification helpers and transitively links hipFile and HIP.

include(AISAddExecutable)

ais_add_executable(
    NAME batch-roundtrip
    DEPS examples_common
    SRCS "batch-roundtrip.cpp"
    SYSINCLS ${HIPFILE_INCLUDE_PATH}
)

if(BUILD_TESTING)
    execute_process(
        COMMAND cat /proc/sys/kernel/random/uuid
        OUTPUT_VARIABLE _batch_scratch_token
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(BATCH_TEST_DIR
        "${AIS_CAPABLE_DIR}/hipfile_batch_tests_${_batch_scratch_token}"
    )
    set(BATCH_INPUT "${BATCH_TEST_DIR}/input.bin")
    set(BATCH_EXISTING_INPUT "${BATCH_TEST_DIR}/existing-input.bin")

    add_test(
        NAME batch_setup_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BATCH_TEST_DIR}
    )
    add_test(
        NAME batch_setup_existing_input
        COMMAND dd if=/dev/urandom of=${BATCH_EXISTING_INPUT} bs=1M count=1 status=none
    )
    add_test(
        NAME batch_cleanup
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${BATCH_TEST_DIR}
    )
    set_tests_properties(batch_setup_dir PROPERTIES
        FIXTURES_SETUP batch_scratch
    )
    set_tests_properties(batch_setup_existing_input PROPERTIES
        FIXTURES_SETUP batch_scratch
        DEPENDS batch_setup_dir
    )
    set_tests_properties(batch_cleanup PROPERTIES
        FIXTURES_CLEANUP batch_scratch
    )

    add_test(
        NAME batch_roundtrip_create_input
        COMMAND $<TARGET_FILE:batch-roundtrip> ${BATCH_INPUT} output.bin 1048577 0
        WORKING_DIRECTORY ${BATCH_TEST_DIR}
    )
    set_tests_properties(batch_roundtrip_create_input PROPERTIES
        FIXTURES_REQUIRED batch_scratch
        LABELS "system;hipfile;batch"
    )

    add_test(
        NAME batch_roundtrip_truncate_input
        COMMAND $<TARGET_FILE:batch-roundtrip> ${BATCH_EXISTING_INPUT} existing-output.bin 65537 0
        WORKING_DIRECTORY ${BATCH_TEST_DIR}
    )
    set_tests_properties(batch_roundtrip_truncate_input PROPERTIES
        FIXTURES_REQUIRED batch_scratch
        LABELS "system;hipfile;batch"
    )
endif()
