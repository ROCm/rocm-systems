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
    set(BATCH_OUTPUT "${BATCH_TEST_DIR}/output.bin")

    add_test(
        NAME batch_setup_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BATCH_TEST_DIR}
    )
    add_test(
        NAME batch_cleanup
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${BATCH_TEST_DIR}
    )
    set_tests_properties(batch_setup_dir PROPERTIES
        FIXTURES_SETUP batch_scratch
    )
    set_tests_properties(batch_cleanup PROPERTIES
        FIXTURES_CLEANUP batch_scratch
    )

    # GPUID is optional; the two tests cover both argument forms. They also use
    # different BATCH_SIZE values so both a full-capacity batch and a smaller
    # one that refills many times are exercised.
    add_test(
        NAME batch_roundtrip
        COMMAND $<TARGET_FILE:batch-roundtrip> ${BATCH_INPUT} ${BATCH_OUTPUT} 1048577 128
        WORKING_DIRECTORY ${BATCH_TEST_DIR}
    )
    set_tests_properties(batch_roundtrip PROPERTIES
        FIXTURES_REQUIRED batch_scratch
        LABELS "system;hipfile;batch"
    )
endif()
