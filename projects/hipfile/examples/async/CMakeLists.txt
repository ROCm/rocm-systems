# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Async examples for the hipFile C API. They build with the AIS infrastructure
# (ais_add_executable) and link the shared helpers from the examples_common
# target defined in examples/common; the parent CMakeLists adds examples/common
# before examples/async, so the target is in scope here. examples_common is
# PUBLIC-linked against hipfile and exports its own include dir, so
# DEPS examples_common transitively supplies the library and headers.

include(AISAddExecutable)

set(ASYNC_EXAMPLES
    roundtrip-async
    roundtrip-async-nonblocking-stream
    roundtrip-async-multi-stream
    roundtrip-async-multi-stream-registered
)

foreach(example ${ASYNC_EXAMPLES})
    ais_add_executable(
        NAME ${example}
        DEPS examples_common
        SRCS "${example}.cpp"
        SYSINCLS ${HIPFILE_INCLUDE_PATH}
    )
endforeach()

# CTest wrappers: run each async example as a hipFile system test.
# Gated on BUILD_TESTING; AIS_INSTALL_EXAMPLES is already enforced by the
# parent CMakeLists which only adds this subdirectory when examples are built.
if(BUILD_TESTING)
    # Unique-per-configure scratch dir so concurrent ctest runs / multiple
    # checkouts don't collide on a shared path. Kept under AIS_CAPABLE_DIR
    # because the e2e tests need an O_DIRECT-capable local FS — the build
    # tree may live on overlay/nfs and isn't a safe default.
    execute_process(
        COMMAND cat /proc/sys/kernel/random/uuid
        OUTPUT_VARIABLE _async_scratch_token
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(ASYNC_TEST_DIR
        "${AIS_CAPABLE_DIR}/hipfile_async_tests_${_async_scratch_token}"
    )

    add_test(
        NAME async_setup_dir
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ASYNC_TEST_DIR}
    )
    add_test(
        NAME async_cleanup
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${ASYNC_TEST_DIR}
    )
    set_tests_properties(async_setup_dir PROPERTIES
        FIXTURES_SETUP async_scratch
    )
    set_tests_properties(async_cleanup PROPERTIES
        FIXTURES_CLEANUP async_scratch
    )

    # Each async example calls seed_read_file() to truncate+seed its READ_FILE,
    # so we give every test its own input path. This avoids parallel-run races
    # on a shared input file and removes the need for an upfront dd seeding.
    foreach(example ${ASYNC_EXAMPLES})
        add_test(
            NAME async_${example}
            COMMAND $<TARGET_FILE:${example}> in_${example}.bin out_${example}.bin
            WORKING_DIRECTORY ${ASYNC_TEST_DIR}
        )
        set_tests_properties(async_${example} PROPERTIES
            FIXTURES_REQUIRED async_scratch
            LABELS "system;hipfile;async"
        )
    endforeach()
endif()
