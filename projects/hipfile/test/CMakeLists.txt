# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(AISAddExecutable)

set(SHARED_SOURCE_FILES
    common/magic-word.cpp
)

# These tests do not require a HIP capable GPU.
# They should pass on a CPU-only system.
set(UNIT_TEST_SOURCE_FILES
)

# These tests call into the GPU Driver.
# They will not pass if a HIP capable GPU is not present
# and configured on the system.
set(SYSTEM_TEST_SOURCE_FILES
    system/async.cpp
    system/buffer.cpp
    system/config.cpp
    system/driver.cpp
    system/main.cpp
    system/version.cpp
)

set(TEST_SYSINCLS
    ${HIPFILE_INCLUDE_PATH}
    ${HIPFILE_TEST_COMMON_PATH}
    ${HIPFILE_THIRD_PARTY_PATH}
)

if(AIS_BUILD_NVIDIA_DETAIL)
    list(APPEND UNIT_TEST_SOURCE_FILES nvidia_detail/cufile-api-compat.cpp)
    list(APPEND TEST_SYSINCLS ${HIPFILE_NVIDIA_SOURCE_PATH})
else()
    list(APPEND SYSTEM_TEST_SOURCE_FILES system/amd/io.cpp)
    list(APPEND TEST_SYSINCLS ${HIPFILE_AMD_SOURCE_PATH})
endif()

# hipfile_tests holds the platform-portable unit tests. Today those sources only
# exist for NVIDIA (cufile-api-compat); on AMD the unit tests live entirely in
# test/amd_detail (internal_tests), so this target would have zero gtest cases.
# Only define it when it will actually contain tests -- gating on the source list
# (rather than the platform) means the target reappears automatically if a
# portable unit test is later added to UNIT_TEST_SOURCE_FILES.
if(UNIT_TEST_SOURCE_FILES)
    ais_add_test_executable(
        NAME hipfile_tests
        DEPS hipfile
        SRCS ${UNIT_TEST_SOURCE_FILES} ${SHARED_SOURCE_FILES}
        SYSINCLS ${TEST_SYSINCLS}
    )
    target_link_libraries(hipfile_tests PRIVATE GTest::gtest)
    target_link_libraries(hipfile_tests PRIVATE GTest::gtest_main)
    # -pthread is critical for sanitizer builds via TheRock
    # NEVER REMOVE IT
    target_compile_options(hipfile_tests PRIVATE -pthread)
    target_link_options(hipfile_tests PRIVATE -pthread)

    ais_gtest_discover_tests(
        hipfile_tests
        PROPERTIES "LABELS;unit;LABELS;hipfile"
        TEST_LIST hipfile_unit_tests
    )
endif()

ais_add_test_executable(
    NAME hipfile_system_tests
    DEPS hipfile
    SRCS ${SYSTEM_TEST_SOURCE_FILES} ${SHARED_SOURCE_FILES}
    SYSINCLS ${TEST_SYSINCLS}
)
target_link_libraries(hipfile_system_tests PRIVATE GTest::gtest)
# -pthread is critical for sanitizer builds via TheRock
# NEVER REMOVE IT
target_compile_options(hipfile_system_tests PRIVATE -pthread)
target_link_options(hipfile_system_tests PRIVATE -pthread)

ais_gtest_discover_tests(
    hipfile_system_tests
    EXTRA_ARGS --ais-capable-dir ${AIS_CAPABLE_DIR}
    PROPERTIES "LABELS;system;LABELS;hipfile"
    TEST_LIST hipfile_system_tests
)

# Temporary Test Files
# Can be deleted once proper testing added
# ----------------------------------------
ais_add_test_executable(
    NAME test_hipfile_basic
    SRCS legacy/test-hipfile.cpp
    DEPS hipfile
)
ais_add_test_executable(
    NAME test_hipfile_batch
    SRCS legacy/test-hipfile-batch.cpp
    DEPS hipfile
)
# ----------------------------------------

# These need setup and command-line options (TBD)
#add_test(NAME test_hipfile_basic COMMAND test_hipfile_basic)
#add_test(NAME test_hipfile_batch COMMAND test_hipfile_batch)
# ----------------------------------------
