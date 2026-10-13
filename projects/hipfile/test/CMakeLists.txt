# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(HipFileAddExecutable)

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
    system/batch.cpp
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

if(HIPFILE_BUILD_NVIDIA_DETAIL)
    list(APPEND UNIT_TEST_SOURCE_FILES nvidia_detail/cufile-api-compat.cpp)
    list(APPEND TEST_SYSINCLS ${HIPFILE_NVIDIA_SOURCE_PATH})
else()
    list(APPEND SYSTEM_TEST_SOURCE_FILES
        system/amd/fd-limit.cpp
        system/amd/io.cpp
        system/amd/io-data-modification.cpp
        system/amd/io-data-modification-kernel.hip
    )
    list(APPEND SHARED_SOURCE_FILES common/ais-capability.cpp)
    list(APPEND TEST_SYSINCLS ${HIPFILE_AMD_SOURCE_PATH})
endif()

# hipfile_tests holds the platform-portable unit tests. Today those sources only
# exist for NVIDIA (cufile-api-compat); on AMD the unit tests live entirely in
# test/amd_detail (internal_tests), so this target would have zero gtest cases.
# Only define it when it will actually contain tests -- gating on the source list
# (rather than the platform) means the target reappears automatically if a
# portable unit test is later added to UNIT_TEST_SOURCE_FILES.
if(UNIT_TEST_SOURCE_FILES)
    hipfile_add_test_executable(
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

    hipfile_gtest_discover_tests(
        hipfile_tests
        PROPERTIES "LABELS;unit;LABELS;hipfile"
        TEST_LIST hipfile_unit_tests
    )
endif()

hipfile_add_test_executable(
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

set(_allow_skip_arg "")
if(HIPFILE_ALLOW_SKIP_FASTPATH_TESTS)
    set(_allow_skip_arg --allow-skip-fastpath)
endif()

hipfile_gtest_discover_tests(
    hipfile_system_tests
    EXTRA_ARGS --ais-capable-dir ${HIPFILE_CAPABLE_DIR} ${_allow_skip_arg}
    # Keep this marker synchronized with the GTEST_SKIP message in system/amd/io.cpp.
    PROPERTIES "LABELS;system;LABELS;hipfile;SKIP_REGULAR_EXPRESSION;fastpath not available"
    TEST_LIST hipfile_system_tests
)

# Deprecated AIS_* CMake options
#
# Configures a small project with every deprecated AIS_* option set and checks
# that each one was mapped to its HIPFILE_* replacement. Remove these tests
# along with cmake/HipFileDeprecatedAISOptions.cmake.
# ----------------------------------------
set(_ais_compat_build_dir "${CMAKE_CURRENT_BINARY_DIR}/ais-compat")
set(_ais_compat_args)
foreach(old_name IN LISTS HIPFILE_DEPRECATED_AIS_OPTIONS)
    list(APPEND _ais_compat_args "-D${old_name}=ais-compat-${old_name}")
endforeach()

# Start from an empty build directory each time so values left in the cache
# by an earlier run can't hide a failure
add_test(NAME hipfile_ais_compat_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf "${_ais_compat_build_dir}"
)
add_test(NAME hipfile_ais_compat
    COMMAND ${CMAKE_COMMAND}
        -G "${CMAKE_GENERATOR}"
        -S "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ais-compat"
        -B "${_ais_compat_build_dir}"
        ${_ais_compat_args}
)
set_tests_properties(hipfile_ais_compat_clean PROPERTIES
    FIXTURES_SETUP hipfile_ais_compat
    LABELS "unit;hipfile"
)
set_tests_properties(hipfile_ais_compat PROPERTIES
    FIXTURES_REQUIRED hipfile_ais_compat
    LABELS "unit;hipfile"
)
# ----------------------------------------

# Temporary Test Files
# Can be deleted once proper testing added
# ----------------------------------------
hipfile_add_test_executable(
    NAME test_hipfile_basic
    SRCS legacy/test-hipfile.cpp
    DEPS hipfile
)
hipfile_add_test_executable(
    NAME test_hipfile_batch
    SRCS legacy/test-hipfile-batch.cpp
    DEPS hipfile
)
# ----------------------------------------

# These need setup and command-line options (TBD)
#add_test(NAME test_hipfile_basic COMMAND test_hipfile_basic)
#add_test(NAME test_hipfile_batch COMMAND test_hipfile_batch)
# ----------------------------------------
