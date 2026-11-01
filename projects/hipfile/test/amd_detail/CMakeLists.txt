# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(HipFileAddExecutable)
find_package(Threads REQUIRED)

set(SHARED_SOURCE_FILES
    "${HIPFILE_TEST_COMMON_PATH}/magic-word.cpp"
)

set(TEST_SOURCE_FILES
    api_trace.cpp
    async.cpp
    backend.cpp
    batch/batch.cpp
    configuration.cpp
    context.cpp
    buffer.cpp
    driver.cpp
    environment.cpp
    file-descriptor.cpp
    handle.cpp
    hip.cpp
    hipfile-api.cpp
    hipfile-stats.cpp
    fallback.cpp
    fastpath.cpp
    host.cpp
    main.cpp
    mountinfo.cpp
    stats.cpp
    stream.cpp
    thread-pool.cpp
)

set(TEST_SYSINCLS
    ${HIPFILE_AMD_SOURCE_PATH}
    ${HIPFILE_INCLUDE_PATH}
    ${HIPFILE_TEST_COMMON_PATH}
    ${HIPFILE_AMD_TEST_PATH}
    ${HIPFILE_THIRD_PARTY_PATH}
)

hipfile_add_test_executable(
    NAME internal_tests
    DEPS hipfile
    SRCS ${TEST_SOURCE_FILES} ${SHARED_SOURCE_FILES}
    SYSINCLS ${TEST_SYSINCLS}
)

# Add gtest
target_link_libraries(internal_tests PRIVATE GTest::gmock)
target_link_libraries(internal_tests PRIVATE GTest::gtest)
target_compile_options(internal_tests PRIVATE -pthread)
target_link_options(internal_tests PRIVATE -pthread)

hipfile_gtest_discover_tests(
    internal_tests
    PROPERTIES "LABELS;unit;LABELS;internal"
    TEST_LIST internal_unit_tests
)

# The following tests are concurrency testers. They run for a short
# time to sniff out any egregious lock errors. Be aware that they
# may "pass" in CI for a while after buggy code has been committed,
# so test failures may not be due to the commit in which you
# observe the failure.

hipfile_add_test_executable(
    NAME state_mt
    DEPS hipfile
    SRCS "state_mt.cpp"
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH} ${HIPFILE_INCLUDE_PATH}
)

target_link_libraries(state_mt PRIVATE Threads::Threads)

add_test(
    NAME state_mt_test
    COMMAND gdb --batch --quiet -ex run -ex "thread apply all bt full" --return-child-result --args $<TARGET_FILE:state_mt>
)
set_tests_properties(state_mt_test PROPERTIES LABELS "stress;internal")

hipfile_add_test_executable(
    NAME batch_mt
    DEPS hipfile
    SRCS "batch/batch_mt.cpp"
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH} ${HIPFILE_INCLUDE_PATH}
)

target_link_libraries(batch_mt PRIVATE Threads::Threads)

add_test(
    NAME batch_mt_test
    COMMAND gdb --batch --quiet -ex run -ex "thread apply all bt full" --return-child-result --args $<TARGET_FILE:batch_mt>
)
set_tests_properties(batch_mt_test PROPERTIES LABELS "stress;internal")

hipfile_add_test_executable(
    NAME hipfile-fastpath-gate
    DEPS hipfile
    SRCS "hipfile-fastpath-gate.cpp" "${HIPFILE_TEST_COMMON_PATH}/ais-capability.cpp"
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH} ${HIPFILE_INCLUDE_PATH} ${HIPFILE_TEST_COMMON_PATH} ${HIPFILE_THIRD_PARTY_PATH}
)

hipfile_add_test_executable(
    NAME hipfileread-cpu-visibility
    DEPS hipfile
    SRCS "hipfileread-cpu-visibility.cpp"
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH} ${HIPFILE_INCLUDE_PATH} ${HIPFILE_TEST_COMMON_PATH} ${HIPFILE_THIRD_PARTY_PATH}
)

target_link_libraries(hipfileread-cpu-visibility PRIVATE Threads::Threads)

# CI runs every stress test, so keep the iteration and watcher counts small
# here. The top of hipfileread-cpu-visibility.cpp describes a soak test.
set(_hipfileread_cpu_visibility_args --directory ${HIPFILE_CAPABLE_DIR} --iterations 100 --watchers 8)
set(_hipfile_fastpath_gate_args)
if(HIPFILE_ALLOW_SKIP_FASTPATH_TESTS)
    set(_hipfile_fastpath_gate_args --allow-skip-fastpath)
endif()

add_test(
    NAME hipfileread_cpu_visibility_fastpath_test
    COMMAND $<TARGET_FILE:hipfile-fastpath-gate> ${_hipfile_fastpath_gate_args} ${HIPFILE_CAPABLE_DIR}
            -- $<TARGET_FILE:hipfileread-cpu-visibility> ${_hipfileread_cpu_visibility_args}
)
add_test(
    NAME hipfileread_cpu_visibility_fallback_test
    COMMAND $<TARGET_FILE:hipfileread-cpu-visibility> ${_hipfileread_cpu_visibility_args}
)
set_tests_properties(
    hipfileread_cpu_visibility_fastpath_test hipfileread_cpu_visibility_fallback_test
    PROPERTIES LABELS "stress;internal"
)
# Report these runs as skipped rather than failed: the program's run on a GPU
# whose VRAM the CPU can't map, and, only with --allow-skip-fastpath,
# hipfile-fastpath-gate's report that fastpath is unavailable. Keep the markers
# synchronized with the messages in hipfileread-cpu-visibility.cpp and
# hipfile-fastpath-gate.cpp.
set_tests_properties(hipfileread_cpu_visibility_fastpath_test PROPERTIES
    ENVIRONMENT "HIPFILE_ALLOW_COMPAT_MODE=false"
    SKIP_REGULAR_EXPRESSION "VRAM is not CPU-accessible;Fast path is not available"
)
set_tests_properties(hipfileread_cpu_visibility_fallback_test PROPERTIES
    ENVIRONMENT "HIPFILE_FORCE_COMPAT_MODE=true"
    SKIP_REGULAR_EXPRESSION "VRAM is not CPU-accessible"
)
