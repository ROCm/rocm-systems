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

# CI runs every stress test, so keep the iteration and CPU counts small here.
# The top of hipfileread-cpu-visibility.cpp describes a soak test.
set(_hipfileread_cpu_visibility_args --directory ${HIPFILE_CAPABLE_DIR} --iterations 100 --cpus 8)
set(_hipfile_fastpath_gate_args)
if(HIPFILE_ALLOW_SKIP_FASTPATH_TESTS)
    set(_hipfile_fastpath_gate_args --allow-skip-fastpath)
endif()

# gdb follows hipfile-fastpath-gate's exec, so a crash in
# hipfileread-cpu-visibility still gets a backtrace
add_test(
    NAME hipfileread_cpu_visibility_fastpath_test
    COMMAND gdb --batch --quiet -ex run -ex "thread apply all bt full" --return-child-result
            --args $<TARGET_FILE:hipfile-fastpath-gate> ${_hipfile_fastpath_gate_args} ${HIPFILE_CAPABLE_DIR}
            -- $<TARGET_FILE:hipfileread-cpu-visibility> ${_hipfileread_cpu_visibility_args}
)
add_test(
    NAME hipfileread_cpu_visibility_fallback_test
    COMMAND gdb --batch --quiet -ex run -ex "thread apply all bt full" --return-child-result
            --args $<TARGET_FILE:hipfileread-cpu-visibility> ${_hipfileread_cpu_visibility_args}
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

hipfile_add_test_executable(
    NAME hipfileread-gpu-visibility
    DEPS hipfile
    SRCS "hipfileread-gpu-visibility.cpp" "hipfileread-gpu-visibility-kernel.hip"
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH} ${HIPFILE_INCLUDE_PATH} ${HIPFILE_TEST_COMMON_PATH} ${HIPFILE_THIRD_PARTY_PATH}
)

# One test per backend for each time the checking kernel can run: launched
# after hipFileRead returns, or running during hipFileRead. --prime fills the
# GPU's caches with the previous pattern first. The top of
# hipfileread-gpu-visibility.cpp describes a soak test.
set(_hipfileread_gpu_visibility_args --directory ${HIPFILE_CAPABLE_DIR} --iterations 100 --prime)
foreach(_kernel IN ITEMS after during)
    # gdb follows hipfile-fastpath-gate's exec, so a crash in
    # hipfileread-gpu-visibility still gets a backtrace
    add_test(
        NAME hipfileread_gpu_visibility_${_kernel}_fastpath_test
        COMMAND gdb --batch --quiet -ex run -ex "thread apply all bt full" --return-child-result
                --args $<TARGET_FILE:hipfile-fastpath-gate> ${_hipfile_fastpath_gate_args} ${HIPFILE_CAPABLE_DIR}
                -- $<TARGET_FILE:hipfileread-gpu-visibility> ${_hipfileread_gpu_visibility_args} --kernel ${_kernel}
    )
    add_test(
        NAME hipfileread_gpu_visibility_${_kernel}_fallback_test
        COMMAND gdb --batch --quiet -ex run -ex "thread apply all bt full" --return-child-result
                --args $<TARGET_FILE:hipfileread-gpu-visibility> ${_hipfileread_gpu_visibility_args} --kernel ${_kernel}
    )
    set_tests_properties(
        hipfileread_gpu_visibility_${_kernel}_fastpath_test hipfileread_gpu_visibility_${_kernel}_fallback_test
        PROPERTIES LABELS "stress;internal"
    )
    # Report the fastpath run as skipped rather than failed when, only with
    # --allow-skip-fastpath, hipfile-fastpath-gate reports that fastpath is
    # unavailable. Keep the marker synchronized with the message in
    # hipfile-fastpath-gate.cpp.
    set_tests_properties(hipfileread_gpu_visibility_${_kernel}_fastpath_test PROPERTIES
        ENVIRONMENT "HIPFILE_ALLOW_COMPAT_MODE=false"
        SKIP_REGULAR_EXPRESSION "Fast path is not available"
    )
    set_tests_properties(hipfileread_gpu_visibility_${_kernel}_fallback_test PROPERTIES
        ENVIRONMENT "HIPFILE_FORCE_COMPAT_MODE=true"
    )
endforeach()

hipfile_add_test_executable(
    NAME hipfilewrite-visibility
    DEPS hipfile
    SRCS "hipfilewrite-visibility.cpp" "hipfilewrite-visibility-kernel.hip"
    SYSINCLS ${HIPFILE_AMD_SOURCE_PATH} ${HIPFILE_INCLUDE_PATH} ${HIPFILE_TEST_COMMON_PATH} ${HIPFILE_THIRD_PARTY_PATH}
)

target_link_libraries(hipfilewrite-visibility PRIVATE Threads::Threads)

# One test per backend for each producer whose data hipFileWrite writes today:
# a kernel the host synchronizes with, after an earlier hipFileWrite read the
# buffer; a kernel still running during hipFileWrite; hipMemcpy from host
# memory; and hipFileRead, after an earlier hipFileWrite read the buffer. The
# top of hipfilewrite-visibility.cpp describes the other producers and options,
# some of which fail with fastpath, and a soak test.
set(_hipfilewrite_visibility_args --directory ${HIPFILE_CAPABLE_DIR} --iterations 100)
set(_hipfilewrite_visibility_kernel_after_args --producer kernel --kernel after --prime write)
set(_hipfilewrite_visibility_kernel_during_args --producer kernel --kernel during)
set(_hipfilewrite_visibility_memcpy_args --producer memcpy)
set(_hipfilewrite_visibility_read_args --producer read --prime write)
foreach(_producer IN ITEMS kernel_after kernel_during memcpy read)
    # gdb follows hipfile-fastpath-gate's exec, so a crash in
    # hipfilewrite-visibility still gets a backtrace
    add_test(
        NAME hipfilewrite_visibility_${_producer}_fastpath_test
        COMMAND gdb --batch --quiet -ex run -ex "thread apply all bt full" --return-child-result
                --args $<TARGET_FILE:hipfile-fastpath-gate> ${_hipfile_fastpath_gate_args} ${HIPFILE_CAPABLE_DIR}
                -- $<TARGET_FILE:hipfilewrite-visibility> ${_hipfilewrite_visibility_args}
                ${_hipfilewrite_visibility_${_producer}_args}
    )
    add_test(
        NAME hipfilewrite_visibility_${_producer}_fallback_test
        COMMAND gdb --batch --quiet -ex run -ex "thread apply all bt full" --return-child-result
                --args $<TARGET_FILE:hipfilewrite-visibility> ${_hipfilewrite_visibility_args}
                ${_hipfilewrite_visibility_${_producer}_args}
    )
    set_tests_properties(
        hipfilewrite_visibility_${_producer}_fastpath_test hipfilewrite_visibility_${_producer}_fallback_test
        PROPERTIES LABELS "stress;internal"
    )
    # Report the fastpath run as skipped rather than failed when, only with
    # --allow-skip-fastpath, hipfile-fastpath-gate reports that fastpath is
    # unavailable. Keep the marker synchronized with the message in
    # hipfile-fastpath-gate.cpp.
    set_tests_properties(hipfilewrite_visibility_${_producer}_fastpath_test PROPERTIES
        ENVIRONMENT "HIPFILE_ALLOW_COMPAT_MODE=false"
        SKIP_REGULAR_EXPRESSION "Fast path is not available"
    )
    set_tests_properties(hipfilewrite_visibility_${_producer}_fallback_test PROPERTIES
        ENVIRONMENT "HIPFILE_FORCE_COMPAT_MODE=true"
    )
endforeach()
