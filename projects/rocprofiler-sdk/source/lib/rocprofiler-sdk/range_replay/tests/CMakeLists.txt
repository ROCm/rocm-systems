rocprofiler_deactivate_clang_tidy()

# C is enabled for the ABI test's C translation unit: the public headers are consumed by C
# tools, and nothing else in the tree compiles them as C.
project(rocprofiler-sdk-range-replay-tests LANGUAGES C CXX)

include(GoogleTest)

# Range eligibility bookkeeping and divergence checking are pure host logic with no GPU or
# HSA runtime dependency, so these are plain CXX unit tests that run unconditionally.
foreach(_TEST range-state digest)
    string(REPLACE "-" "_" _SOURCE_STEM "${_TEST}")
    set(_SOURCE "${_SOURCE_STEM}.cpp")

    add_executable(range-replay-${_TEST}-test)
    target_sources(range-replay-${_TEST}-test PRIVATE ${_SOURCE})

    target_link_libraries(
        range-replay-${_TEST}-test
        PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
                rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
                GTest::gtest_main)

    rocprofiler_add_unit_test(
        TARGET range-replay-${_TEST}-test
        SOURCES ${_SOURCE}
        TIMEOUT 60
        ENVIRONMENT
            "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
        FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
endforeach()

# Layout and C/C++ agreement checks for the public range-replay header. These are properties
# of the header alone: no GPU, no HSA, and no link against the SDK, so the test runs
# anywhere. The C source is half the point of the target; see replay_abi_c.c for why.
set(RANGE_REPLAY_ABI_SOURCES replay_abi.cpp replay_abi_c.c)

add_executable(range-replay-abi-test)
target_sources(range-replay-abi-test PRIVATE ${RANGE_REPLAY_ABI_SOURCES})

set_source_files_properties(replay_abi_c.c PROPERTIES LANGUAGE C)

# Pin the C dialect rather than inheriting whatever the toolchain defaults to, so the check
# means the same thing everywhere. This matches kernel-replay-abi-test.
set_target_properties(range-replay-abi-test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)

target_link_libraries(
    range-replay-abi-test PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers GTest::gtest
                                  GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET range-replay-abi-test
    SOURCES ${RANGE_REPLAY_ABI_SOURCES}
    TIMEOUT 60
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
