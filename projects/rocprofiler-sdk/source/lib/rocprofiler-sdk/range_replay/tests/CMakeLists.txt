rocprofiler_deactivate_clang_tidy()

# C is enabled for the ABI test's C translation unit: the public headers are consumed by C
# tools, and nothing else in the tree compiles them as C.
project(rocprofiler-sdk-range-replay-tests LANGUAGES C CXX)

include(GoogleTest)

# Range eligibility bookkeeping, divergence checking, the queue hook's early-return paths,
# the kernarg block retention policy, the configuration path (rocprofiler_force_configure,
# the domain's name/operation tables, the range API's no-service error returns), the
# pass-loop decisions and CONFIG/PASS/CLOSE user_data lifecycle, and the executor's
# pre-GPU decisions are pure host logic with no GPU or HSA runtime dependency, so these
# are plain CXX unit tests that run unconditionally. The replay window's drain accounting
# and the host fences run against queues whose signals are plain counters
# (fake_queue.hpp), so they need no runtime either.
#
# They share one executable: every executable that links the static library carries its
# own copy of the SDK's debug info (about 250 MB in RelWithDebInfo) into the tests
# package. ctest still runs each test in its own process.
set(RANGE_REPLAY_TEST_SOURCES
    range_state.cpp
    digest.cpp
    queue_hooks.cpp
    retained_kernarg.cpp
    host_replay.cpp
    replay_window.cpp
    replay_configure.cpp
    replay_phases.cpp
    executor.cpp)

add_executable(range-replay-tests)
target_sources(range-replay-tests PRIVATE ${RANGE_REPLAY_TEST_SOURCES})

target_link_libraries(
    range-replay-tests
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET range-replay-tests
    SOURCES ${RANGE_REPLAY_TEST_SOURCES}
    TIMEOUT 60
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Layout and C/C++ agreement checks for the public range-replay header. These are
# properties of the header alone: no GPU, no HSA, and no link against the SDK, so the test
# runs anywhere. The C source is half the point of the target; see replay_abi_c.c for why.
set(RANGE_REPLAY_ABI_SOURCES replay_abi.cpp replay_abi_c.c)

add_executable(range-replay-abi-test)
target_sources(range-replay-abi-test PRIVATE ${RANGE_REPLAY_ABI_SOURCES})

set_source_files_properties(replay_abi_c.c PROPERTIES LANGUAGE C)

# Pin the C dialect rather than inheriting whatever the toolchain defaults to, so the
# check means the same thing everywhere. This matches kernel-replay-abi-test.
set_target_properties(range-replay-abi-test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED
                                                                     ON)

target_link_libraries(
    range-replay-abi-test PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers GTest::gtest
                                  GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET range-replay-abi-test
    SOURCES ${RANGE_REPLAY_ABI_SOURCES}
    TIMEOUT 60
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
