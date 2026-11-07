#
# rocprofv3 --replay-mode range. The CLI decisions need no GPU: one run for every counter group,
# the beta acknowledgement, and the same services kernel replay declines.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-range-replay
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

rocprofiler_configure_pytest_files(COPY test_range_replay_cli.py)

add_test(
    NAME rocprofv3-test-range-replay-cli-unit
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_range_replay_cli.py
            --script $<TARGET_FILE:rocprofiler-sdk::rocprofv3>)

set_tests_properties(
    rocprofv3-test-range-replay-cli-unit
    PROPERTIES TIMEOUT 60 LABELS "unit-tests;range-replay" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
