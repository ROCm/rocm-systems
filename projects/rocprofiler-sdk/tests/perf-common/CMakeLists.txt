#
# Shared helpers for the performance regression tests, plus their own unit tests.
#
# perf_stats.py decides whether the perf tests pass or fail. A bug in it either hides a real
# regression or reddens a healthy PR, and because the perf tests themselves need a GPU, nothing
# would catch such a bug on a machine without one. These tests are pure Python: no GPU, no HSA and
# no rocprofiler runtime, so they run everywhere and fail fast when a helper changes behaviour.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-perf-common LANGUAGES NONE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME test-perf-common-helpers
         COMMAND ${Python3_EXECUTABLE} -m unittest discover -s
                 ${CMAKE_CURRENT_SOURCE_DIR} -p "test_*.py" -v)

set_tests_properties(
    test-perf-common-helpers
    PROPERTIES TIMEOUT 60 LABELS "integration-tests;kernel-replay;queue-hooks"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
