#
# Unit tests for the benchmark harness itself.
#
# The harness decides which benchmark_config row a run is recorded against, and a run that lands in
# the wrong row is worse than a missing measurement: it is silently averaged together with a run
# that measured something else. These tests are pure Python and sqlite3, so they need neither a GPU
# nor a built rocprofiler, and they run wherever the benchmark project is configured.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-benchmark-tests LANGUAGES NONE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME test-benchmark-replay-bookkeeping
         COMMAND ${Python3_EXECUTABLE} -m unittest discover -s
                 ${CMAKE_CURRENT_SOURCE_DIR} -p "test_*.py" -v)

set_tests_properties(test-benchmark-replay-bookkeeping
                     PROPERTIES TIMEOUT 120 LABELS "benchmark;kernel-replay")
