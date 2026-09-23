#
# Formal models of the queue-hook and kernel-replay state machines.
#
# These check properties that no GPU-free test can reach and that a GPU test would only
# sample: that the shared serialization refcount cannot drift from the serializer, that
# the replay isolation window admits no concurrent dispatch, and that per-pass service
# isolation actually holds. Two of the models are expected to report a violation --
# they pin known gaps (an unbalanced serialization release, and PC sampling ignoring the
# per-pass override) so that a change in either shows up as a test failure rather than
# as a surprise in production.
#
# Neither CBMC nor TLC is a build dependency. When the checker is absent the test reports
# SKIP and costs nothing, so this directory can never redden a build that does not opt in.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-formal LANGUAGES NONE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME test-formal-cbmc
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_formal.py cbmc)

add_test(NAME test-formal-tlc
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/run_formal.py tlc)

set_tests_properties(
    test-formal-cbmc test-formal-tlc
    PROPERTIES TIMEOUT 600 SKIP_RETURN_CODE 2 LABELS
               "integration-tests;kernel-replay;queue-hooks;formal"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
