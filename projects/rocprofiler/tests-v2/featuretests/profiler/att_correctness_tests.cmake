# counter correctness test
add_test(
    NAME att_correctness_vectoradd_run
    COMMAND
        ${PROJECT_BINARY_DIR}/rocprofv2 -i
        ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps/goldentraces/att_vadd.txt
        -d /tmp/tests-v2/att/ -o /tmp/tests-v2/att/vadd
        --plugin att auto --mode csv tests-v2/featuretests/profiler/apps/att_vectoradd
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    att_correctness_vectoradd_run PROPERTIES LABELS "v2;rocprofv2" ENVIRONMENT
                                        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

add_test(
    NAME att_correctness_vectoradd_parse
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_att_vectoradd.py
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    att_correctness_vectoradd_parse
    PROPERTIES DEPENDS
                att_correctness_vectoradd_run
                LABELS
                "v2;validation"
                PASS_REGULAR_EXPRESSION
                "Test Passed"
                FAIL_REGULAR_EXPRESSION
                "AssertionError"
                SKIP_REGULAR_EXPRESSION
                "Skipped")
