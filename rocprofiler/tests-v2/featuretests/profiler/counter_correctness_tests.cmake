# counter correctness test
add_test(
    NAME pmc_correctnes_vadd_test
    COMMAND
        ${PROJECT_BINARY_DIR}/rocprofv2 -i
        ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps/goldentraces/pmc.txt -d
        /tmp/tests-v2/pmc -o vadd tests-v2/featuretests/profiler/apps/pmc_vectoradd
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    pmc_correctnes_vadd_test PROPERTIES LABELS "v2;rocprofv2" ENVIRONMENT
                                        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

add_test(
    NAME pmc_correctnes_vadd_test_validation
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_vectoradd.py
            "/tmp/tests-v2/pmc/pmc_1/results_vadd.csv"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    pmc_correctnes_vadd_test_validation
    PROPERTIES DEPENDS
               pmc_correctnes_vadd_test
               LABELS
               "v2;validation"
               PASS_REGULAR_EXPRESSION
               "Test Passed"
               FAIL_REGULAR_EXPRESSION
               "Test Failed"
               SKIP_REGULAR_EXPRESSION
               "Skipped")

add_test(
    NAME pmc_correctnes_histogram_test
    COMMAND
        ${PROJECT_BINARY_DIR}/rocprofv2 -i
        ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps/goldentraces/pmc.txt -d
        /tmp/tests-v2/pmc -o histo tests-v2/featuretests/profiler/apps/pmc_histogram
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    pmc_correctnes_histogram_test PROPERTIES LABELS "v2;rocprofv2" ENVIRONMENT
                                             "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

add_test(
    NAME pmc_correctnes_histogram_test_validation
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_histogram.py
            "/tmp/tests-v2/pmc/pmc_1/results_histo.csv"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    pmc_correctnes_histogram_test_validation
    PROPERTIES DEPENDS
               pmc_correctnes_histogram_test
               LABELS
               "v2;validation"
               PASS_REGULAR_EXPRESSION
               "Test Passed"
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               SKIP_REGULAR_EXPRESSION
               "Skipped")
