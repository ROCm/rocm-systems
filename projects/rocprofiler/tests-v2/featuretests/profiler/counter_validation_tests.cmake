# counter validation test - GRBM_COUNT
add_test(
    NAME grbm_count_helloworld_test
    COMMAND
        ${PROJECT_BINARY_DIR}/rocprofv2 -i
        ${PROJECT_BINARY_DIR}/tests-v2/featuretests/profiler/apps/input.txt -d
        ${PROJECT_BINARY_DIR}/out-grbm_count -o grbm
        tests-v2/featuretests/profiler/apps/hip_helloworld
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    grbm_count_helloworld_test PROPERTIES LABELS "v2;rocprofv2" ENVIRONMENT
                                          "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

add_test(
    NAME grbm_count_helloworld_test_validation
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/grbm_validate.py
            "out-grbm_count/pmc_1/results_grbm.csv"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    grbm_count_helloworld_test_validation
    PROPERTIES DEPENDS
               grbm_count_helloworld_test
               LABELS
               "v2;validation"
               PASS_REGULAR_EXPRESSION
               "Test Passed"
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               SKIP_REGULAR_EXPRESSION
               "Skipped")
