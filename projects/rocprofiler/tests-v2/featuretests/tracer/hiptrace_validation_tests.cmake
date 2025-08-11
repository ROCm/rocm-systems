# hip-trace validation test - Timestamp
add_test(
    NAME hiptrace_helloworld_test
    COMMAND ${PROJECT_BINARY_DIR}/rocprofv2 --hip-api -d ${PROJECT_BINARY_DIR}/out-trace
            -o out tests-v2/featuretests/profiler/apps/hip_helloworld
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    hiptrace_helloworld_test PROPERTIES LABELS "v2;rocprofv2" ENVIRONMENT
                                        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

add_test(
    NAME hiptrace_helloworld_test_validation
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/hip_trace_validate.py
            "out-trace/hip_api_trace_out.csv"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}")

set_tests_properties(
    hiptrace_helloworld_test_validation
    PROPERTIES DEPENDS
               hiptrace_helloworld_test
               LABELS
               "v2;validation"
               PASS_REGULAR_EXPRESSION
               "Test Passed"
               FAIL_REGULAR_EXPRESSION
               "Test Failed"
               SKIP_REGULAR_EXPRESSION
               "Skipped")
