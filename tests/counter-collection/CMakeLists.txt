#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

set(counter-collection-env
    "${PRELOAD_ENV}" "ROCPROFILER_TOOL_OUTPUT_FILE=counter-collection-test.json"
    "ROCPROFILER_TOOL_CONTEXTS=COUNTER_COLLECTION" "ROCPROF_COUNTERS=SQ_WAVES_sum")

add_test(NAME test-counter-collection-execute COMMAND $<TARGET_FILE:multistream>)

set_tests_properties(
    test-counter-collection-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${counter-collection-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}"
               FIXTURES_SETUP
               test-counter-collection)

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py CONFIG pytest.ini)

add_test(NAME test-counter-collection-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/counter-collection-test.json)

set_tests_properties(
    test-counter-collection-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               test-counter-collection-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}"
               FIXTURES_REQUIRED
               test-counter-collection)
