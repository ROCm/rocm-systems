#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-scratch-memory-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(scratch-memory-tracing-env "${PRELOAD_ENV}")

set(VALIDATION_DEPENDS)
foreach(_OUTPUT_FORMAT csv json)
    add_test(
        NAME rocprofv3-test-scratch-memory-tracing-${_OUTPUT_FORMAT}-execute
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --scratch-memory-trace -d
            ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format
            ${_OUTPUT_FORMAT} --log-level env -- $<TARGET_FILE:scratch-memory>)

    set_tests_properties(
        rocprofv3-test-scratch-memory-tracing-${_OUTPUT_FORMAT}-execute
        PROPERTIES TIMEOUT
                   45
                   LABELS
                   "integration-tests"
                   ENVIRONMENT
                   "${scratch-memory-tracing-env}"
                   FAIL_REGULAR_EXPRESSION
                   "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
                   DISABLED
                   "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}"
                   FIXTURES_SETUP
                   rocprofv3-scratch-memory-tracing)

    list(APPEND VALIDATION_DEPENDS
         rocprofv3-test-scratch-memory-tracing-${_OUTPUT_FORMAT}-execute)
endforeach()

add_test(
    NAME rocprofv3-test-scratch-memory-tracing-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/scratch-memory-trace/out_results.json --csv-input
        ${CMAKE_CURRENT_BINARY_DIR}/scratch-memory-trace/out_scratch_memory_trace.csv)

set_tests_properties(
    rocprofv3-test-scratch-memory-tracing-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "${VALIDATION_DEPENDS}"
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}"
               FIXTURES_REQUIRED
               rocprofv3-scratch-memory-tracing)
