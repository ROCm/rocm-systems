#
#
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-kernel-tracing
    LANGUAGES CXX HIP
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_library(kernel-flush-contract-client SHARED flush_client.cpp)
target_link_libraries(kernel-flush-contract-client
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk Threads::Threads)

set_source_files_properties(flush.cpp PROPERTIES LANGUAGE HIP)
add_executable(kernel-flush-contract flush.cpp)
target_link_libraries(kernel-flush-contract PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(kernel-flush-contract kernel-flush-contract-client)
configure_file(flush_runner.py ${CMAKE_CURRENT_BINARY_DIR}/flush_runner.py COPYONLY)

foreach(_MODE signal expanded signal-less)
    foreach(_ORDER stop-first flush-first)
        foreach(_CASE tail delayed unfinished reentry repeat concurrent)
            rocprofiler_add_integration_execute_test(
                kernel-flush-${_MODE}-${_ORDER}-${_CASE}
                COMMAND
                    ${Python3_EXECUTABLE} -B ${CMAKE_CURRENT_BINARY_DIR}/flush_runner.py
                    --app $<TARGET_FILE:kernel-flush-contract> --client
                    $<TARGET_FILE:kernel-flush-contract-client> --mode ${_MODE} --order
                    ${_ORDER} --case ${_CASE}
                DEPENDS kernel-flush-contract
                TIMEOUT 60
                LABELS "integration-tests;kernel-flush"
                PASS_REGULAR_EXPRESSION "\\[flush-contract\\] VERIFIED"
                FAIL_REGULAR_EXPRESSION
                    "\\[flush-contract\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
                SKIP_REGULAR_EXPRESSION "\\[flush-contract\\] SKIP")
        endforeach()
    endforeach()
endforeach()

set(kernel-tracing-env "ROCPROFILER_TOOL_DISABLE_PERFETTO=1"
                       "ROCPROFILER_TOOL_OUTPUT_FILE=kernel-tracing-test.json")

rocprofiler_add_integration_execute_test(
    kernel-tracing
    TARGET reproducible-runtime
    TIMEOUT 100
    LABELS "integration-tests"
    ENVIRONMENT "${kernel-tracing-env}"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    FIXTURES_SETUP kernel-tracing)

# Stable structural/logical kernel-tracing validation tests. test_total_runtime is
# excluded here and registered as its own (unstable) test below.
rocprofiler_add_integration_validate_test(
    kernel-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED kernel-tracing
    DISCOVERY_ARGS -k "not test_total_runtime"
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-test.json)

rocprofiler_add_integration_validate_test(
    kernel-tracing-total-runtime
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED kernel-tracing
    DISCOVERY_ARGS -k "test_total_runtime"
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-test.json
    DISABLED_CODECOV
    DISABLED_MEMCHECKS "(Address|Thread|UndefinedBehavior)Sanitizer"
    UNSTABLE)
