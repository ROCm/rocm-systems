#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-rccl-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rccl)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(UNSUPPORTED_GFX OFF)
# disable when system has a vega gpu
if("gfx906" IN_LIST rocprofiler-sdk-tests-gfx-info)
    set(UNSUPPORTED_GFX ON)
endif()

set(TEST_NAME_SINGLE_PROCESS "rccl-tracing-single-process")

find_program(BASH_EXE "bash" REQUIRED)

foreach(_OUTPUT_FORMAT csv json pftrace)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-${TEST_NAME_SINGLE_PROCESS}-${_OUTPUT_FORMAT}
        COMMAND
            ${BASH_EXE} -c
            "rm -fr ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME_SINGLE_PROCESS}/*.${_OUTPUT_FORMAT} &&\
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --rccl-trace -o ${TEST_NAME_SINGLE_PROCESS}-%pid% \
            --output-format ${_OUTPUT_FORMAT} -d ${TEST_NAME_SINGLE_PROCESS} --log-level env -- $<TARGET_FILE:rccl-single-process> 64"
        DEPENDS rccl-single-process
        TIMEOUT 120
        LABELS "integration-tests"
        PRELOAD "${PRELOAD_ENV}"
        FAIL_REGULAR_EXPRESSION "threw an exception"
        DISABLED $<OR:$<NOT:${rccl_FOUND}>,$<BOOL:${UNSUPPORTED_GFX}>>)
endforeach()

rocprofiler_add_integration_validate_test(
    rocprofv3-test-${TEST_NAME_SINGLE_PROCESS}
    TEST_PATHS validate-single-process.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED
        rocprofv3-test-${TEST_NAME_SINGLE_PROCESS}-csv
        rocprofv3-test-${TEST_NAME_SINGLE_PROCESS}-json
        rocprofv3-test-${TEST_NAME_SINGLE_PROCESS}-pftrace
    ARGS --json-input-dir "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME_SINGLE_PROCESS}"
         --csv-input-dir "${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME_SINGLE_PROCESS}"
    FAIL_REGULAR_EXPRESSION "AssertionError"
    DISABLED $<OR:$<NOT:${rccl_FOUND}>,$<BOOL:${UNSUPPORTED_GFX}>>)
