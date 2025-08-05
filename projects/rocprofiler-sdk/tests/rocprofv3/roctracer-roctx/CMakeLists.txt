#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-roctracer-roctx-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(tracing-env "${PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG pytest.ini roctracer-roctx-input.yml
                                   COPY validate.py conftest.py)

if(TARGET transpose-roctracer-roctx)
    set(TRANSPOSE_ROCTRACER_ROCTX_EXE $<TARGET_FILE:transpose-roctracer-roctx>)
    set(TRANSPOSE_ROCTRACER_ROCTX_DISABLED OFF)
else()
    set(TRANSPOSE_ROCTRACER_ROCTX_EXE)
    set(TRANSPOSE_ROCTRACER_ROCTX_DISABLED ON)
endif()

add_test(NAME rocprofv3-test-roctracer-roctx-trace-execute
         COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i roctracer-roctx-input.yml
                 -- ${TRANSPOSE_ROCTRACER_ROCTX_EXE})

set_tests_properties(
    rocprofv3-test-roctracer-roctx-trace-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${tracing-env}"
               DISABLED "${TRANSPOSE_ROCTRACER_ROCTX_DISABLED}")

add_test(
    NAME rocprofv3-test-roctracer-roctx-trace-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-trace/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-trace/out_results.pftrace
        --otf2-input ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-trace/out_results.otf2)

set_tests_properties(
    rocprofv3-test-roctracer-roctx-trace-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "rocprofv3-test-roctracer-roctx-trace-execute"
               DISABLED
               "${TRANSPOSE_ROCTRACER_ROCTX_DISABLED}"
               FAIL_REGULAR_EXPRESSION
               "AssertionError")
