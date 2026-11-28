#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-roctracer-roctx-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Convert the rocpd database to pftrace and otf2 via the rocpd python module
find_package(Python3 REQUIRED)

set(roctracer-roctx-rocpd-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/roctracer-roctx-input.yml.in
               ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-input.yml @ONLY)

if(TARGET transpose-roctracer-roctx)
    set(TRANSPOSE_ROCTRACER_ROCTX_EXE $<TARGET_FILE:transpose-roctracer-roctx>)
    set(TRANSPOSE_ROCTRACER_ROCTX_DISABLED OFF)
else()
    set(TRANSPOSE_ROCTRACER_ROCTX_EXE)
    set(TRANSPOSE_ROCTRACER_ROCTX_DISABLED ON)
endif()

rocprofiler_add_integration_execute_test(
    rocprofv3-test-roctracer-roctx-trace
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-input.yml --
        ${TRANSPOSE_ROCTRACER_ROCTX_EXE}
    DEPENDS transpose-roctracer-roctx
    TIMEOUT 45
    WSL_TIMEOUT_SCALE
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED "${TRANSPOSE_ROCTRACER_ROCTX_DISABLED}"
    FIXTURES_SETUP rocprofv3-test-roctracer-roctx-trace)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-roctracer-roctx-trace-rocpd-convert
    COMMAND
        ${Python3_EXECUTABLE} -m rocpd convert -f pftrace otf2 -d
        ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-trace -o out -i
        ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-trace/out_results.db
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${roctracer-roctx-rocpd-env}"
    FIXTURES_REQUIRED rocprofv3-test-roctracer-roctx-trace
    FIXTURES_SETUP rocprofv3-test-roctracer-roctx-trace-rocpd-convert)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-roctracer-roctx-trace
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-trace/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-trace/out_results.pftrace
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/roctracer-roctx-trace/out_results.otf2
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-roctracer-roctx-trace
                      rocprofv3-test-roctracer-roctx-trace-rocpd-convert
    DISABLED "${TRANSPOSE_ROCTRACER_ROCTX_DISABLED}")
