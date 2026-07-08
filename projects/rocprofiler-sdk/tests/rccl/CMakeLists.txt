#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rccl-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rccl)

set(UNSUPPORTED_GFX OFF)
# disable when system has a vega gpu
if("gfx906" IN_LIST rocprofiler-sdk-tests-gfx-info)
    set(UNSUPPORTED_GFX ON)
endif()

set(rccl-tracing-env
    "ROCPROFILER_TOOL_OUTPUT_FILE=rccl-tracing-test.json"
    "ROCPROFILER_TOOL_CONTEXTS_EXCLUDE=HSA_API_CALLBACK:HSA_API_BUFFERED"
    "ROCPROFILER_DISABLE_PERFETTO_ANNOTATIONS=1"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

rocprofiler_add_integration_execute_test(
    test-rccl-tracing-single-process
    COMMAND $<TARGET_FILE:rccl-single-process> 64
    DEPENDS rccl-single-process
    TIMEOUT 240
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT "${rccl-tracing-env}"
    DISABLED $<OR:$<NOT:${rccl_FOUND}>,$<BOOL:${UNSUPPORTED_GFX}>>)

rocprofiler_add_integration_validate_test(
    test-rccl-tracing-single-process
    TEST_PATHS validate-single-process.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 240
    LABELS "integration-tests"
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/rccl-tracing-test.json
    ATTACHED_FILES_ON_FAIL ${CMAKE_CURRENT_BINARY_DIR}/rccl-tracing-test.json
    DISABLED $<OR:$<NOT:${rccl_FOUND}>,$<BOOL:${UNSUPPORTED_GFX}>>)
