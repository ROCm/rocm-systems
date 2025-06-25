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

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate-single-process.py conftest.py
                                   CONFIG pytest.ini)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

set(UNSUPPORTED_GFX OFF)
# disable when system has a vega gpu
if("gfx906" IN_LIST rocprofiler-sdk-tests-gfx-info)
    set(UNSUPPORTED_GFX ON)
endif()

find_program(BASH_EXE "bash" REQUIRED)

set(rccl-tracing-env
    "${PRELOAD_ENV}"
    "ROCPROFILER_TOOL_OUTPUT_FILE=rccl-tracing-test.json"
    "ROCPROFILER_TOOL_CONTEXTS_EXCLUDE=HSA_API_CALLBACK:HSA_API_BUFFERED"
    "ROCPROFILER_DISABLE_PERFETTO_ANNOTATIONS=1"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

rocprofiler_add_test(
    NAME test-rccl-tracing-single-process-execute
    TARGET rccl-single-process
    ARGS 64
    TIMEOUT 240
    LABELS "integration-tests"
    ENVIRONMENT "${rccl-tracing-env}"
    FAIL_REGULAR_EXPRESSION ${ROCPROFILER_DEFAULT_FAIL_REGEX}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DISABLED $<OR:$<NOT:${rccl_FOUND}>,$<BOOL:${UNSUPPORTED_GFX}>>)

add_test(
    NAME test-rccl-tracing-single-process-validate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate-single-process.py
            --input ${CMAKE_CURRENT_BINARY_DIR}/rccl-tracing-test.json)

set_tests_properties(
    test-rccl-tracing-single-process-validate
    PROPERTIES TIMEOUT
               240
               LABELS
               "integration-tests"
               DEPENDS
               test-rccl-tracing-single-process-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               WORKING_DIRECTORY
               ${CMAKE_CURRENT_BINARY_DIR}
               ATTACHED_FILES_ON_FAIL
               ${CMAKE_CURRENT_BINARY_DIR}/rccl-tracing-test.json
               DISABLED
               $<OR:$<NOT:${rccl_FOUND}>,$<BOOL:${UNSUPPORTED_GFX}>>)
