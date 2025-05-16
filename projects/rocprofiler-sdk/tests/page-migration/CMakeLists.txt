#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-page-migration
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK_PRELOAD_ENV)
    set(PRELOAD_ENV
        "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}:$<TARGET_FILE:rocprofiler-sdk-json-tool>")
else()
    set(PRELOAD_ENV "LD_PRELOAD=$<TARGET_FILE:rocprofiler-sdk-json-tool>")
endif()

list(GET rocprofiler-sdk-tests-gfx-info 0 page-migration-gpu-0-gfx-info)

# disabled on all architectures
if("${page-migration-gpu-0-gfx-info}" MATCHES "^gfx([0-9a-fA-F]+)$")
    set(IS_DISABLED ON)
endif()

add_test(NAME test-page-migration-execute COMMAND $<TARGET_FILE:page-migration> 4 1024)

set(page-migration-env
    "${PRELOAD_ENV}"
    "ROCPROFILER_DISABLE_PERFETTO_ANNOTATIONS=1"
    "ROCPROFILER_TOOL_OUTPUT_FILE=page-migration-test.json"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set_tests_properties(
    test-page-migration-execute
    PROPERTIES TIMEOUT
               60
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${page-migration-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "KFD does not support SVM event reporting"
               WORKING_DIRECTORY
               ${CMAKE_CURRENT_BINARY_DIR}
               DISABLED
               "${IS_DISABLED}")

# copy to binary directory
rocprofiler_configure_pytest_files(COPY validate.py conftest.py CONFIG pytest.ini)

add_test(NAME test-page-migration-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/page-migration-test.json)

set_tests_properties(
    test-page-migration-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               test-page-migration-execute
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               SKIP_REGULAR_EXPRESSION
               "KFD does not support SVM event reporting"
               WORKING_DIRECTORY
               ${CMAKE_CURRENT_BINARY_DIR}
               DISABLED
               "${IS_DISABLED}")
