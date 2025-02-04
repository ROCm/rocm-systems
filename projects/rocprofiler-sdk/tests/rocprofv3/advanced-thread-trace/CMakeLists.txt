#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-att
    LANGUAGES CXX
    VERSION 0.0.0)

string(REPLACE "LD_PRELOAD=" "--preload;" PRELOAD_ARGS
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          att_input.json)

find_package(rocprofiler-sdk REQUIRED)

# hsa multiqueue dependency test

add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-cmd-env-ld-lib-path-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --advanced-thread-trace 1
        --att-target-cu 1 --att-shader-engine-mask 0x11 --kernel-include-regex copyD
        --att-buffer-size 0x6000000 --att-simd-select 0x3 --att-parse testing
        --att-serialize-all 1 -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/cmd_input -o
        out --output-format json ${PRELOAD_ARGS} --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-cmd-env-ld-lib-path-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}
               FAIL_REGULAR_EXPRESSION "HSA_API|HIP_API")
# hsa multiqueue dependency test
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-cmd-env-att-lib-path-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --advanced-thread-trace 1
        --att-target-cu 1 --att-shader-engine-mask 0x11 --kernel-include-regex copyD
        --att-buffer-size 0x6000000 --att-simd-select 0x3 --att-parse testing
        --att-serialize-all 1 -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/cmd_input -o
        out --output-format json ${PRELOAD_ARGS} --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-cmd-env-att-lib-path-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               ATT_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib FAIL_REGULAR_EXPRESSION
               "HSA_API|HIP_API")

# hsa multiqueue dependency test
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att-library-path
        ${CMAKE_BINARY_DIR}/lib -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/json_input -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input.json ${PRELOAD_ARGS} --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-json-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" FAIL_REGULAR_EXPRESSION
               "HSA_API|HIP_API")

add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-cmd-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
        ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/cmd_input/out_results.json
        --code-object-input ${CMAKE_CURRENT_BINARY_DIR} --output-path
        ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/cmd_input)

add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-json-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
        ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/json_input/out_results.json
        --code-object-input ${CMAKE_CURRENT_BINARY_DIR} --output-path
        ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/json_input)

set(MULTIQUEUE_CMD_VALIDATION_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/cmd_input/out_results.json)

set(MULTIQUEUE_JSON_VALIDATION_FILES
    ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/json_input/pass_1/out_results.json
    )

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-cmd-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-hsa-multiqueue-att-cmd-ld-lib-path-execute"
               FAIL_REGULAR_EXPRESSION "AssertionError")

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-json-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-hsa-multiqueue-att-json-execute" FAIL_REGULAR_EXPRESSION
               "AssertionError")
