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
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --advanced-thread-trace
        1 --att-target-cu 1 --att-shader-engine-mask 0x11 --kernel-include-regex copyD
        --att-buffer-size 0x6000000 --att-simd-select 0x3 --att-parse testing
        --att-serialize-all 1 -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/cmd_input -o
        out --output-format json ${PRELOAD_ARGS} --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-cmd-env-ld-lib-path-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               LD_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:$ENV{LD_LIBRARY_PATH})
# hsa multiqueue dependency test
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-cmd-env-att-lib-path-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --advanced-thread-trace
        1 --att-target-cu 1 --att-shader-engine-mask 0x11 --kernel-include-regex copyD
        --att-buffer-size 0x6000000 --att-simd-select 0x3 --att-parse testing
        --att-serialize-all 1 -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/cmd_input -o
        out --output-format json ${PRELOAD_ARGS} --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-cmd-env-att-lib-path-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               ROCPROF_ATT_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# hsa multiqueue dependency test
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --att-library-path
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY} -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/json_input -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input.json ${PRELOAD_ARGS} --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(rocprofv3-test-hsa-multiqueue-att-json-execute
                     PROPERTIES TIMEOUT 45 LABELS "integration-tests")

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

if(TARGET rocprofiler-sdk::att-decoder-testing AND TARGET
                                                   rocprofiler-sdk::att-decoder-summary)
    set(MISSING_TEST_DECODER_LIBS OFF)
else()
    set(MISSING_TEST_DECODER_LIBS ON)
endif()

function(configure_att_input _FILENAME _OUTDIR)
    set(LIBRARY_OUTPUT_DIR ${_OUTDIR})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/att_input.yml.in
                   ${CMAKE_CURRENT_BINARY_DIR}/${_FILENAME} @ONLY)
endfunction()

configure_att_input(att_input.yml "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
configure_att_input(att_input_will_fail.yml "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

#
# Uses att_library_path in YAML input to specify ATT library path
#
add_test(
    NAME rocprofv3-test-att-library-path-yaml-input
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
            ${CMAKE_CURRENT_BINARY_DIR}/att_input.yml --log-level env --echo -- sleep 0)

set_tests_properties(
    rocprofv3-test-att-library-path-yaml-input
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DISABLED
               "${MISSING_TEST_DECODER_LIBS}")

add_test(
    NAME rocprofv3-test-att-library-path-yaml-input-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input_will_fail.yml --log-level env --echo --
        sleep 0)

set_tests_properties(
    rocprofv3-test-att-library-path-yaml-input-will-fail
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" WILL_FAIL ON DISABLED
               "${MISSING_TEST_DECODER_LIBS}")

#
# Uses --att-library-path to specify ATT library path
#
add_test(
    NAME rocprofv3-test-att-library-path-cmd-line
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/att --att-parse summary --log-level env --echo
        -- sleep 0)

set_tests_properties(
    rocprofv3-test-att-library-path-cmd-line
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DISABLED
               "${MISSING_TEST_DECODER_LIBS}")

add_test(
    NAME rocprofv3-test-att-library-path-cmd-line-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/att --att-parse testing --log-level env --echo
        -- sleep 0)

set_tests_properties(
    rocprofv3-test-att-library-path-cmd-line-will-fail
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" WILL_FAIL ON DISABLED
               "${MISSING_TEST_DECODER_LIBS}")

#
# Uses ROCPROF_ATT_LIBRARY_PATH to specify ATT library path
#
add_test(NAME rocprofv3-test-att-library-path-env-var
         COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-parse summary
                 --log-level env --echo -- sleep 0)

set_tests_properties(
    rocprofv3-test-att-library-path-env-var
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               "ROCPROF_ATT_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/att" DISABLED
               "${MISSING_TEST_DECODER_LIBS}")

add_test(NAME rocprofv3-test-att-library-path-env-var-will-fail
         COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-parse testing
                 --log-level env --echo -- sleep 0)

set_tests_properties(
    rocprofv3-test-att-library-path-env-var-will-fail
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "ROCPROF_ATT_LIBRARY_PATH=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/att"
               WILL_FAIL
               ON
               DISABLED
               "${MISSING_TEST_DECODER_LIBS}")
