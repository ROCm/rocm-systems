# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-att
    LANGUAGES CXX
    VERSION 0.0.0)

string(REPLACE "LD_PRELOAD=" "--preload;" PRELOAD_ARGS
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          att_input.json)

find_package(rocprofiler-sdk REQUIRED)

set(IS_DISABLED ON)

set(LIB_PATH_LOC "${ROCM_PATH}/lib")
set(LIB_PATH_ENV "ROCPROF_ATT_LIBRARY_PATH=${LIB_PATH_LOC}")

find_library(
    attdecoder
    HINTS ${LIB_PATH_LOC}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib
    NAMES att_decoder_trace)

if(attdecoder)
    set(IS_DISABLED OFF)
endif()

# hsa multiqueue dependency test with lib path
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-cmd-env-att-lib-path-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --advanced-thread-trace
        1 --att-target-cu 1 --att-shader-engine-mask 0x11 --kernel-include-regex copyD
        --att-buffer-size 0x6000000 --att-simd-select 0x3 --att-serialize-all 1 -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/cmd_input -o out --output-format json
        ${PRELOAD_ARGS} --att-library-path ${LIB_PATH_LOC} --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-cmd-env-att-lib-path-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DISABLED ${IS_DISABLED})

# hsa multiqueue dependency test with json input
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --att-library-path
        ${LIB_PATH_LOC} -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/json_input -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input.json ${PRELOAD_ARGS} --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-json-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DISABLED ${IS_DISABLED} ENVIRONMENT
               ${LIB_PATH_ENV})

# validate output
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
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "rocprofv3-test-hsa-multiqueue-att-cmd-ld-lib-path-execute"
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               ${IS_DISABLED})

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-json-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "rocprofv3-test-hsa-multiqueue-att-json-execute"
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               ${IS_DISABLED})

function(configure_att_input _FILENAME _OUTDIR)
    set(LIBRARY_OUTPUT_DIR ${_OUTDIR})
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/att_input.yml.in
                   ${CMAKE_CURRENT_BINARY_DIR}/${_FILENAME} @ONLY)
endfunction()

configure_att_input(att_input.yml "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
configure_att_input(att_input_will_fail.yml "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")

# test yaml input
add_test(
    NAME rocprofv3-test-att-yaml-input
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input.yml --log-level env --echo --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-att-yaml-input PROPERTIES TIMEOUT 45 LABELS "integration-tests"
                                             DISABLED ${IS_DISABLED})

# Invalid lib path has to fail
add_test(
    NAME rocprofv3-test-att-yaml-input-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input_will_fail.yml --log-level env --echo --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-att-yaml-input-will-fail
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" WILL_FAIL ON DISABLED ${IS_DISABLED})

add_test(NAME rocprofv3-test-att-library-path-cmd-line-will-fail
         COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path .
                 --log-level env --echo -- $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-att-library-path-cmd-line-will-fail
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" WILL_FAIL ON DISABLED ${IS_DISABLED})

add_test(NAME rocprofv3-test-att-library-path-env-var-will-fail
         COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --log-level env --echo
                 -- $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-att-library-path-env-var-will-fail
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "ROCPROF_ATT_LIBRARY_PATH=."
               WILL_FAIL
               ON
               DISABLED
               ${IS_DISABLED})

#
# Uses ATT and Counter Collection at the same time
#
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-plus-pmc-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --pmc SQ_WAVES
        --advanced-thread-trace -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/cmd_input -o
        out --output-format json ${PRELOAD_ARGS} -- $<TARGET_FILE:vector-ops>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-plus-pmc-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DISABLED ${IS_DISABLED} ENVIRONMENT
               ${LIB_PATH_ENV})
