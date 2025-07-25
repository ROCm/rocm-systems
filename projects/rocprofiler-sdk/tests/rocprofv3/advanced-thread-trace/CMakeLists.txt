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

include(FindPackageHandleStandardArgs)

project(
    rocprofiler-sdk-tests-rocprofv3-att
    LANGUAGES CXX
    VERSION 0.0.0)

set(CMAKE_MESSAGE_INDENT "[${PROJECT_NAME}] ")

string(REPLACE "LD_PRELOAD=" "--preload;" PRELOAD_ARGS
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          att_input.json)

find_package(rocprofiler-sdk REQUIRED)

set(IS_DISABLED ON)

find_library(
    attdecoder_LIBRARY
    NAMES rocprof-trace-decoder
    HINTS ${ROCM_PATH}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib)

if(attdecoder_LIBRARY)
    cmake_path(GET attdecoder_LIBRARY PARENT_PATH attdecoder_LIB_DIR)
endif()

find_package_handle_standard_args(attdecoder REQUIRED_VARS attdecoder_LIB_DIR
                                                           attdecoder_LIBRARY)

if(attdecoder_FOUND)
    set(IS_DISABLED OFF)
    set(LIB_PATH_ENV "ROCPROF_ATT_LIBRARY_PATH=${attdecoder_LIB_DIR}")
endif()

# hsa multiqueue dependency test with lib path
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-cmd-env-att-lib-path-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --advanced-thread-trace
        1 --att-target-cu 1 --att-shader-engine-mask 0x11 --kernel-include-regex copyD
        --att-buffer-size 0x6000000 --att-simd-select 0x3 --att-serialize-all 1 -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/cmd_input -o out --output-format json
        ${PRELOAD_ARGS} --att-library-path ${attdecoder_LIB_DIR} --att-activity 8 --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-cmd-env-att-lib-path-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DISABLED ${IS_DISABLED})

# hsa multiqueue dependency test with json input
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-json-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --att-library-path
        ${attdecoder_LIB_DIR} -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/json_input -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input.json ${PRELOAD_ARGS} --
        $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-json-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DISABLED ${IS_DISABLED} ENVIRONMENT
               "${LIB_PATH_ENV}")

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

configure_att_input(att_input.yml "${attdecoder_LIB_DIR}")
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
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" WILL_FAIL ON DISABLED True)

add_test(NAME rocprofv3-test-att-library-path-cmd-line-will-fail
         COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path .
                 --log-level env --echo -- $<TARGET_FILE:hsa_code_object_testapp>)

set_tests_properties(
    rocprofv3-test-att-library-path-cmd-line-will-fail
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" WILL_FAIL ON DISABLED True)

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
               True)

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
               "${LIB_PATH_ENV}")

# Check for conflict PMC + activity
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-activity-pmc-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --pmc SQ_WAVES
        --advanced-thread-trace -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/cmd_input -o
        out --att-activity 8 ${PRELOAD_ARGS} -- $<TARGET_FILE:vector-ops>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-activity-pmc-will-fail
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DISABLED
               ${IS_DISABLED}
               ENVIRONMENT
               "${LIB_PATH_ENV}"
               WILL_FAIL
               ON)

# Check for conflict Perfcounters + activity
add_test(
    NAME rocprofv3-test-hsa-multiqueue-att-activity-perf-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --log-level env --att
        --att-perfcounter-ctrl 8 -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/cmd_input -o
        out --att-activity 8 ${PRELOAD_ARGS} -- $<TARGET_FILE:vector-ops>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-att-activity-perf-will-fail
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DISABLED
               ${IS_DISABLED}
               ENVIRONMENT
               "${LIB_PATH_ENV}"
               WILL_FAIL
               ON)
