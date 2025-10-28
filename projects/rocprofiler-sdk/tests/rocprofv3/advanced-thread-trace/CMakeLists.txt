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

set(COMMON_PARAMS_NO_LIB ${PRELOAD_ARGS} --att --output-format json --log-level env -d
                         ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace/)

set(IS_DISABLED ON)
if(attdecoder_FOUND)
    set(IS_DISABLED OFF)
    set(COMMON_PARAMS --att-library-path ${attdecoder_LIB_DIR} ${COMMON_PARAMS_NO_LIB})
endif()
if(ROCPROFILER_DISABLE_UNSTABLE_CTESTS)
    set(IS_DISABLED ON)
endif()

# hsa multiqueue dependency test with lib path
rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-hsa-multiqueue-cmd-env-att-lib-path
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS}/cmd_input
        --att-target-cu 1 --att-shader-engine-mask 0x11 --kernel-include-regex copyD
        --att-buffer-size 0x6000000 --att-simd-select 0x3 --att-serialize-all 1 -o out --
        $<TARGET_FILE:hsa_code_object_testapp>
    DEPENDS hsa_code_object_testapp
    LABELS "integration-tests"
    TIMEOUT 45
    DISABLED ${IS_DISABLED}
    FIXTURES_SETUP rocprofv3-test-att-hsa-multiqueue-cmd-env-att-lib-path)

# hsa multiqueue dependency test with json input
rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-hsa-multiqueue-json
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS}/json_input -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input.json --
        $<TARGET_FILE:hsa_code_object_testapp>
    DEPENDS hsa_code_object_testapp
    LABELS "integration-tests"
    TIMEOUT 45
    DISABLED ${IS_DISABLED}
    FIXTURES_SETUP rocprofv3-test-att-hsa-multiqueue-json)

# validate output
rocprofiler_add_integration_validate_test(
    rocprofv3-test-att-hsa-multiqueue-cmd
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests"
    DISABLED ${IS_DISABLED}
    FIXTURES_REQUIRED rocprofv3-test-att-hsa-multiqueue-cmd-env-att-lib-path
    ARGS --input
         ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/cmd_input/out_results.json
         --code-object-input
         ${CMAKE_CURRENT_BINARY_DIR}
         --output-path
         ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/cmd_input
    FAIL_REGULAR_EXPRESSION "AssertionError")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-att-hsa-multiqueue-json
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-att-hsa-multiqueue-json
    DISABLED ${IS_DISABLED}
    ARGS --input
         ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/json_input/out_results.json
         --code-object-input
         ${CMAKE_CURRENT_BINARY_DIR}
         --output-path
         ${CMAKE_CURRENT_BINARY_DIR}/hsa_code_object_testapp-trace/json_input
    FAIL_REGULAR_EXPRESSION "AssertionError")

function(configure_att_input _FILENAME)
    set(LIB_PATH_LOC "${attdecoder_LIB_DIR}")
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}//${_FILENAME}.in
                   ${CMAKE_CURRENT_BINARY_DIR}/${_FILENAME} @ONLY)
endfunction()

configure_att_input(att_input.yml)
configure_att_input(att_input_will_fail.yml)

# test yaml input
rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-yaml-input
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS_NO_LIB}/yaml -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input.yml --echo --
        $<TARGET_FILE:hsa_code_object_testapp>
    DEPENDS hsa_code_object_testapp
    TIMEOUT 45
    LABELS "integration-tests"
    DISABLED ${IS_DISABLED})

# Invalid lib path has to fail
rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-yaml-input-will-fail
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/att_input_will_fail.yml --log-level env --echo --
        $<TARGET_FILE:hsa_code_object_testapp>
    DEPENDS hsa_code_object_testapp
    TIMEOUT 45
    LABELS "integration-tests"
    WILL_FAIL
    DISABLED ${IS_DISABLED})

rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-library-path-cmd-line-will-fail
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path .
            --log-level env --echo -- $<TARGET_FILE:hsa_code_object_testapp>
    DEPENDS hsa_code_object_testapp
    TIMEOUT 45
    LABELS "integration-tests"
    WILL_FAIL
    DISABLED ${IS_DISABLED})

rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-library-path-env-var-will-fail
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS_NO_LIB}/envfail
            --att-library-path . --echo -- $<TARGET_FILE:hsa_code_object_testapp>
    DEPENDS hsa_code_object_testapp
    TIMEOUT 45
    LABELS "integration-tests"
    WILL_FAIL
    DISABLED ${IS_DISABLED}
    ENVIRONMENT "ROCPROF_ATT_LIBRARY_PATH=${attdecoder_LIB_DIR}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-env-var
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS_NO_LIB}/envvar --
            $<TARGET_FILE:hsa_code_object_testapp>
    DEPENDS hsa_code_object_testapp
    TIMEOUT 45
    LABELS "integration-tests"
    DISABLED ${IS_DISABLED}
    ENVIRONMENT "ROCPROF_ATT_LIBRARY_PATH=${attdecoder_LIB_DIR}")
#
# Uses ATT and Counter Collection at the same time
#
rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-hsa-multiqueue-plus-pmc
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS}/cmd_input --pmc
            SQ_WAVES -o out -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    DISABLED ${IS_DISABLED})

# Check for conflict PMC + activity
rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-hsa-multiqueue-activity-pmc-will-fail
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS}/cmd_input --pmc
            SQ_WAVES -o out --att-activity 8 -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    WILL_FAIL
    DISABLED ${IS_DISABLED})

# Check for conflict Perfcounters + activity
rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-hsa-multiqueue-activity-perf-will-fail
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS}/cmd_input
            --att-perfcounter-ctrl 8 -o out --att-activity 8 -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    WILL_FAIL
    DISABLED ${IS_DISABLED})

# ATT plus PC sampling

rocprofiler_sdk_pc_sampling_stochastic_disabled(IS_PC_SAMPLING_STOCHASTIC_DISABLED)

set(ATT_PLUS_PCS_DISABLE ${IS_DISABLED})
if(${IS_PC_SAMPLING_STOCHASTIC_DISABLED})
    set(ATT_PLUS_PCS_DISABLE ON)
endif()
if(${ROCPROFILER_DISABLE_UNSTABLE_CTESTS})
    set(ATT_PLUS_PCS_DISABLE ON)
endif()
set(PCS_ARGS --pc-sampling-beta-enabled 1 --pc-sampling-unit cycles
             --pc-sampling-interval 16384 --pc-sampling-method stochastic)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-plus-pc-sampling
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS}/cmd_input
            ${PCS_ARGS} -o out -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 90
    LABELS "integration-tests;thread-trace;pc-sampling"
    DISABLED ${ATT_PLUS_PCS_DISABLE})

# Trace two GPUs
rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-gpu-index-two-gpus
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS}/cmd_input -o out
            --att-gpu-index 0,1 -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    DISABLED ${IS_DISABLED})

# No machine has gpu index = 9999 The FAIL regex for this test is the ROCP_ERROR specific
# to invalid device index
rocprofiler_add_integration_execute_test(
    rocprofv3-test-att-gpu-index-will-fail
    COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> ${COMMON_PARAMS}/cmd_input -o out
            --att-gpu-index 0,9999 -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    WILL_FAIL
    DISABLED ${IS_DISABLED}
    FAIL_REGULAR_EXPRESSION "Invalid GPU Device Index")
