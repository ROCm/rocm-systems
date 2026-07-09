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
    rocprofiler-sdk-tests-counter-collection-multiplex
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# GPU-free unit test: YAML and JSON multiplex layouts parse identically and preserve group
# order. Needs no GPU, so it gates CI.
rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-input-parsing
    TEST_PATHS test_multiplex_input.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --rocprofv3 $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
    TARGET_DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests")

# JSON input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o
        out_json -f csv -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml input_3group.json input_3group.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input.json
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex)

# YAML input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiple-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.yml
        -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o out_yaml -f csv --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiple-yaml)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiple-yaml
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml input_3group.json input_3group.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_yaml_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_yaml_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input.yml
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiple-yaml)

# 3-group layout: reproducible-dispatch-count issues 9 deterministic dispatches so all
# three groups are scheduled; interval 2 exercises the (dispatch-index / interval) group
# derivation and wrap-around.

# JSON 3-group input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex-3group
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_3group.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-json-cc -o out_3group_json -f csv --
        $<TARGET_FILE:reproducible-dispatch-count> 9 1 100 1
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex-3group)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-3group
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml input_3group.json input_3group.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-json-cc/out_3group_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-json-cc/out_3group_json_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input_3group.json
         --max-value-ratio
         2.0
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex-3group)

# YAML 3-group input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex-3group-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_3group.yml -d
        ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-yaml-cc -o out_3group_yaml -f csv --
        $<TARGET_FILE:reproducible-dispatch-count> 9 1 100 1
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex-3group-yaml)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-3group-yaml
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml input_3group.json input_3group.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-yaml-cc/out_3group_yaml_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-3group-yaml-cc/out_3group_yaml_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input_3group.yml
         --max-value-ratio
         2.0
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex-3group-yaml)

# interval > 1: interval 4 with 12 dispatches -> groupA[1-4], groupB[5-8], groupA[9-12]
# (stay-on-a-group-then-switch, plus wrap-around).

# JSON interval-4 input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex-interval4
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_interval4.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/multiplex-interval4-json-cc -o out_interval4_json -f
        csv -- $<TARGET_FILE:reproducible-dispatch-count> 12 1 100 1
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex-interval4)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-interval4
    TEST_PATHS validate.py
    COPY conftest.py input_interval4.json input_interval4.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-interval4-json-cc/out_interval4_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-interval4-json-cc/out_interval4_json_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input_interval4.json
         --max-value-ratio
         2.0
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex-interval4)

# YAML interval-4 input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex-interval4-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_interval4.yml -d
        ${CMAKE_CURRENT_BINARY_DIR}/multiplex-interval4-yaml-cc -o out_interval4_yaml -f
        csv -- $<TARGET_FILE:reproducible-dispatch-count> 12 1 100 1
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex-interval4-yaml)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-interval4-yaml
    TEST_PATHS validate.py
    COPY conftest.py input_interval4.json input_interval4.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-interval4-yaml-cc/out_interval4_yaml_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-interval4-yaml-cc/out_interval4_yaml_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input_interval4.yml
         --max-value-ratio
         2.0
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex-interval4-yaml)

# Run-level JSON == YAML equivalence: re-uses the two 2-group execute fixtures and asserts
# they collect the same counters per dispatch (no extra GPU run).
rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-json-yaml-equiv
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_counter_collection.csv
         --counter-input-b
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_yaml_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input.json
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED
        "rocprofv3-test-counter-collection-multiplex;rocprofv3-test-counter-collection-multiple-yaml"
    )

# Oversubscription: 12 SQ counters exceed a single hardware pass, so they are split across
# two 6-counter groups that multiplexing collects across dispatches. The integer-only
# workload leaves some SQ counters at 0, hence --allow-zero-counter-values.
#
# The negative case (all 12 in one group) is not a ctest: rocprofv3 aborts (SIGABRT) on it
# instead of exiting cleanly, which would hang the test.
#
# The per-pass counter limit is architecture-specific (verified on gfx942), so this is
# gated to the gfx9/CDNA family and disabled elsewhere.
set(_MULTIPLEX_OVERSUB_DISABLED ON)
if(rocprofiler-sdk-tests-gfx-info)
    list(GET rocprofiler-sdk-tests-gfx-info 0 _multiplex_gpu0_gfx)
    if("${_multiplex_gpu0_gfx}" MATCHES "^gfx9")
        set(_MULTIPLEX_OVERSUB_DISABLED OFF)
    endif()
endif()

rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex-oversub
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_oversub.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/multiplex-oversub-json-cc -o out_oversub_json -f csv
        -- $<TARGET_FILE:reproducible-dispatch-count> 8 1 100 1
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    DISABLED ${_MULTIPLEX_OVERSUB_DISABLED}
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex-oversub)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-oversub
    TEST_PATHS validate.py
    COPY conftest.py input_oversub.json
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-oversub-json-cc/out_oversub_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-oversub-json-cc/out_oversub_json_counter_collection.csv
         --multiplex-input
         ${CMAKE_CURRENT_BINARY_DIR}/input_oversub.json
         --allow-zero-counter-values
    TIMEOUT 120
    LABELS "integration-tests"
    DISABLED ${_MULTIPLEX_OVERSUB_DISABLED}
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex-oversub)

# Graceful degradation: a layout with an unresolvable (unknown counter) group and an empty
# group must drop only those groups and still collect the valid one. --present-counters is
# the exact surviving set. No --multiplex-input, so the layout test skips here.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex-degraded
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input_badcounter.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/multiplex-degraded-json-cc -o out_degraded_json -f csv
        -- $<TARGET_FILE:reproducible-dispatch-count> 4 1 100 1
    DEPENDS reproducible-dispatch-count
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex-degraded)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex-degraded
    TEST_PATHS validate.py
    COPY conftest.py input_badcounter.json
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-degraded-json-cc/out_degraded_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiplex-degraded-json-cc/out_degraded_json_counter_collection.csv
         --present-counters
         "SQ_WAVES,GRBM_COUNT"
    TIMEOUT 120
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex-degraded)
