# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# rocprofv3 counter collection option coverage
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-extended-options
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(_output_dir "${CMAKE_CURRENT_BINARY_DIR}/output")

set(_openmp_disabled OFF)
set(_openmp_memcheck_types "ThreadSanitizer" "AddressSanitizer"
                           "UndefinedBehaviorSanitizer")
if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST _openmp_memcheck_types)
    set(_openmp_disabled ON)
endif()
list(GET rocprofiler-sdk-tests-gfx-info 0 _openmp_gpu_0)
if("${_openmp_gpu_0}" MATCHES "^gfx(10|11|12)[0-9][0-9]$")
    set(_openmp_disabled ON)
endif()

set(_naming_fixtures)
foreach(_mode IN ITEMS reference truncated mangled)
    set(_option)
    if(_mode STREQUAL "truncated")
        set(_option -T)
    elseif(_mode STREQUAL "mangled")
        set(_option -M)
    endif()

    set(_fixture "rocprofv3-test-counter-collection-naming-${_mode}")
    list(APPEND _naming_fixtures ${_fixture})
    rocprofiler_add_integration_execute_test(
        ${_fixture}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES ${_option} -d
            ${_output_dir}/naming/${_mode} -o out --output-format csv --
            $<TARGET_FILE:vector-ops>
        DEPENDS vector-ops
        PRELOAD "${PRELOAD_ENV}"
        TIMEOUT 60
        LABELS "counter-collection"
        FIXTURES_SETUP ${_fixture})
endforeach()

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-naming
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    DISCOVERY_ARGS -k test_kernel_name_modes
    ARGS --reference-input ${_output_dir}/naming/reference/out_counter_collection.csv
         --truncated-input ${_output_dir}/naming/truncated/out_counter_collection.csv
         --mangled-input ${_output_dir}/naming/mangled/out_counter_collection.csv
    FIXTURES_REQUIRED ${_naming_fixtures}
    TIMEOUT 60
    LABELS "counter-collection")

foreach(_mode IN ITEMS absolute relative type-relative)
    set(_fixture "rocprofv3-test-counter-collection-agent-index-${_mode}")
    rocprofiler_add_integration_execute_test(
        ${_fixture}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --agent-index
            ${_mode} -d ${_output_dir}/agent-index/${_mode} -o out --output-format csv
            json -- $<TARGET_FILE:vector-ops>
        DEPENDS vector-ops
        PRELOAD "${PRELOAD_ENV}"
        TIMEOUT 60
        LABELS "counter-collection"
        FIXTURES_SETUP ${_fixture})

    rocprofiler_add_integration_validate_test(
        ${_fixture}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        DISCOVERY_ARGS -k test_agent_index_mode
        ARGS --counter-input
             ${_output_dir}/agent-index/${_mode}/out_counter_collection.csv --json-input
             ${_output_dir}/agent-index/${_mode}/out_results.json --agent-index ${_mode}
        FIXTURES_REQUIRED ${_fixture}
        TIMEOUT 60
        LABELS "counter-collection")
endforeach()

if(TARGET reproducible-dispatch-count)
    set(_fixture rocprofv3-test-counter-collection-execution-contexts)
    rocprofiler_add_integration_execute_test(
        ${_fixture}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --group-by-queue
            --kernel-rename --marker-trace -d ${_output_dir}/execution-contexts -o out
            --output-format csv json pftrace -- $<TARGET_FILE:reproducible-dispatch-count>
            200 2
        DEPENDS reproducible-dispatch-count
        PRELOAD "${PRELOAD_ENV}"
        TIMEOUT 120
        LABELS "counter-collection"
        FIXTURES_SETUP ${_fixture})

    rocprofiler_add_integration_validate_test(
        ${_fixture}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        DISCOVERY_ARGS -k test_execution_contexts
        ARGS --context-input ${_output_dir}/execution-contexts/out_counter_collection.csv
             --context-json ${_output_dir}/execution-contexts/out_results.json
             --context-pftrace ${_output_dir}/execution-contexts/out_results.pftrace
        FIXTURES_REQUIRED ${_fixture}
        TIMEOUT 60
        LABELS "counter-collection")
endif()

foreach(_mode IN ITEMS stats summary)
    if(_mode STREQUAL "stats")
        set(_options --stats)
    else()
        set(_options --summary --summary-per-domain)
    endif()

    set(_fixture "rocprofv3-test-counter-collection-${_mode}")
    rocprofiler_add_integration_execute_test(
        ${_fixture}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES ${_options} -d
            ${_output_dir}/${_mode} -o out --output-format csv --
            $<TARGET_FILE:vector-ops>
        DEPENDS vector-ops
        PRELOAD "${PRELOAD_ENV}"
        TIMEOUT 60
        LABELS "counter-collection"
        FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}" "COUNTER_COLLECTION"
        FIXTURES_SETUP ${_fixture})

    rocprofiler_add_integration_validate_test(
        ${_fixture}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        DISCOVERY_ARGS -k test_counter_collection_with_post_processing
        ARGS --post-input ${_output_dir}/${_mode}/out_counter_collection.csv
        FIXTURES_REQUIRED ${_fixture}
        TIMEOUT 60
        LABELS "counter-collection")
endforeach()

set(_fixture rocprofv3-test-counter-collection-device-qualifier)
rocprofiler_add_integration_execute_test(
    ${_fixture}
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES GRBM_GUI_ACTIVE:device=0
        --agent-index relative -d ${_output_dir}/device-qualifier -o out --output-format
        csv json -- $<TARGET_FILE:vector-ops>
    DEPENDS vector-ops
    PRELOAD "${PRELOAD_ENV}"
    TIMEOUT 60
    LABELS "counter-collection"
    FIXTURES_SETUP ${_fixture})

rocprofiler_add_integration_validate_test(
    ${_fixture}
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    DISCOVERY_ARGS -k test_device_qualifier_restricts_counter
    ARGS --device-input ${_output_dir}/device-qualifier/out_counter_collection.csv
         --json-input ${_output_dir}/device-qualifier/out_results.json
    FIXTURES_REQUIRED ${_fixture}
    TIMEOUT 60
    LABELS "counter-collection")

if(TARGET openmp-target)
    set(_fixture rocprofv3-test-counter-collection-openmp)
    rocprofiler_add_integration_execute_test(
        ${_fixture}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES -d
            ${_output_dir}/openmp -o out --output-format csv --
            $<TARGET_FILE:openmp-target>
        DEPENDS openmp-target
        PRELOAD "${PRELOAD_ENV}"
        TIMEOUT 60
        LABELS "counter-collection"
        DISABLED "${_openmp_disabled}"
        FIXTURES_SETUP ${_fixture})

    rocprofiler_add_integration_validate_test(
        ${_fixture}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        DISCOVERY_ARGS -k test_openmp_counter_collection
        ARGS --openmp-input ${_output_dir}/openmp/out_counter_collection.csv
        FIXTURES_REQUIRED ${_fixture}
        TIMEOUT 60
        LABELS "counter-collection"
        DISABLED "${_openmp_disabled}")
endif()
