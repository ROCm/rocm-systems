# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# rocprofv3 --collector: the same HIP API trace as in-process rocprofv3 --hip-trace, with
# nothing loaded into the application. Each case runs the application twice, once on each
# path, and validate.py compares the outputs with the parity rules: every in-process call
# is recorded, on the same thread and in the same order, with the same arguments (pointers
# excepted: the runs are separate); extra calls only before the first or after the last
# in-process call of a thread (calls before rocprofv3 wraps the HIP tables, and teardown).
#
# The collector runs are skipped where the collector's prerequisites are missing: a
# rocprofiler-sdk built without it, no LTTng session daemon, or a HIP runtime without its
# trace probes.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-collector
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(collector-skip-regex
    "--collector \\(ROCPROF_COLLECTOR\\): |rocprofv3-collector not found|no HIP API calls were recorded"
    )
set(collector-formats csv json rocpd)

# rocprofv3_collector_parity_test(<name> DEPENDS <targets> COMMAND <application...>)
function(rocprofv3_collector_parity_test _name)
    cmake_parse_arguments(ARG "" "" "DEPENDS;COMMAND" ${ARGN})
    set(_inprocess ${CMAKE_CURRENT_BINARY_DIR}/${_name}-in-process)
    set(_collector ${CMAKE_CURRENT_BINARY_DIR}/${_name}-collector)

    rocprofiler_add_integration_execute_test(
        rocprofv3-test-collector-${_name}-in-process
        COMMAND $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hip-trace -d ${_inprocess} -o
                %pid% --output-format ${collector-formats} -- ${ARG_COMMAND}
        DEPENDS ${ARG_DEPENDS}
        TIMEOUT 120
        LABELS "integration-tests;collector"
        FIXTURES_SETUP rocprofv3-test-collector-${_name})

    rocprofiler_add_integration_execute_test(
        rocprofv3-test-collector-${_name}-collector
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --collector --hip-trace -d
            ${_collector} -o %pid% --output-format ${collector-formats} -- ${ARG_COMMAND}
        DEPENDS ${ARG_DEPENDS}
        TIMEOUT 120
        LABELS "integration-tests;collector"
        SKIP_REGULAR_EXPRESSION "${collector-skip-regex}"
        FIXTURES_SETUP rocprofv3-test-collector-${_name})

    rocprofiler_add_integration_validate_test(
        rocprofv3-test-collector-${_name}
        TEST_PATHS validate.py
        COPY conftest.py
        CONFIG pytest.ini
        ARGS --in-process-dir ${_inprocess} --collector-dir ${_collector}
        TIMEOUT 120
        LABELS "integration-tests;collector"
        FIXTURES_REQUIRED rocprofv3-test-collector-${_name})
endfunction()

# Several threads, each with its own HIP calls.
rocprofv3_collector_parity_test(transpose-threads DEPENDS transpose COMMAND
                                $<TARGET_FILE:transpose> 4 50)

# One thread, several streams.
rocprofv3_collector_parity_test(vector-ops-streams DEPENDS vector-ops COMMAND
                                $<TARGET_FILE:vector-ops> 4 1)

# A process tree: a shell that starts two traced processes; one set of files each.
rocprofv3_collector_parity_test(
    process-tree DEPENDS transpose COMMAND sh -c
    "$<TARGET_FILE:transpose> 2 20 & $<TARGET_FILE:transpose> 2 20; wait")
