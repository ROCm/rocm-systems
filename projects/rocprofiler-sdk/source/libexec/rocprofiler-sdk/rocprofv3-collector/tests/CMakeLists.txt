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

# rocprofv3-collector and rocprofv3 --collector, without a GPU: the collector's test
# program stands in for the HIP runtime (its "hip" mode writes rocm_hip events with the
# test probe schema).

if(NOT TARGET collector-test-app)
    return()
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(_collector_tests_dir ${PROJECT_SOURCE_DIR}/source/lib/rocprofiler-sdk-collector/tests)

add_test(
    NAME rocprofv3-collector-writes-outputs
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check_outputs.py
        $<TARGET_FILE:rocprofv3-collector> $<TARGET_FILE:collector-test-app>
        ${_collector_tests_dir}/test_schema ${CMAKE_CURRENT_BINARY_DIR}/outputs)
set_tests_properties(
    rocprofv3-collector-writes-outputs
    PROPERTIES
        SKIP_RETURN_CODE
        77
        LABELS
        "unittests"
        TIMEOUT
        120
        ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:$ENV{LD_LIBRARY_PATH}"
    )

# Services the collector path does not provide are errors before the application starts.
set(_rocprofv3 ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/rocprofv3)

function(rocprofv3_collector_rejects _name _expected)
    add_test(NAME rocprofv3-collector-rejects-${_name}
             COMMAND ${Python3_EXECUTABLE} ${_rocprofv3} ${ARGN} -- true)
    set_tests_properties(
        rocprofv3-collector-rejects-${_name}
        PROPERTIES PASS_REGULAR_EXPRESSION "${_expected}" LABELS "unittests" TIMEOUT 60)
endfunction()

rocprofv3_collector_rejects(kernel-trace "--kernel-trace not available with --collector"
                            --collector --kernel-trace)
rocprofv3_collector_rejects(
    explicit-hip-graph "--hip-graph-trace not available with --collector" --collector
    --hip-trace --hip-graph-trace)
rocprofv3_collector_rejects(sys-trace "--kernel-trace.*not available with --collector"
                            --collector --sys-trace)
rocprofv3_collector_rejects(hsa-trace "--hsa-core-trace.*not available with --collector"
                            --collector --hsa-trace)
rocprofv3_collector_rejects(nothing-to-trace "needs --hip-trace" --collector)

# The environment variable selects the path too.
add_test(NAME rocprofv3-collector-env-selects-path
         COMMAND ${Python3_EXECUTABLE} ${_rocprofv3} --kernel-trace -- true)
set_tests_properties(
    rocprofv3-collector-env-selects-path
    PROPERTIES PASS_REGULAR_EXPRESSION "not available with --collector" ENVIRONMENT
               "ROCPROF_COLLECTOR=1" LABELS "unittests" TIMEOUT 60)

# Before launch, rocprofv3 --collector checks for the LTTng session daemon and the HIP
# runtime's probe schema. A fake lttng-tools prefix stands in for the daemon.
set(_fake_lttng ${CMAKE_CURRENT_BINARY_DIR}/fake-lttng)
file(
    WRITE ${_fake_lttng}/bin/lttng-sessiond
    "#!/bin/sh\n# stands in for lttng-sessiond in the rocprofv3 --collector tests\nexit 1\n"
    )
file(
    CHMOD
    ${_fake_lttng}/bin/lttng-sessiond
    PERMISSIONS
    OWNER_READ
    OWNER_WRITE
    OWNER_EXECUTE
    GROUP_READ
    GROUP_EXECUTE
    WORLD_READ
    WORLD_EXECUTE)
set(_no_dir ${CMAKE_CURRENT_BINARY_DIR}/does-not-exist)

function(rocprofv3_collector_prelaunch _name _expected)
    cmake_parse_arguments(ARG "" "" "ENVIRONMENT" ${ARGN})
    add_test(NAME rocprofv3-collector-prelaunch-${_name}
             COMMAND ${Python3_EXECUTABLE} ${_rocprofv3} --collector --hip-trace
                     ${ARG_UNPARSED_ARGUMENTS} -- true)
    # Only the variables given: neither is inherited from the build environment.
    set(_modifications "ROCPROFILER_COLLECTOR_LTTNG_PREFIX=unset:"
                       "ROCPROFILER_COLLECTOR_SCHEMA_DIR=unset:")
    foreach(_var ${ARG_ENVIRONMENT})
        list(APPEND _modifications "${_var}")
    endforeach()
    # NAME=value -> NAME=set:value (the unset: entries above contain a colon already)
    list(TRANSFORM _modifications REPLACE "^([A-Z_]+)=([^:]*)$" "\\1=set:\\2")
    set_tests_properties(
        rocprofv3-collector-prelaunch-${_name}
        PROPERTIES PASS_REGULAR_EXPRESSION "${_expected}" ENVIRONMENT_MODIFICATION
                   "${_modifications}" LABELS "unittests" TIMEOUT 60)
endfunction()

rocprofv3_collector_prelaunch(
    no-lttng-in-prefix "lttng-sessiond not found in ROCPROFILER_COLLECTOR_LTTNG_PREFIX"
    ENVIRONMENT "ROCPROFILER_COLLECTOR_LTTNG_PREFIX=${_no_dir}"
    "ROCPROFILER_COLLECTOR_SCHEMA_DIR=${_collector_tests_dir}/test_schema")
rocprofv3_collector_prelaunch(
    no-lttng "lttng-sessiond not found: neither in this ROCm installation" ENVIRONMENT
    "PATH=${_no_dir}"
    "ROCPROFILER_COLLECTOR_SCHEMA_DIR=${_collector_tests_dir}/test_schema")
rocprofv3_collector_prelaunch(
    no-schema "probe schema .*rocm_hip.json. was not found.*HIP_ENABLE_ROCM_TRACE"
    ENVIRONMENT "ROCPROFILER_COLLECTOR_LTTNG_PREFIX=${_fake_lttng}"
    "ROCPROFILER_COLLECTOR_SCHEMA_DIR=${_no_dir}")
rocprofv3_collector_prelaunch(
    both-missing "lttng-sessiond not found.*; the HIP runtime's probe schema" ENVIRONMENT
    "ROCPROFILER_COLLECTOR_LTTNG_PREFIX=${_no_dir}"
    "ROCPROFILER_COLLECTOR_SCHEMA_DIR=${_no_dir}")
# Both found: rocprofv3 runs the application under rocprofv3-collector.
rocprofv3_collector_prelaunch(
    ready "command: .*rocprofv3-collector', '--', 'true'" --echo ENVIRONMENT
    "ROCPROFILER_COLLECTOR_LTTNG_PREFIX=${_fake_lttng}"
    "ROCPROFILER_COLLECTOR_SCHEMA_DIR=${_collector_tests_dir}/test_schema")
