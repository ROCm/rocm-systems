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
# Integration tests: rocprofv3 coexisting with a second OMPT tool.
#
# The OpenMP runtime hosts exactly one OMPT tool. rocprofv3 takes the role when it was
# asked to collect OMPT and declines it otherwise, so a user profiling with rocprofv3 can
# still run their own OMPT tool (TAU, Score-P) by leaving --ompt-trace off. Every case
# here runs the same workload against the same stand-in tool, built in
# tests/bin/openmp/openmp-tools-coexist, and differs only in how that tool reaches the
# runtime and what the rocprofv3 command line asks for:
#
# In the keep and defer cases the tool is offered through OMP_TOOL_LIBRARIES, which the
# runtime consults only once no loaded library has claimed the role. rocprofv3 is asked
# for OMPT, or is not, and the role follows.
#
# In the defer-back cases the tool is loaded ahead of rocprofv3 and claims the role, then
# hands it back through the public rocprofiler_ompt_start_tool. That bypasses rocprofv3's
# own entry point, so the command line has to still decide there.
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-ompt-coexist
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer"
                               "UndefinedBehaviorSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

# disable when GPU-0 is navi2, navi3, or navi4 (no OpenMP target offload support)
list(GET rocprofiler-sdk-tests-gfx-info 0 ompt-coexist-gpu-0-gfx-info)
if("${ompt-coexist-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$")
    set(IS_DISABLED ON)
endif()

# the workload and the stand-in tool are both required
if(NOT TARGET openmp-target OR NOT TARGET openmp-tools-coexist-mock)
    set(IS_DISABLED ON)
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(ompt-coexist-base-env
    "OMP_NUM_THREADS=2"
    "OMP_TARGET_OFFLOAD=mandatory"
    "OMP_DISPLAY_ENV=1"
    "ROCR_VISIBLE_DEVICES=0"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

# OMP_TOOL_LIBRARIES is the runtime's own tool search, consulted only once no already
# loaded library claims the role. Offering the stand-in tool through it is what makes the
# keep and defer cases differ in outcome rather than in setup.
set(ompt-coexist-env ${ompt-coexist-base-env}
                     "OMP_TOOL_LIBRARIES=$<TARGET_FILE:openmp-tools-coexist-mock>")

# ---- keep: --ompt-trace, so rocprofv3 holds the OMPT tool role -----------------------
set(ompt-coexist-keep-output-dir ${CMAKE_CURRENT_BINARY_DIR}/ompt-coexist-keep)
set(ompt-coexist-keep-mock-output ${ompt-coexist-keep-output-dir}/mock-summary.json)
# the stand-in tool writes its summary from inside the workload, so the directory has to
# exist before rocprofv3 creates its own output there
file(MAKE_DIRECTORY ${ompt-coexist-keep-output-dir})
set(ompt-coexist-keep-env ${ompt-coexist-env}
                          "COEXIST_MOCK_OUTPUT=${ompt-coexist-keep-mock-output}")
set(ompt-coexist-keep-validate-k
    "test_rocprofv3_kept_ompt_role or test_other_tool_did_not_get_role or test_kernel_dispatch_tracing_intact"
    )

rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-coexist-keep
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace --kernel-trace -d
        ${ompt-coexist-keep-output-dir} -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-target>,$<TARGET_FILE:openmp-target>,openmp-target>
    DEPENDS openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target;ompt-coexist"
    ENVIRONMENT "${ompt-coexist-keep-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-coexist-keep
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-coexist-keep
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS -k "${ompt-coexist-keep-validate-k}" --rocpd-input
         ${ompt-coexist-keep-output-dir}/out_results.db --mock-input
         ${ompt-coexist-keep-mock-output}
    DISCOVERY_ARGS -k "${ompt-coexist-keep-validate-k}"
    TIMEOUT 45
    LABELS "integration-tests;openmp-target;ompt-coexist"
    FIXTURES_REQUIRED rocprofv3-test-ompt-coexist-keep
    DISABLED "${IS_DISABLED}")

# ---- defer: no OMPT requested, so the stand-in tool gets the role --------------------
set(ompt-coexist-defer-output-dir ${CMAKE_CURRENT_BINARY_DIR}/ompt-coexist-defer)
set(ompt-coexist-defer-mock-output ${ompt-coexist-defer-output-dir}/mock-summary.json)
file(MAKE_DIRECTORY ${ompt-coexist-defer-output-dir})
set(ompt-coexist-defer-env ${ompt-coexist-env}
                           "COEXIST_MOCK_OUTPUT=${ompt-coexist-defer-mock-output}")
set(ompt-coexist-defer-validate-k
    "test_rocprofv3_deferred_ompt_role or test_other_tool_got_role or test_kernel_dispatch_tracing_intact"
    )

rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-coexist-defer
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace -d
        ${ompt-coexist-defer-output-dir} -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-target>,$<TARGET_FILE:openmp-target>,openmp-target>
    DEPENDS openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target;ompt-coexist"
    ENVIRONMENT "${ompt-coexist-defer-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-coexist-defer
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-coexist-defer
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS -k "${ompt-coexist-defer-validate-k}" --rocpd-input
         ${ompt-coexist-defer-output-dir}/out_results.db --mock-input
         ${ompt-coexist-defer-mock-output}
    DISCOVERY_ARGS -k "${ompt-coexist-defer-validate-k}"
    TIMEOUT 45
    LABELS "integration-tests;openmp-target;ompt-coexist"
    FIXTURES_REQUIRED rocprofv3-test-ompt-coexist-defer
    DISABLED "${IS_DISABLED}")

# ---- defer-back: the other tool is bound first and hands the role back ---------------
# ROCPROF_PRELOAD is prepended to LD_PRELOAD by rocprofv3, ahead of its own libraries, so
# the runtime binds the stand-in tool rather than rocprofv3. The tool then calls
# rocprofiler_ompt_start_tool directly, reaching the SDK without passing through
# rocprofv3's own entry point. OMP_TOOL_LIBRARIES is deliberately absent here: the tool is
# already loaded and claims the role, so the runtime never reaches its own search. (These
# tests are disabled under sanitizers, so this does not contend with the sanitizer
# ROCPROF_PRELOAD built above.)
set(ompt-coexist-deferback-env ${ompt-coexist-base-env} "COEXIST_MOCK_MODE=defer"
                               "ROCPROF_PRELOAD=$<TARGET_FILE:openmp-tools-coexist-mock>")

# asked for OMPT: accepting the hand-back must collect it
set(ompt-coexist-deferback-output-dir ${CMAKE_CURRENT_BINARY_DIR}/ompt-coexist-deferback)
set(ompt-coexist-deferback-mock-output
    ${ompt-coexist-deferback-output-dir}/mock-summary.json)
file(MAKE_DIRECTORY ${ompt-coexist-deferback-output-dir})
set(ompt-coexist-deferback-asked-env
    ${ompt-coexist-deferback-env}
    "COEXIST_MOCK_OUTPUT=${ompt-coexist-deferback-mock-output}")
set(ompt-coexist-deferback-validate-k
    "test_deferback_rocprofv3_collected_ompt or test_deferback_other_tool_forwarded or test_kernel_dispatch_tracing_intact"
    )

rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-coexist-deferback
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace --kernel-trace -d
        ${ompt-coexist-deferback-output-dir} -o out --output-format rocpd --log-level env
        -- $<IF:$<TARGET_EXISTS:openmp-target>,$<TARGET_FILE:openmp-target>,openmp-target>
    DEPENDS openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target;ompt-coexist"
    ENVIRONMENT "${ompt-coexist-deferback-asked-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-coexist-deferback
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-coexist-deferback
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS -k "${ompt-coexist-deferback-validate-k}" --rocpd-input
         ${ompt-coexist-deferback-output-dir}/out_results.db --mock-input
         ${ompt-coexist-deferback-mock-output}
    DISCOVERY_ARGS -k "${ompt-coexist-deferback-validate-k}"
    TIMEOUT 45
    LABELS "integration-tests;openmp-target;ompt-coexist"
    FIXTURES_REQUIRED rocprofv3-test-ompt-coexist-deferback
    DISABLED "${IS_DISABLED}")

# not asked for OMPT: the hand-back must be declined rather than silently honored
set(ompt-coexist-deferback-unasked-output-dir
    ${CMAKE_CURRENT_BINARY_DIR}/ompt-coexist-deferback-unasked)
set(ompt-coexist-deferback-unasked-mock-output
    ${ompt-coexist-deferback-unasked-output-dir}/mock-summary.json)
file(MAKE_DIRECTORY ${ompt-coexist-deferback-unasked-output-dir})
set(ompt-coexist-deferback-unasked-env
    ${ompt-coexist-deferback-env}
    "COEXIST_MOCK_OUTPUT=${ompt-coexist-deferback-unasked-mock-output}")
set(ompt-coexist-deferback-unasked-validate-k
    "test_deferback_without_request_collects_no_ompt or test_deferback_other_tool_forwarded or test_kernel_dispatch_tracing_intact"
    )

rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-coexist-deferback-unasked
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-trace -d
        ${ompt-coexist-deferback-unasked-output-dir} -o out --output-format rocpd
        --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-target>,$<TARGET_FILE:openmp-target>,openmp-target>
    DEPENDS openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target;ompt-coexist"
    ENVIRONMENT "${ompt-coexist-deferback-unasked-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-coexist-deferback-unasked
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-coexist-deferback-unasked
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS -k "${ompt-coexist-deferback-unasked-validate-k}" --rocpd-input
         ${ompt-coexist-deferback-unasked-output-dir}/out_results.db --mock-input
         ${ompt-coexist-deferback-unasked-mock-output}
    DISCOVERY_ARGS -k "${ompt-coexist-deferback-unasked-validate-k}"
    TIMEOUT 45
    LABELS "integration-tests;openmp-target;ompt-coexist"
    FIXTURES_REQUIRED rocprofv3-test-ompt-coexist-deferback-unasked
    DISABLED "${IS_DISABLED}")
