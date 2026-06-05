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

# Counter collection on this app can hit a rare, transient GPU memory fault in the SDK
# (~<0.1% on a healthy GPU). It is non-deterministic and re-running succeeds, so the
# execute step is run through this tiny retry wrapper: it re-runs the command up to
# CC_MAX_ATTEMPTS times until it succeeds. A genuine regression still fails (every attempt
# fails). Combined with ROCPROF_SIGNAL_HANDLERS=0 on the tests below, a faulting attempt
# aborts fast and clean (no GPU core-dump hang/wedge), so the retry runs on a healthy GPU.
# CC_CMD is passed as a ';'-separated list of the command + arguments.
file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/cc_retry.cmake
    [==[
if(NOT DEFINED CC_MAX_ATTEMPTS)
    set(CC_MAX_ATTEMPTS 3)
endif()
set(_attempt 1)
while(_attempt LESS_EQUAL ${CC_MAX_ATTEMPTS})
    execute_process(COMMAND ${CC_CMD} RESULT_VARIABLE _rc)
    if(_rc EQUAL 0)
        return()
    endif()
    message(STATUS "cc_retry: attempt ${_attempt}/${CC_MAX_ATTEMPTS} failed (rc=${_rc}); retrying")
    math(EXPR _attempt "${_attempt} + 1")
endwhile()
message(FATAL_ERROR "cc_retry: command failed after ${CC_MAX_ATTEMPTS} attempts")
]==])

# JSON input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiplex
    COMMAND
        ${CMAKE_COMMAND} -DCC_MAX_ATTEMPTS=3
        "-DCC_CMD=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>$<SEMICOLON>-i$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/input.json$<SEMICOLON>-d$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc$<SEMICOLON>-o$<SEMICOLON>out_json$<SEMICOLON>-f$<SEMICOLON>csv$<SEMICOLON>--$<SEMICOLON>$<TARGET_FILE:simple-transpose>"
        -P ${CMAKE_CURRENT_BINARY_DIR}/cc_retry.cmake
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    # Disable rocprofv3's signal handler so the rare GPU fault (see cc_retry.cmake note)
    # aborts fast instead of attempting a GPU core dump that hangs until the ctest timeout
    # and leaves the GPU wedged. This keeps the retry attempts (and the rest of the suite)
    # running on a healthy GPU.
    ENVIRONMENT "ROCPROF_SIGNAL_HANDLERS=0"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiplex)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiplex
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_json_counter_collection.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiplex
    UNSTABLE)

# YAML input test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-multiple-yaml
    COMMAND
        ${CMAKE_COMMAND} -DCC_MAX_ATTEMPTS=3
        "-DCC_CMD=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>$<SEMICOLON>-i$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/input.yml$<SEMICOLON>-d$<SEMICOLON>${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc$<SEMICOLON>-o$<SEMICOLON>out_yaml$<SEMICOLON>-f$<SEMICOLON>csv$<SEMICOLON>--$<SEMICOLON>$<TARGET_FILE:simple-transpose>"
        -P ${CMAKE_CURRENT_BINARY_DIR}/cc_retry.cmake
    DEPENDS simple-transpose
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    # See note above: disable the signal handler so a faulting attempt aborts fast (no
    # hanging GPU core dump / wedge) and the retry can run on a healthy GPU.
    ENVIRONMENT "ROCPROF_SIGNAL_HANDLERS=0"
    FIXTURES_SETUP rocprofv3-test-counter-collection-multiple-yaml)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-multiple-yaml
    TEST_PATHS validate.py
    COPY conftest.py input.json input.yml
    CONFIG pytest.ini
    ARGS --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_yaml_agent_info.csv
         --counter-input
         ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc/out_yaml_counter_collection.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-multiple-yaml
    UNSTABLE)
