# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

set(CMAKE_BUILD_TYPE "Release")
find_package(Threads REQUIRED)

add_library(tests-compile-options INTERFACE)
target_compile_options(tests-compile-options INTERFACE -g)

add_executable(thread-limit thread-limit.cpp)
target_compile_definitions(thread-limit PRIVATE MAX_THREADS=${ROCPROFSYS_MAX_THREADS})
target_link_libraries(thread-limit PRIVATE Threads::Threads tests-compile-options)

set(_thread_limit_environment
    "${_base_environment}"
    "ROCPROFSYS_TRACE=ON"
    "ROCPROFSYS_PROFILE=ON"
    "ROCPROFSYS_COUT_OUTPUT=ON"
    "ROCPROFSYS_USE_SAMPLING=ON"
    "ROCPROFSYS_SAMPLING_FREQ=250"
    "ROCPROFSYS_VERBOSE=2"
    "ROCPROFSYS_TIMEMORY_COMPONENTS=wall_clock,peak_rss,page_rss"
)

math(EXPR THREAD_LIMIT_TEST_VALUE "${ROCPROFSYS_MAX_THREADS} + 24")
math(EXPR THREAD_LIMIT_TEST_VALUE_PLUS_ONE "${THREAD_LIMIT_TEST_VALUE} + 1")

set(_thread_limit_pass_regex "\\|${THREAD_LIMIT_TEST_VALUE}>>>")
set(_thread_limit_fail_regex
    "\\|${THREAD_LIMIT_TEST_VALUE_PLUS_ONE}>>>|ROCPROFSYS_ABORT_FAIL_REGEX"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE
    NAME thread-limit
    TARGET thread-limit
    LABELS "max-threads"
    REWRITE_ARGS -e -v 2 -i 1024 --label return args
    RUNTIME_ARGS -e -v 1 -i 1024 --label return args
    RUN_ARGS 35 2 ${THREAD_LIMIT_TEST_VALUE}
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    RUNTIME_PASS_REGEX "${_thread_limit_pass_regex}"
    SAMPLING_PASS_REGEX "${_thread_limit_pass_regex}"
    REWRITE_RUN_PASS_REGEX "${_thread_limit_pass_regex}"
    RUNTIME_FAIL_REGEX "${_thread_limit_fail_regex}"
    SAMPLING_FAIL_REGEX "${_thread_limit_fail_regex}"
    REWRITE_RUN_FAIL_REGEX "${_thread_limit_fail_regex}"
    ENVIRONMENT "${_thread_limit_environment}"
)
