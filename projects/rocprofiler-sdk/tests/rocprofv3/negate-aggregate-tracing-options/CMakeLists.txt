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
# rocprofv3 tool tests for negating tracing options
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-negate-aggregate-tracing-options
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(negate-aggregate-tracing-options-env "${PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG input-negate-aggregate-tracing-options.yml)

##########################################################################################
#
#   Command line input
#
##########################################################################################

add_test(
    NAME rocprofv3-test-negate-aggregate-tracing-options-cmd-line
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -M --sys-trace -d
        ${CMAKE_CURRENT_BINARY_DIR} -o out --log-level info --hip-trace=no --hsa-trace=no
        --marker-trace=no --kernel-trace=no --memory-copy-trace=no
        --memory-allocation-trace=no --scratch-memory-trace=no --rccl-trace=no
        --kokkos-trace=no --rocdecode-trace=no --rocjpeg-trace=no --hip-runtime-trace=no
        --hip-compiler-trace=no --hsa-core-trace=no --hsa-amd-trace=no
        --hsa-image-trace=no --hsa-finalizer-trace=no --advanced-thread-trace=no
        --kernel-rename=no --summary --echo -- ${CMAKE_COMMAND})

set_tests_properties(
    rocprofv3-test-negate-aggregate-tracing-options-cmd-line
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${negate-aggregate-tracing-options-env}"
               PASS_REGULAR_EXPRESSION
               "command:.*${CMAKE_COMMAND}"
               WILL_FAIL
               ON)

##########################################################################################
#
#   YAML input
#
##########################################################################################

add_test(
    NAME rocprofv3-test-negate-aggregate-tracing-options-inp-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-negate-aggregate-tracing-options.yml --
        ${CMAKE_COMMAND})

set_tests_properties(
    rocprofv3-test-negate-aggregate-tracing-options-inp-yaml
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${negate-aggregate-tracing-options-env}"
               PASS_REGULAR_EXPRESSION
               "command:.*${CMAKE_COMMAND}"
               WILL_FAIL
               ON)
