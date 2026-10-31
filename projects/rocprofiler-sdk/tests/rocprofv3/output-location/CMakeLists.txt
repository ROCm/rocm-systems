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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#
# rocprofv3 tool tests for the output location
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-output-location
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/input-output-location.yml.in
               ${CMAKE_CURRENT_BINARY_DIR}/input-output-location.yml @ONLY)

##########################################################################################
#
#   Output location
#
##########################################################################################

# rocprofv3 hands the output location it settled on to the application through
# ROCPROF_OUTPUT_PATH and ROCPROF_OUTPUT_FILE_NAME, so the environment of the application
# shows which source won. No tracing service is requested, so the tests do not depend on
# GPU activity.

# An input file names the output location for the run, so it takes precedence over a
# ROCPROF_OUTPUT_* variable that was only inherited from the caller's environment.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-output-location-input-file
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-output-location.yml -- bash -c
        "test \"$ROCPROF_OUTPUT_PATH\" = \"${CMAKE_CURRENT_BINARY_DIR}/input_dir\" && test \"$ROCPROF_OUTPUT_FILE_NAME\" = input_name && echo resolved-from-input-file"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT
        "ROCPROF_OUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/environment_dir;ROCPROF_OUTPUT_FILE_NAME=environment_name"
    PASS_REGULAR_EXPRESSION "resolved-from-input-file")

# Without an input file, the inherited variables are the only request, and they still
# decide where the output goes.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-output-location-inherited-environment
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -- bash -c
        "test \"$ROCPROF_OUTPUT_PATH\" = \"${CMAKE_CURRENT_BINARY_DIR}/environment_dir\" && test \"$ROCPROF_OUTPUT_FILE_NAME\" = environment_name && echo resolved-from-environment"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT
        "ROCPROF_OUTPUT_PATH=${CMAKE_CURRENT_BINARY_DIR}/environment_dir;ROCPROF_OUTPUT_FILE_NAME=environment_name"
    PASS_REGULAR_EXPRESSION "resolved-from-environment")

# A command line option is an explicit request for this run, so it conflicts with an input
# file that names a different location rather than silently overriding it.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-output-location-command-line-conflict
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/command_line_dir -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-output-location.yml -- bash -c "exit 0"
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "Option 'output_directory' has conflicting values")
