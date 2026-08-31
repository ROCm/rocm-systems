# MIT License
#
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# rocprofv3 CLI multi-pass counter collection test
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-cli-multipass
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(
    CONFIG pytest.ini COPY validate.py validate_no_output.py conftest.py input.json
                           input.yml input-combined.json input-pmc-groups.json)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# CLI multi-pass test using multiple --pmc flags (one group per pass)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-cli-multipass
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --pmc GRBM_COUNT --pmc
        GRBM_GUI_ACTIVE -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc -o out -f csv --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-cli-multipass)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-cli-multipass
    TEST_PATHS validate.py
    CONFIG pytest.ini
    ARGS --output-dir ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc --expected-counters
         SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE
    TIMEOUT 45
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-cli-multipass)

# multi-pass test defining the pmc groups in a JSON input file
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-cli-multipass-json
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc-json -o out -f csv --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-cli-multipass-json)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-cli-multipass-json
    TEST_PATHS validate.py
    CONFIG pytest.ini
    ARGS --output-dir ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc-json
         --expected-counters SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE
    TIMEOUT 45
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-cli-multipass-json)

# multi-pass test defining the pmc groups in a YAML input file
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-cli-multipass-yaml
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.yml
        -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc-yaml -o out -f csv --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-cli-multipass-yaml)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-cli-multipass-yaml
    TEST_PATHS validate.py
    CONFIG pytest.ini
    ARGS --output-dir ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc-yaml
         --expected-counters SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE
    TIMEOUT 45
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-cli-multipass-yaml)

# combined multi-pass: two CLI --pmc passes (first) plus one input-file pass (last)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-cli-multipass-combined
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-combined.json --pmc SQ_WAVES --pmc GRBM_COUNT -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc-combined -o out -f csv --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-counter-collection-cli-multipass-combined)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-cli-multipass-combined
    TEST_PATHS validate.py
    CONFIG pytest.ini
    ARGS --output-dir ${CMAKE_CURRENT_BINARY_DIR}/simple-transpose-cc-combined
         --expected-counters SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE
    TIMEOUT 45
    FIXTURES_REQUIRED rocprofv3-test-counter-collection-cli-multipass-combined)

# Negative tests: illegal combinations must fail before launch. PASS_REGULAR_EXPRESSION
# strictly asserts the message (WILL_FAIL would accept any non-zero exit).

# multiple --pmc with --pid
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-cli-multipass-rejects-pid
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --pmc GRBM_COUNT --pid
        12345 -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc-err-pid -o out -f csv --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "multiple --pmc flags\\) is not compatible with attach mode")

# multiple --pmc with --collection-period
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-cli-multipass-rejects-collection-period
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES --pmc GRBM_COUNT
        --collection-period 0:100:1 -d ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc-err-cp -o out
        -f csv -- $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION
        "multiple --pmc flags\\) is not compatible with --collection-period")

# CLI --pmc plus input-file pmc_groups
rocprofiler_add_integration_execute_test(
    rocprofv3-test-counter-collection-cli-multipass-rejects-pmc-groups
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-pmc-groups.json --pmc SQ_WAVES -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-cc-err-groups -o out -f csv --
        $<TARGET_FILE:simple-transpose>
    DEPENDS simple-transpose
    TIMEOUT 45
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "Cannot specify both --pmc and \\(input file\\) pmc_groups")

# A rejected run must fail before the tool library loads, so none of the negative tests
# above may leave an output directory behind.
rocprofiler_add_integration_validate_test(
    rocprofv3-test-counter-collection-cli-multipass-no-output
    TEST_PATHS validate_no_output.py
    CONFIG pytest.ini
    ARGS --binary-dir
         ${CMAKE_CURRENT_BINARY_DIR}
         --present-globs
         input.json
         pytest.ini
         --absent-globs
         "*-cc-err-pid"
         "*-cc-err-cp"
         "*-cc-err-groups"
    TIMEOUT 45
    FIXTURES_REQUIRED
        rocprofv3-test-counter-collection-cli-multipass-rejects-pid
        rocprofv3-test-counter-collection-cli-multipass-rejects-collection-period
        rocprofv3-test-counter-collection-cli-multipass-rejects-pmc-groups)
