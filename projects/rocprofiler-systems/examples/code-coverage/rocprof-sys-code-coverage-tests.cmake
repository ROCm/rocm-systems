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

# -------------------------------------------------------------------------------------- #
#
# code-coverage tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage
    TARGET code-coverage
    REWRITE_ARGS
        -e
        -v
        2
        --min-instructions=4
        -E
        ^std::
        -M
        coverage
        --coverage
        function
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        -M
        coverage
        --coverage
        function
        --module-restrict
        code.coverage
    LABELS "coverage;function-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage-hybrid
    TARGET code-coverage
    REWRITE_ARGS -e -v 2 --min-instructions=4 -E ^std:: --coverage function
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        --coverage
        function
        --module-restrict
        code.coverage
    LABELS "coverage;function-coverage;hybrid-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage-basic-blocks
    TARGET code-coverage
    REWRITE_ARGS
        -e
        -v
        2
        --min-instructions=4
        -E
        ^std::
        -M
        coverage
        --coverage
        basic_block
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        -M
        coverage
        --coverage
        basic_block
        --module-restrict
        code.coverage
    LABELS "coverage;bb-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME code-coverage-basic-blocks-hybrid
    TARGET code-coverage
    REWRITE_ARGS -e -v 2 --min-instructions=4 -E ^std:: --coverage basic_block
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        --coverage
        basic_block
        --module-restrict
        code.coverage
    LABELS "coverage;bb-coverage;hybrid-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
)
