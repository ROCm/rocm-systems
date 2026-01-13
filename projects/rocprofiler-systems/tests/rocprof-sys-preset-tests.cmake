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
# Preset options tests - verify presets work correctly with simple commands
#
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
# rocprof-sys-sample preset tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME preset-sample-quick
    TARGET rocprofiler-systems-sample
    ARGS --quick -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --quick"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-simple
    TARGET rocprofiler-systems-sample
    ARGS --simple -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --simple"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-detailed
    TARGET rocprofiler-systems-sample
    ARGS --detailed -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --detailed"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-trace-hpc
    TARGET rocprofiler-systems-sample
    ARGS --trace-hpc -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-hpc"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-trace-ai
    TARGET rocprofiler-systems-sample
    ARGS --trace-ai -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-ai"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-mutual-exclusion
    TARGET rocprofiler-systems-sample
    ARGS --quick --simple -- ls
    LABELS preset sample
    TIMEOUT 30
    FAIL_REGEX "Multiple preset modes specified|Only ONE preset"
    PROPERTIES WILL_FAIL ON
)

# -------------------------------------------------------------------------------------- #
# rocprof-sys-run preset tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME preset-run-quick
    TARGET rocprofiler-systems-run
    ARGS --quick -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --quick"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-simple
    TARGET rocprofiler-systems-run
    ARGS --simple -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --simple"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-detailed
    TARGET rocprofiler-systems-run
    ARGS --detailed -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --detailed"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-trace-hpc
    TARGET rocprofiler-systems-run
    ARGS --trace-hpc -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-hpc"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-trace-ai
    TARGET rocprofiler-systems-run
    ARGS --trace-ai -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-ai"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-mutual-exclusion
    TARGET rocprofiler-systems-run
    ARGS --trace-hpc --trace-ai -- ls
    LABELS preset run
    TIMEOUT 30
    FAIL_REGEX "Multiple preset modes specified|Only ONE preset"
    PROPERTIES WILL_FAIL ON
)
