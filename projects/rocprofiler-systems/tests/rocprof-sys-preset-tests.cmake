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
    NAME preset-sample-workload-trace
    TARGET rocprofiler-systems-sample
    ARGS --workload-trace -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --workload-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-sys-trace
    TARGET rocprofiler-systems-sample
    ARGS --sys-trace -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --sys-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-runtime-trace
    TARGET rocprofiler-systems-sample
    ARGS --runtime-trace -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --runtime-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-trace-gpu
    TARGET rocprofiler-systems-sample
    ARGS --trace-gpu -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-gpu"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-trace-openmp
    TARGET rocprofiler-systems-sample
    ARGS --trace-openmp -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-openmp"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-profile-mpi
    TARGET rocprofiler-systems-sample
    ARGS --profile-mpi -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --profile-mpi"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-trace-hw-counters
    TARGET rocprofiler-systems-sample
    ARGS --trace-hw-counters -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-hw-counters"
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
    NAME preset-run-workload-trace
    TARGET rocprofiler-systems-run
    ARGS --workload-trace -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --workload-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-sys-trace
    TARGET rocprofiler-systems-run
    ARGS --sys-trace -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --sys-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-runtime-trace
    TARGET rocprofiler-systems-run
    ARGS --runtime-trace -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --runtime-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-trace-gpu
    TARGET rocprofiler-systems-run
    ARGS --trace-gpu -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-gpu"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-trace-openmp
    TARGET rocprofiler-systems-run
    ARGS --trace-openmp -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-openmp"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-profile-mpi
    TARGET rocprofiler-systems-run
    ARGS --profile-mpi -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --profile-mpi"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-trace-hw-counters
    TARGET rocprofiler-systems-run
    ARGS --trace-hw-counters -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-hw-counters"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-mutual-exclusion
    TARGET rocprofiler-systems-run
    ARGS --trace-hpc --workload-trace -- ls
    LABELS preset run
    TIMEOUT 30
    FAIL_REGEX "Multiple preset modes specified|Only ONE preset"
    PROPERTIES WILL_FAIL ON
)

# -------------------------------------------------------------------------------------- #
# rocprof-sys-instrument preset tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-quick
    TARGET rocprofiler-systems-instrument
    ARGS --quick -v 2 --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 60
    PASS_REGEX "Preset:        --quick"
)

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-trace-hpc
    TARGET rocprofiler-systems-instrument
    ARGS --trace-hpc -v 2 --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-hpc"
)

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-workload-trace
    TARGET rocprofiler-systems-instrument
    ARGS --workload-trace -v 2 --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 60
    PASS_REGEX "Preset:        --workload-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-sys-trace
    TARGET rocprofiler-systems-instrument
    ARGS --sys-trace -v 2 --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 60
    PASS_REGEX "Preset:        --sys-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-runtime-trace
    TARGET rocprofiler-systems-instrument
    ARGS --runtime-trace -v 2 --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 60
    PASS_REGEX "Preset:        --runtime-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-trace-gpu
    TARGET rocprofiler-systems-instrument
    ARGS --trace-gpu -v 2 --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-gpu"
)

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-trace-openmp
    TARGET rocprofiler-systems-instrument
    ARGS --trace-openmp -v 2 --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-openmp"
)

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-profile-mpi
    TARGET rocprofiler-systems-instrument
    ARGS --profile-mpi -v 2 --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 60
    PASS_REGEX "Preset:        --profile-mpi"
)

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-trace-hw-counters
    TARGET rocprofiler-systems-instrument
    ARGS --trace-hw-counters -v 2 --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 60
    PASS_REGEX "Preset:        --trace-hw-counters"
)

rocprofiler_systems_add_bin_test(
    NAME preset-instrument-mutual-exclusion
    TARGET rocprofiler-systems-instrument
    ARGS --profile-mpi --trace-openmp --simulate -- rocprof-sys-sample
    LABELS preset instrument
    TIMEOUT 30
    FAIL_REGEX "Multiple preset modes specified|Only ONE preset"
    PROPERTIES WILL_FAIL ON
)
