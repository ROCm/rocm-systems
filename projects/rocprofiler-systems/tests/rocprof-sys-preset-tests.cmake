# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -------------------------------------------------------------------------------------- #
#
# Preset options tests - verify presets work correctly with simple commands
#
# -------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------- #
# rocprof-sys-sample preset tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME preset-sample-balanced
    TARGET rocprofiler-systems-sample
    ARGS --balanced -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --balanced"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-profile-only
    TARGET rocprofiler-systems-sample
    ARGS --profile-only -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        --profile-only"
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

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    rocprofiler_systems_add_bin_test(
        NAME preset-sample-workload-trace
        TARGET rocprofiler-systems-sample
        ARGS --workload-trace -v 2 -- ls
        LABELS preset sample
        TIMEOUT 60
        PASS_REGEX "Preset:        --workload-trace"
    )
endif()

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
    ARGS --balanced --profile-only -- ls
    LABELS preset sample
    TIMEOUT 30
    FAIL_REGEX "Multiple preset modes specified|Only ONE preset"
    PROPERTIES WILL_FAIL ON
)

# -------------------------------------------------------------------------------------- #
# rocprof-sys-run preset tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME preset-run-balanced
    TARGET rocprofiler-systems-run
    ARGS --balanced -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --balanced"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-profile-only
    TARGET rocprofiler-systems-run
    ARGS --profile-only -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        --profile-only"
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

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    rocprofiler_systems_add_bin_test(
        NAME preset-run-workload-trace
        TARGET rocprofiler-systems-run
        ARGS --workload-trace -v 2 -- ls
        LABELS preset run
        TIMEOUT 60
        PASS_REGEX "Preset:        --workload-trace"
    )
endif()

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
