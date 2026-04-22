# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
    ARGS --preset=balanced -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        balanced"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-profile-only
    TARGET rocprofiler-systems-sample
    ARGS --preset=profile-only -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        profile-only"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-detailed
    TARGET rocprofiler-systems-sample
    ARGS --preset=detailed -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        detailed"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-trace-hpc
    TARGET rocprofiler-systems-sample
    ARGS --preset=trace-hpc -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        trace-hpc"
)

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    rocprofiler_systems_add_bin_test(
        NAME preset-sample-workload-trace
        TARGET rocprofiler-systems-sample
        ARGS --preset=workload-trace -v 2 -- ls
        LABELS preset sample
        TIMEOUT 60
        PASS_REGEX "Preset:        workload-trace"
    )
endif()

rocprofiler_systems_add_bin_test(
    NAME preset-sample-sys-trace
    TARGET rocprofiler-systems-sample
    ARGS --preset=sys-trace -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        sys-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-runtime-trace
    TARGET rocprofiler-systems-sample
    ARGS --preset=runtime-trace -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        runtime-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-trace-gpu
    TARGET rocprofiler-systems-sample
    ARGS --preset=trace-gpu -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        trace-gpu"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-trace-openmp
    TARGET rocprofiler-systems-sample
    ARGS --preset=trace-openmp -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        trace-openmp"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-profile-mpi
    TARGET rocprofiler-systems-sample
    ARGS --preset=profile-mpi -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        profile-mpi"
)

rocprofiler_systems_add_bin_test(
    NAME preset-sample-trace-hw-counters
    TARGET rocprofiler-systems-sample
    ARGS --preset=trace-hw-counters -v 2 -- ls
    LABELS preset sample
    TIMEOUT 60
    PASS_REGEX "Preset:        trace-hw-counters"
)

# -------------------------------------------------------------------------------------- #
# rocprof-sys-sample domain flag tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME domain-sample-gpu
    TARGET rocprofiler-systems-sample
    ARGS --gpu -v 2 -- ls
    LABELS domain sample
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_USE_AMD_SMI=true"
)

rocprofiler_systems_add_bin_test(
    NAME domain-sample-gpu-metrics
    TARGET rocprofiler-systems-sample
    ARGS --gpu=temp,power -v 2 -- ls
    LABELS domain sample
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_AMD_SMI_METRICS=temp,power"
)

rocprofiler_systems_add_bin_test(
    NAME domain-sample-rocm
    TARGET rocprofiler-systems-sample
    ARGS --rocm=hip,kernel -v 2 -- ls
    LABELS domain sample
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"
)

rocprofiler_systems_add_bin_test(
    NAME domain-sample-cpu
    TARGET rocprofiler-systems-sample
    ARGS --cpu=50 -v 2 -- ls
    LABELS domain sample
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_SAMPLING_FREQ=50"
)

rocprofiler_systems_add_bin_test(
    NAME domain-sample-parallel
    TARGET rocprofiler-systems-sample
    ARGS --parallel=mpi,openmp -v 2 -- ls
    LABELS domain sample
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_USE_MPIP=true"
)

rocprofiler_systems_add_bin_test(
    NAME domain-sample-preset-plus-domain
    TARGET rocprofiler-systems-sample
    ARGS --preset=balanced --gpu=temp,power -v 2 -- ls
    LABELS domain sample preset
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_AMD_SMI_METRICS=temp,power"
)

# -------------------------------------------------------------------------------------- #
# rocprof-sys-run preset tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME preset-run-balanced
    TARGET rocprofiler-systems-run
    ARGS --preset=balanced -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        balanced"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-profile-only
    TARGET rocprofiler-systems-run
    ARGS --preset=profile-only -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        profile-only"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-detailed
    TARGET rocprofiler-systems-run
    ARGS --preset=detailed -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        detailed"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-trace-hpc
    TARGET rocprofiler-systems-run
    ARGS --preset=trace-hpc -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        trace-hpc"
)

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    rocprofiler_systems_add_bin_test(
        NAME preset-run-workload-trace
        TARGET rocprofiler-systems-run
        ARGS --preset=workload-trace -v 2 -- ls
        LABELS preset run
        TIMEOUT 60
        PASS_REGEX "Preset:        workload-trace"
    )
endif()

rocprofiler_systems_add_bin_test(
    NAME preset-run-sys-trace
    TARGET rocprofiler-systems-run
    ARGS --preset=sys-trace -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        sys-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-runtime-trace
    TARGET rocprofiler-systems-run
    ARGS --preset=runtime-trace -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        runtime-trace"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-trace-gpu
    TARGET rocprofiler-systems-run
    ARGS --preset=trace-gpu -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        trace-gpu"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-trace-openmp
    TARGET rocprofiler-systems-run
    ARGS --preset=trace-openmp -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        trace-openmp"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-profile-mpi
    TARGET rocprofiler-systems-run
    ARGS --preset=profile-mpi -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        profile-mpi"
)

rocprofiler_systems_add_bin_test(
    NAME preset-run-trace-hw-counters
    TARGET rocprofiler-systems-run
    ARGS --preset=trace-hw-counters -v 2 -- ls
    LABELS preset run
    TIMEOUT 60
    PASS_REGEX "Preset:        trace-hw-counters"
)

# -------------------------------------------------------------------------------------- #
# rocprof-sys-run domain flag tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME domain-run-gpu
    TARGET rocprofiler-systems-run
    ARGS --gpu -v 2 -- ls
    LABELS domain run
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_USE_AMD_SMI=true"
)

rocprofiler_systems_add_bin_test(
    NAME domain-run-gpu-metrics
    TARGET rocprofiler-systems-run
    ARGS --gpu=temp,power -v 2 -- ls
    LABELS domain run
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_AMD_SMI_METRICS=temp,power"
)

rocprofiler_systems_add_bin_test(
    NAME domain-run-rocm
    TARGET rocprofiler-systems-run
    ARGS --rocm=hip,kernel -v 2 -- ls
    LABELS domain run
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"
)

rocprofiler_systems_add_bin_test(
    NAME domain-run-cpu
    TARGET rocprofiler-systems-run
    ARGS --cpu=50 -v 2 -- ls
    LABELS domain run
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_SAMPLING_FREQ=50"
)

rocprofiler_systems_add_bin_test(
    NAME domain-run-parallel
    TARGET rocprofiler-systems-run
    ARGS --parallel=mpi,openmp -v 2 -- ls
    LABELS domain run
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_USE_MPIP=true"
)

rocprofiler_systems_add_bin_test(
    NAME domain-run-preset-plus-domain
    TARGET rocprofiler-systems-run
    ARGS --preset=balanced --gpu=temp,power -v 2 -- ls
    LABELS domain run preset
    TIMEOUT 60
    PASS_REGEX "ROCPROFSYS_AMD_SMI_METRICS=temp,power"
)

# -------------------------------------------------------------------------------------- #
# Export config tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME export-config-run
    TARGET rocprofiler-systems-run
    ARGS --preset=balanced --export-config
    LABELS export run
    TIMEOUT 30
    PASS_REGEX "\"name\": \"balanced\""
)

rocprofiler_systems_add_bin_test(
    NAME export-config-sample
    TARGET rocprofiler-systems-sample
    ARGS --preset=balanced --export-config
    LABELS export sample
    TIMEOUT 30
    PASS_REGEX "\"name\": \"balanced\""
)

# -------------------------------------------------------------------------------------- #
# List presets and explain tests
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_bin_test(
    NAME list-presets-run
    TARGET rocprofiler-systems-run
    ARGS --list-presets
    LABELS preset run
    TIMEOUT 30
    PASS_REGEX "Available Presets:"
)

rocprofiler_systems_add_bin_test(
    NAME list-presets-sample
    TARGET rocprofiler-systems-sample
    ARGS --list-presets
    LABELS preset sample
    TIMEOUT 30
    PASS_REGEX "Available Presets:"
)

rocprofiler_systems_add_bin_test(
    NAME explain-preset-run
    TARGET rocprofiler-systems-run
    ARGS --explain=balanced
    LABELS preset run
    TIMEOUT 30
    PASS_REGEX "Preset: balanced"
)

rocprofiler_systems_add_bin_test(
    NAME explain-preset-sample
    TARGET rocprofiler-systems-sample
    ARGS --explain=balanced
    LABELS preset sample
    TIMEOUT 30
    PASS_REGEX "Preset: balanced"
)
