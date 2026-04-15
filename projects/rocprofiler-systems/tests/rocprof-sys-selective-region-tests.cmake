# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

find_package(ROCmVersion)

if(NOT ROCmVersion_FOUND)
    message(
        WARNING
        "ROCmVersion_FOUND not found, skipping tests in ${CMAKE_CURRENT_LIST_FILE}"
    )
    return()
endif()

# -------------------------------------------------------------------------------------- #
#
# selective region tests
#
# -------------------------------------------------------------------------------------- #

set(_selective_region_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,marker_api,kernel_dispatch,marker_core_range_api"
)

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _selective_region_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

# Validation strategy: use -p to print kernel dispatch data, then FAIL_REGEX to
# verify that excluded kernels are NOT present.  This mirrors the shell test
# (test_roctx_regions.sh) which uses assert_kernel_absent for negative checks.
# Positional label matching (-s/-c/-d) is intentionally omitted because:
#   1. Internal kernels (e.g. __amd_rocclr_fillBufferAligned.kd) can appear at
#      any position and shift the expected order.
#   2. Per-kernel counts vary with thread count and iteration count.

set(_print_args -p)

# =========================================================================
# pause_resume — no region filter
# =========================================================================
# Code flow: Z (profiled), A (profiled), pause, B (NOT profiled), resume,
#            C (profiled), D (profiled)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME pause-resume
    TARGET pause_resume
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_selective_region_environment}"
)

rocprofiler_systems_add_validation_test(
    NAME pause-resume-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_Z|CodeBlock_A|CodeBlock_C|CodeBlock_D"
    FAIL_REGEX "CodeBlock_B|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME pause-resume-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_Z|CodeBlock_A|CodeBlock_C|CodeBlock_D"
    FAIL_REGEX "CodeBlock_B|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region — no filter (all regions traced)
# =========================================================================
# All 7 kernels should be present, all 3 regions present.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-all
    TARGET selective_region
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_selective_region_environment}"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-all-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_C|CodeBlock_D|CodeBlock_E|CodeBlock_F|CodeBlock_G"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-all-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_C|CodeBlock_D|CodeBlock_E|CodeBlock_F|CodeBlock_G"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region — Region 1 filter
# =========================================================================
# Region 1 spans: B, C (nested Region 2), D, F (second Region 1)
# Outside: A (before), E (Region 3), G (after)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-r1
    TARGET selective_region
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_selective_region_environment};ROCPROFSYS_SELECTED_REGIONS=Region1"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-r1-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    FAIL_REGEX "CodeBlock_A|CodeBlock_E|CodeBlock_G|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-r1-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    FAIL_REGEX "CodeBlock_A|CodeBlock_E|CodeBlock_G|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region — Region 2 + Region 3 filter
# =========================================================================
# Region 2 spans: C (nested inside Region 1)
# Region 3 spans: E
# Outside: A, B, D, F, G

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-r2r3
    TARGET selective_region
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT
    "${_selective_region_environment};ROCPROFSYS_SELECTED_REGIONS=Region2,Region3"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-r2r3-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_C|CodeBlock_E"
    FAIL_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_D|CodeBlock_F|CodeBlock_G|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-r2r3-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_C|CodeBlock_E"
    FAIL_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_D|CodeBlock_F|CodeBlock_G|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region_pause_1 — no filter
# =========================================================================
# Pause/Resume both inside region. Without filter, pause still applies.
# Code: Z, Region 1, A, pause, B, resume, C, Region 1 stop, D
# Expected: Z, A, C, D profiled. B paused.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-pause1-all
    TARGET selective_region_pause_1
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_selective_region_environment}"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause1-all-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_Z|CodeBlock_A|CodeBlock_C|CodeBlock_D"
    FAIL_REGEX "CodeBlock_B|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause1-all-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_Z|CodeBlock_A|CodeBlock_C|CodeBlock_D"
    FAIL_REGEX "CodeBlock_B|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region_pause_1 — Region 1 filter
# =========================================================================
# With filter: Z outside, A profiled, B paused, C profiled, D outside
# Expected: A, C profiled.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-pause1-r1
    TARGET selective_region_pause_1
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_selective_region_environment};ROCPROFSYS_SELECTED_REGIONS=Region1"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause1-r1-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_A|CodeBlock_C"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_B|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause1-r1-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_A|CodeBlock_C"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_B|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region_pause_2 — no filter
# =========================================================================
# Pause before region. Without filter, pause is global.
# Code: pause, Z, Region 1, A, B, resume, C, Region 1 stop, D
# Expected: C, D profiled. Z, A, B paused.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-pause2-all
    TARGET selective_region_pause_2
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_selective_region_environment}"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause2-all-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_C|CodeBlock_D"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_A|CodeBlock_B|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause2-all-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_C|CodeBlock_D"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_A|CodeBlock_B|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region_pause_2 — Region 1 filter
# =========================================================================
# With filter: pause outside is invalid, A/B/C profiled, Z/D outside
# Expected: A, B, C profiled.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-pause2-r1
    TARGET selective_region_pause_2
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_selective_region_environment};ROCPROFSYS_SELECTED_REGIONS=Region1"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause2-r1-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_C"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause2-r1-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_C"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region_pause_3 — no filter
# =========================================================================
# Pause inside region, resume outside after region stop.
# Code: Region 1, A, pause, C, Region 1 stop, D, resume
# Expected: A profiled. C, D paused.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-pause3-all
    TARGET selective_region_pause_3
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_selective_region_environment}"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause3-all-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_A"
    FAIL_REGEX "CodeBlock_C|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause3-all-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_A"
    FAIL_REGEX "CodeBlock_C|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region_pause_3 — Region 1 filter
# =========================================================================
# With filter: A profiled, C paused, D outside
# Expected: A profiled.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-pause3-r1
    TARGET selective_region_pause_3
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_selective_region_environment};ROCPROFSYS_SELECTED_REGIONS=Region1"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause3-r1-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_A"
    FAIL_REGEX "CodeBlock_C|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause3-r1-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_A"
    FAIL_REGEX "CodeBlock_C|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# ConditionB-only tests: no marker_api in ROCM_DOMAINS
# =========================================================================
# When marker_api is NOT in ROCM_DOMAINS but ROCPROFSYS_SELECTED_REGIONS is set
# (ConditionB only), region filtering still works but pause/resume is IGNORED.
# When neither marker_api nor TRACE_REGION is set, pause/resume is also IGNORED.

set(_no_marker_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"
)

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _no_marker_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

# =========================================================================
# pause_resume without marker_api — no filter
# =========================================================================
# Neither ConditionA (marker_api) nor ConditionB (TRACE_REGION) is set.
# Pause/resume should be IGNORED — ALL kernels profiled.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME pause-resume-no-marker
    TARGET pause_resume
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_no_marker_environment}"
)

rocprofiler_systems_add_validation_test(
    NAME pause-resume-no-marker-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_Z|CodeBlock_A|CodeBlock_B|CodeBlock_C|CodeBlock_D"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME pause-resume-no-marker-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_Z|CodeBlock_A|CodeBlock_B|CodeBlock_C|CodeBlock_D"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region — Region 1 filter, no marker_api (ConditionB only)
# =========================================================================
# Region filtering works. Pause/resume IGNORED.
# Expected: B,C,D,F present. A,E,G absent.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-r1-no-marker
    TARGET selective_region
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_no_marker_environment};ROCPROFSYS_SELECTED_REGIONS=Region1"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-r1-no-marker-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_B|CodeBlock_C|CodeBlock_D|CodeBlock_F"
    FAIL_REGEX "CodeBlock_A|CodeBlock_E|CodeBlock_G|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-r1-no-marker-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_B|CodeBlock_C|CodeBlock_D|CodeBlock_F"
    FAIL_REGEX "CodeBlock_A|CodeBlock_E|CodeBlock_G|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region_pause_1 — Region 1 filter, no marker_api (ConditionB only)
# =========================================================================
# Pause/resume INSIDE region but IGNORED (no marker_api).
# All in-region kernels profiled: A,B,C. Z,D outside.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-pause1-r1-no-marker
    TARGET selective_region_pause_1
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_no_marker_environment};ROCPROFSYS_SELECTED_REGIONS=Region1"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause1-r1-no-marker-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_C"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause1-r1-no-marker-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_C"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region_pause_2 — Region 1 filter, no marker_api (ConditionB only)
# =========================================================================
# Pause OUTSIDE region but IGNORED (no marker_api).
# All in-region kernels profiled: A,B,C. Z,D outside.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-pause2-r1-no-marker
    TARGET selective_region_pause_2
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_no_marker_environment};ROCPROFSYS_SELECTED_REGIONS=Region1"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause2-r1-no-marker-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_C"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause2-r1-no-marker-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_A|CodeBlock_B|CodeBlock_C"
    FAIL_REGEX "CodeBlock_Z|CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

# =========================================================================
# selective_region_pause_3 — Region 1 filter, no marker_api (ConditionB only)
# =========================================================================
# Pause inside region but IGNORED (no marker_api). Region ends while paused.
# All in-region kernels profiled: A,C. D outside.

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_REWRITE SKIP_RUNTIME
    NAME selective-region-pause3-r1-no-marker
    TARGET selective_region_pause_3
    GPU ON
    LABELS "selective_regions;roctx"
    ENVIRONMENT "${_no_marker_environment};ROCPROFSYS_SELECTED_REGIONS=Region1"
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause3-r1-no-marker-sys-run
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx"
    PASS_REGEX "CodeBlock_A|CodeBlock_C"
    FAIL_REGEX "CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)

rocprofiler_systems_add_validation_test(
    NAME selective-region-pause3-r1-no-marker-sampling
    PERFETTO_METRIC "rocm_kernel_dispatch"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "selective_regions;roctx;sampling"
    PASS_REGEX "CodeBlock_A|CodeBlock_C"
    FAIL_REGEX "CodeBlock_D|ROCPROFSYS_ABORT_FAIL_REGEX"
    ARGS ${_print_args}
)
