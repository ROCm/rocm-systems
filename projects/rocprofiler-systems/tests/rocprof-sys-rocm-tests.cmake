# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -------------------------------------------------------------------------------------- #
#
# ROCm transpose tests
#
# -------------------------------------------------------------------------------------- #

set(_transpose_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy,memory_allocation,hsa_api"
)

# Enable ROCPD for tests only if valid ROCm is installed and a valid GPU is detected
if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _transpose_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

rocprofiler_systems_add_test(
    NAME transpose
    TARGET transpose
    MPI ${TRANSPOSE_USE_MPI}
    GPU ON
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS -e -v 2 --print-instructions -E uniform_int_distribution
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        -E
        uniform_int_distribution
    ENVIRONMENT "${_transpose_environment}"
    RUNTIME_TIMEOUT 480
)

rocprofiler_systems_add_test(
    SKIP_REWRITE SKIP_RUNTIME
    NAME transpose-two-kernels
    TARGET transpose
    MPI OFF
    GPU ON
    NUM_PROCS 1
    RUN_ARGS 1 2 2
    ENVIRONMENT "${_transpose_environment}"
)

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME transpose-loops
    TARGET transpose
    LABELS "loops"
    MPI ${TRANSPOSE_USE_MPI}
    GPU ON
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS
        -e
        -v
        2
        --label
        return
        args
        -l
        -i
        8
        -E
        uniform_int_distribution
    RUN_ARGS 2 100 50
    ENVIRONMENT "${_transpose_environment}"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure transpose"
)

# -------------------------------------------------------------------------------------- #
#
# ROCProfiler tests (counter collection)
#
# -------------------------------------------------------------------------------------- #

if(ROCPROFSYS_GFX_TARGETS)
    foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
        rocprofiler_systems_lookup_gfx(${arch} GPU_CATEGORY)
        if("instinct" IN_LIST GPU_CATEGORY)
            continue()
        endif()
        set(NAVI_DETECTED TRUE)
        break()
    endforeach()
endif()

if(NAVI_DETECTED)
    set(ROCPROFSYS_ROCM_EVENTS_TEST "SQ_WAVES")
    set(ROCPROFSYS_COUNTER_NAMES_ARG "SQ_WAVES")
else()
    set(ROCPROFSYS_ROCM_EVENTS_TEST "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY")
    set(ROCPROFSYS_COUNTER_NAMES_ARG "GRBM_COUNT" "SQ_WAVES" "SQ_INSTS_VALU" "TA_TA_BUSY")
endif()

rocprofiler_systems_add_test(
    SKIP_BASELINE SKIP_RUNTIME
    NAME transpose-rocprofiler
    TARGET transpose
    LABELS "rocprofiler"
    MPI ${TRANSPOSE_USE_MPI}
    GPU ON
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS -e -v 2 -E uniform_int_distribution
    ENVIRONMENT
        "${_transpose_environment};ROCPROFSYS_ROCM_EVENTS=${ROCPROFSYS_ROCM_EVENTS_TEST}"
    REWRITE_RUN_PASS_REGEX "${_ROCP_PASS_REGEX}"
    SAMPLING_PASS_REGEX "${_ROCP_PASS_REGEX}"
)

rocprofiler_systems_add_validation_test(
    NAME transpose-rocprofiler-sampling
    PERFETTO_FILE "perfetto-trace.proto"
    ARGS --counter-names ${ROCPROFSYS_COUNTER_NAMES_ARG} -p
    LABELS "rocprofiler"
)

rocprofiler_systems_add_validation_test(
    NAME transpose-rocprofiler-binary-rewrite
    PERFETTO_FILE "perfetto-trace.proto"
    ARGS --counter-names ${ROCPROFSYS_COUNTER_NAMES_ARG} -p
    LABELS "rocprofiler"
)

# Verify counter output files exist for any device ID (0-9).
# The device number in the filename comes from device_type_index which depends on
# the CI runner's GPU topology, so we check all possible IDs.
set(_rocprof_output_dir "${PROJECT_BINARY_DIR}/rocprof-sys-tests-output")
foreach(
    _PARENT_TEST
    transpose-rocprofiler-sampling
    transpose-rocprofiler-binary-rewrite-run
)
    if(NOT TEST "${_PARENT_TEST}")
        continue()
    endif()

    # The output directory strips "-run" from the test name
    # (e.g., "binary-rewrite-run" outputs to "binary-rewrite/")
    string(REGEX REPLACE "-run$" "" _OUTPUT_SUBDIR "${_PARENT_TEST}")

    # Binary-rewrite tests have a cleanup fixture that deletes the output directory,
    # so validation must run before cleanup by requiring the same fixture.
    set(_FIXTURES "rocprofsys-global-tmp-files")
    if("${_OUTPUT_SUBDIR}" MATCHES "-binary-rewrite$")
        list(APPEND _FIXTURES "${_OUTPUT_SUBDIR}-fixture")
    endif()

    foreach(_COUNTER ${ROCPROFSYS_COUNTER_NAMES_ARG})
        add_test(
            NAME validate-${_PARENT_TEST}-rocprof-device-${_COUNTER}-exists
            COMMAND
                bash -c
                "for i in {0..9}; do test -e ${_rocprof_output_dir}/${_OUTPUT_SUBDIR}/rocprof-device-\${i}-${_COUNTER}.txt && exit 0; done; exit 1"
        )

        set_tests_properties(
            validate-${_PARENT_TEST}-rocprof-device-${_COUNTER}-exists
            PROPERTIES
                DEPENDS "${_PARENT_TEST}"
                LABELS "rocprofiler;validate;rocm"
                TIMEOUT 30
                FIXTURES_REQUIRED "${_FIXTURES}"
        )
    endforeach()
endforeach()

# -------------------------------------------------------------------------------------- #
#
# ROCpd tests
#
# -------------------------------------------------------------------------------------- #

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU} AND TEST transpose-sampling)
    set_property(TEST transpose-sampling APPEND PROPERTY LABELS rocpd)

    rocprofiler_systems_add_validation_test(
        NAME transpose-sampling
        ROCPD_FILE "rocpd.db"
        ARGS --validation-rules
        "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/transpose/validation-rules.json"
        "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/default-rules.json"
        "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/transpose/amd-smi-rules.json"
        "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/transpose/cpu-metrics-rules.json"
        "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/transpose/timer-sampling-rules.json"
        "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/transpose/sdk-metrics-rules.json"
        LABELS "rocpd"
    )
endif()
