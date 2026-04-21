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
# HIP graph + rocTX range tests
#
# -------------------------------------------------------------------------------------- #

if(NOT TARGET hip-graph-bubbles)
    return()
endif()

if(ROCPROFSYS_GFX_TARGETS)
    foreach(arch IN LISTS ROCPROFSYS_GFX_TARGETS)
        rocprofiler_systems_lookup_gfx(${arch} GPU_CATEGORY)
        if("instinct" IN_LIST GPU_CATEGORY)
            continue()
        endif()
        set(_HIP_GRAPH_BUBBLES_NAVI_DETECTED TRUE)
        break()
    endforeach()
endif()

if(_HIP_GRAPH_BUBBLES_NAVI_DETECTED)
    set(_HIP_GRAPH_BUBBLES_ROCM_EVENTS "SQ_WAVES")
else()
    set(_HIP_GRAPH_BUBBLES_ROCM_EVENTS "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU")
endif()

# Omit AMD SMI PMC/metrics for this test so ROCPD PMC row counts are not skewed
set(_hip_graph_bubbles_environment
    "${_base_environment}"
    "ROCPROFSYS_ROCM_EVENTS=${_HIP_GRAPH_BUBBLES_ROCM_EVENTS}"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,marker_api"
    "ROCPROFSYS_USE_AMD_SMI=OFF"
)

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU})
    list(APPEND _hip_graph_bubbles_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

# RUN_ARGS: num_kernels num_iterations for hip-graph-bubbles (argv order matches the example).
# If these change, update tests/rocpd-validation-rules/hip-graph-bubbles/graph-bubbles-rules.json:
#   kernels expected count = num_kernels * num_iterations, min_rows, and regions graph_launch count = num_iterations.
rocprofiler_systems_add_test(
    SKIP_REWRITE
    SKIP_RUNTIME
    NAME hip-graph-bubbles
    TARGET hip-graph-bubbles
    GPU ON
    RUN_ARGS 2000 10
    LABELS "hip-graph-bubbles"
    ENVIRONMENT "${_hip_graph_bubbles_environment}"
    REWRITE_TIMEOUT 900
    SAMPLING_TIMEOUT 900
    SYS_RUN_TIMEOUT 900
    BASELINE_PASS_REGEX "Test completed successfully"
    SAMPLING_PASS_REGEX "Test completed successfully"
    SYS_RUN_PASS_REGEX "Test completed successfully"
)

if(TEST hip-graph-bubbles-sampling AND NOT ROCPROFSYS_INSIDE_DOCKER)
    set(_hip_graph_bubbles_fail_regex "HIP error" "ROCPROFSYS_ABORT_FAIL_REGEX")

    set_property(
        TEST hip-graph-bubbles-sampling
        APPEND
        PROPERTY FAIL_REGULAR_EXPRESSION "${_hip_graph_bubbles_fail_regex}"
    )
endif()

if(NOT ROCPROFSYS_INSIDE_DOCKER AND TEST hip-graph-bubbles-sampling)
    rocprofiler_systems_add_validation_test(
        NAME hip-graph-bubbles-sampling
        PERFETTO_METRIC "rocm_marker_api"
        PERFETTO_FILE "perfetto-trace.proto"
        LABELS "hip-graph-bubbles"
        ARGS -p
    )
endif()

if(${ENABLE_ROCPD_TEST} AND ${_VALID_GPU} AND TEST hip-graph-bubbles-sampling)
    set_property(TEST hip-graph-bubbles-sampling APPEND PROPERTY LABELS rocpd)

    rocprofiler_systems_add_validation_test(
        NAME hip-graph-bubbles-sampling
        ROCPD_FILE "rocpd.db"
        ARGS --validation-rules
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/hip-graph-bubbles/graph-bubbles-rules.json"
        LABELS "hip-graph-bubbles;rocpd"
    )
endif()
