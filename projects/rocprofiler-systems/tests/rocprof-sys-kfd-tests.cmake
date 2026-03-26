# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

# -------------------------------------------------------------------------------------- #
#
# KFD event tests using jacobi-fortran-usm (requires XNACK-capable GPU)
#
# -------------------------------------------------------------------------------------- #

if(NOT TARGET jacobi-fortran-usm)
    rocprofiler_systems_message(
        WARNING "KFD tests disabled: jacobi-fortran-usm target not available"
    )
    return()
endif()

if(NOT _VALID_GPU)
    rocprofiler_systems_message(WARNING "KFD tests disabled: no valid GPU detected")
    return()
endif()

check_rocminfo("xnack" _XNACK_SUPPORTED)
if(NOT _XNACK_SUPPORTED)
    rocprofiler_systems_message(
        WARNING "KFD tests disabled: GPU does not support XNACK (required for KFD page fault/migrate events)"
    )
    return()
endif()

set(_kfd_environment
    "${_base_environment}"
    "HSA_XNACK=1"
    "OMPX_APU_MAPS=1"
    "ROCPROFSYS_USE_AMD_SMI=OFF"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,kfd_events"
)

if(${ENABLE_ROCPD_TEST})
    list(APPEND _kfd_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

# -------------------------------------------------------------------------------------- #
#
# KFD sampling test — runs jacobi-fortran-usm with KFD event tracing
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_REWRITE SKIP_RUNTIME
    NAME kfd-jacobi-usm
    TARGET jacobi-fortran-usm
    GPU ON
    MPI OFF
    NUM_PROCS 1
    RUN_ARGS -m 512
    ENVIRONMENT "${_kfd_environment}"
    LABELS "kfd"
    PASS_REGEX "Total Jacobi run time:"
)

# -------------------------------------------------------------------------------------- #
#
# KFD perfetto validation — checks that KFD categories appear in the perfetto trace
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_validation_test(
    NAME kfd-jacobi-usm-sampling
    PERFETTO_FILE "perfetto-trace.proto"
    ARGS -m kfd_page_fault kfd_page_migrate kfd_queue -p
    LABELS "kfd"
)

# -------------------------------------------------------------------------------------- #
#
# KFD rocpd validation — checks that KFD data is present in the database
#
# -------------------------------------------------------------------------------------- #

if(${ENABLE_ROCPD_TEST} AND TEST kfd-jacobi-usm-sampling)
    set_property(TEST kfd-jacobi-usm-sampling APPEND PROPERTY LABELS rocpd kfd)

    rocprofiler_systems_add_validation_test(
        NAME kfd-jacobi-usm-sampling
        ROCPD_FILE "rocpd.db"
        ARGS --validation-rules
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/default-rules.json"
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/kfd/kfd-rules.json"
        LABELS "rocpd;kfd"
    )
endif()
