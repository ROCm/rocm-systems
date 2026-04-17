# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# -------------------------------------------------------------------------------------- #
#
# KFD event tests using unified-memory (requires XNACK-capable GPU)
#
# -------------------------------------------------------------------------------------- #

if(NOT TARGET unified-memory)
    rocprofiler_systems_message(
        WARNING "KFD tests disabled: unified-memory target not available"
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
        WARNING
            "KFD tests disabled: GPU does not support XNACK (required for KFD page fault/migrate events)"
    )
    return()
endif()

# ROCprofiker-SDK version < 1.2.1 does not handle KFD_IOCTL_SVM_LOCATION_UNDEFINED
# node IDs (0xFFFFFFFF), causing a fatal crash in the SDK's KFD parsing thread.
if(rocprofiler-sdk_VERSION VERSION_LESS "1.2.2")
    rocprofiler_systems_message(
        WARNING
            "KFD tests disabled: ROCm ${ROCPROFSYS_ROCM_VERSION} (rocprofiler-sdk ${rocprofiler-sdk_VERSION}) has a rocprofiler-sdk bug with undefined node IDs (fixed in ROCm >= 7.3.0)"
    )
    return()
endif()

set(_kfd_environment
    "${_base_environment}"
    "HSA_XNACK=1"
    "ROCPROFSYS_USE_AMD_SMI=OFF"
    "ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,kfd_events"
)

if(${ENABLE_ROCPD_TEST})
    list(APPEND _kfd_environment "ROCPROFSYS_USE_ROCPD=ON")
endif()

# -------------------------------------------------------------------------------------- #
#
# KFD sampling test — runs unified-memory with KFD event tracing
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
    SKIP_REWRITE SKIP_RUNTIME
    NAME kfd-unified-memory
    TARGET unified-memory
    GPU ON
    MPI OFF
    NUM_PROCS 1
    RUN_ARGS -s 4 -p 32 -i 2
    ENVIRONMENT "${_kfd_environment}"
    LABELS "kfd"
    TIMEOUT 120
    PASS_REGEX "All [0-9]+ tests completed"
)

# -------------------------------------------------------------------------------------- #
#
# KFD perfetto validation — checks that KFD categories appear in the perfetto trace
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_validation_test(
    NAME kfd-unified-memory-sampling
    PERFETTO_FILE "perfetto-trace.proto"
    ARGS -m rocm_kfd_page_fault rocm_kfd_page_migrate rocm_kfd_queue rocm_kfd_event_unmap_from_gpu -p
    LABELS "kfd"
)

# -------------------------------------------------------------------------------------- #
#
# KFD rocpd validation — checks that KFD data is present in the database
#
# -------------------------------------------------------------------------------------- #

if(${ENABLE_ROCPD_TEST} AND TEST kfd-unified-memory-sampling)
    set_property(TEST kfd-unified-memory-sampling APPEND PROPERTY LABELS rocpd kfd)

    rocprofiler_systems_add_validation_test(
        NAME kfd-unified-memory-sampling
        ROCPD_FILE "rocpd.db"
        ARGS --validation-rules
            "${CMAKE_CURRENT_LIST_DIR}/rocpd-validation-rules/kfd/kfd-rules.json"
        LABELS "rocpd;kfd"
    )
endif()
