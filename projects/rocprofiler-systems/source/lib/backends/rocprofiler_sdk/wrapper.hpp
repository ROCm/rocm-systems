// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/cxx/details/name_info.hpp>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/marker/api_id.h>
#include <rocprofiler-sdk/version.h>

#if ROCPROFILER_VERSION >= 600
#    include <rocprofiler-sdk/ompt/api_id.h>
#endif
#include <rocprofiler-sdk/rocprofiler.h>

#if __has_include(<rocprofiler-sdk/experimental/registration.h>)
#    include <rocprofiler-sdk/experimental/registration.h>
#else
#    include <rocprofiler-sdk/registration.h>
#endif

#if ROCPROFILER_VERSION >= 10000
#    include <rocprofiler-sdk/counter_config.h>
#    include <rocprofiler-sdk/dispatch_counting_service.h>
#    include <rocprofiler-sdk/kfd/kfd_id.h>
#else
#    include <rocprofiler-sdk/dispatch_counting_service.h>
#    include <rocprofiler-sdk/profile_config.h>
#endif

#if ROCPROFILER_VERSION >= 600
#    include <rocprofiler-sdk/rccl/api_args.h>
#    include <rocprofiler-sdk/rccl/api_id.h>
#endif

#if ROCPROFILER_VERSION >= 700
#    include <rocprofiler-sdk/hip.h>
#endif

namespace rocprofsys::rocprofiler_sdk
{

/// 1:1 abstraction layer over the rocprofiler-sdk C API.
///
/// Exposes SDK types as nested `using` aliases, SDK enum values as
/// `static constexpr` constants, and SDK functions as `static` methods.
/// No logic beyond direct forwarding lives here.
///
/// This is the `Wrapper` template argument consumed by
/// rocprofsys::backends::rocprofiler_sdk::backend<Wrapper> (backend.hpp); it must
/// expose the same nested types/constants/static methods that struct relies on.
struct backend
{
    // ─── Compile-time SDK version ────────────────────────────────────────────────
    static constexpr std::uint32_t compile_time_version = ROCPROFILER_VERSION;

    // ─── Scalar handle / ID types ────────────────────────────────────────────────
    using status_t              = rocprofiler_status_t;
    using context_id            = rocprofiler_context_id_t;
    using buffer_id             = rocprofiler_buffer_id_t;
    using agent_id              = rocprofiler_agent_id_t;
    using queue_id              = rocprofiler_queue_id_t;
    using timestamp_t           = rocprofiler_timestamp_t;
    using thread_id             = rocprofiler_thread_id_t;
    using counter_id            = rocprofiler_counter_id_t;
    using dispatch_id_t         = rocprofiler_dispatch_id_t;
    using counter_instance_id_t = rocprofiler_counter_instance_id_t;
    using callback_thread_id    = rocprofiler_callback_thread_t;

    // ─── Agent / device types ────────────────────────────────────────────────────
    using agent_t         = rocprofiler_agent_t;
    using agent_type_t    = rocprofiler_agent_type_t;
    using agent_version_t = rocprofiler_agent_version_t;

    // ─── Enum / flag types ────────────────────────────────────────────────────────
    using buffer_category_t     = rocprofiler_buffer_category_t;
    using buffer_policy_t       = rocprofiler_buffer_policy_t;
    using runtime_library_t     = rocprofiler_runtime_library_t;
    using callback_phase_t      = rocprofiler_callback_phase_t;
    using tracing_operation     = rocprofiler_tracing_operation_t;
    using callback_tracing_kind = rocprofiler_callback_tracing_kind_t;
    using buffer_tracing_kind   = rocprofiler_buffer_tracing_kind_t;
    using external_correlation_request_kind =
        rocprofiler_external_correlation_id_request_kind_t;
    using counter_info_version_id_t = rocprofiler_counter_info_version_id_t;
    using counter_info_v0_t         = rocprofiler_counter_info_v0_t;
#if ROCPROFILER_VERSION >= 10000
    using counter_info_v1_t = rocprofiler_counter_info_v1_t;
#endif
    using scratch_memory_operation_t = rocprofiler_scratch_memory_operation_t;
    using marker_op_t                = rocprofiler_marker_core_api_id_t;
    using marker_control_op_t        = rocprofiler_marker_control_api_id_t;

    // ─── Client / registration types ────────────────────────────────────────────
    using client_id_t       = rocprofiler_client_id_t;
    using client_finalize_t = rocprofiler_client_finalize_t;
#if __has_include(<rocprofiler-sdk/experimental/registration.h>)
    using client_detach_t = rocprofiler_client_detach_t;
#else
    using client_detach_t = rocprofiler_client_finalize_t;
#endif

    // ─── Correlation types ────────────────────────────────────────────────────────
    using correlation_id_t = rocprofiler_correlation_id_t;

    // ─── Buffer/callback tracing record types ────────────────────────────────────
    using record_header_t         = rocprofiler_record_header_t;
    using user_data_t             = rocprofiler_user_data_t;
    using kernel_id_t             = rocprofiler_kernel_id_t;
    using kernel_dispatch_record  = rocprofiler_buffer_tracing_kernel_dispatch_record_t;
    using kernel_dispatch_data    = rocprofiler_callback_tracing_kernel_dispatch_data_t;
    using dimension_info_t        = rocprofiler_record_dimension_info_t;
    using memory_copy_record      = rocprofiler_buffer_tracing_memory_copy_record_t;
    using scratch_memory_record   = rocprofiler_buffer_tracing_scratch_memory_record_t;
    using callback_tracing_record = rocprofiler_callback_tracing_record_t;
    using code_object_load_data   = rocprofiler_callback_tracing_code_object_load_data_t;
    using kernel_symbol_data =
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
    using marker_payload_t = rocprofiler_callback_tracing_marker_api_data_t;

    // ─── Callback / iterator function pointer types ──────────────────────────────
    using buffer_tracing_cb_t   = rocprofiler_buffer_tracing_cb_t;
    using callback_tracing_cb_t = rocprofiler_callback_tracing_cb_t;
    using external_correlation_id_request_cb_t =
        rocprofiler_external_correlation_id_request_cb_t;
    using internal_thread_library_cb_t = rocprofiler_internal_thread_library_cb_t;
    using query_available_agents_cb_t  = rocprofiler_query_available_agents_cb_t;
    using callback_tracing_operation_args_cb_t =
        rocprofiler_callback_tracing_operation_args_cb_t;
    using available_counters_cb_t   = rocprofiler_available_counters_cb_t;
    using available_dimensions_cb_t = rocprofiler_available_dimensions_cb_t;
#if ROCPROFILER_VERSION >= 10000
    using device_counting_agent_cb_t   = rocprofiler_device_counting_agent_cb_t;
    using device_counting_service_cb_t = rocprofiler_device_counting_service_cb_t;
#else
    using device_counting_agent_cb_t   = rocprofiler_agent_set_profile_callback_t;
    using device_counting_service_cb_t = rocprofiler_device_counting_service_callback_t;
#endif
    using counter_flag_t = rocprofiler_counter_flag_t;

    // ─── Counter / dispatch-counting types (version-gated) ───────────────────────
#if ROCPROFILER_VERSION >= 10000
    using counter_config_id            = rocprofiler_counter_config_id_t;
    using counter_record               = rocprofiler_counter_record_t;
    using dispatch_counting_data       = rocprofiler_dispatch_counting_service_data_t;
    using dispatch_counting_service_cb = rocprofiler_dispatch_counting_service_cb_t;
    using dispatch_counting_record_cb  = rocprofiler_dispatch_counting_record_cb_t;
#else
    using counter_config_id            = rocprofiler_profile_config_id_t;
    using counter_record               = rocprofiler_record_counter_t;
    using dispatch_counting_data       = rocprofiler_dispatch_counting_service_data_t;
    using dispatch_counting_service_cb = rocprofiler_dispatch_counting_service_callback_t;
    using dispatch_counting_record_cb  = rocprofiler_profile_counting_record_callback_t;
#endif

    // ─── Stream ID (synthesized for SDK < 700) ───────────────────────────────────
#if ROCPROFILER_VERSION >= 700
    using stream_id = rocprofiler_stream_id_t;
#else
    struct stream_id
    {
        std::uint64_t handle{};
        bool          operator==(const stream_id&) const = default;
    };
#endif

    // ─── Version-gated record types ──────────────────────────────────────────────
#if ROCPROFILER_VERSION < 10000
    using page_migration_record = rocprofiler_buffer_tracing_page_migration_record_t;
#endif

#if ROCPROFILER_VERSION >= 600
    using ompt_data_t      = rocprofiler_callback_tracing_ompt_data_t;
    using ompt_operation_t = rocprofiler_ompt_operation_t;
    using ompt_thread_t    = ::ompt_thread_t;

    static constexpr ompt_thread_t OMPT_THREAD_INITIAL = ompt_thread_initial;

    // ─── OMPT operation constants ─────────────────────────────────────────────
    static constexpr ompt_operation_t OMPT_ID_NONE = ROCPROFILER_OMPT_ID_NONE;
    static constexpr ompt_operation_t OMPT_ID_thread_begin =
        ROCPROFILER_OMPT_ID_thread_begin;
    static constexpr ompt_operation_t OMPT_ID_thread_end = ROCPROFILER_OMPT_ID_thread_end;
    static constexpr ompt_operation_t OMPT_ID_parallel_begin =
        ROCPROFILER_OMPT_ID_parallel_begin;
    static constexpr ompt_operation_t OMPT_ID_parallel_end =
        ROCPROFILER_OMPT_ID_parallel_end;
    static constexpr ompt_operation_t OMPT_ID_task_create =
        ROCPROFILER_OMPT_ID_task_create;
    static constexpr ompt_operation_t OMPT_ID_task_schedule =
        ROCPROFILER_OMPT_ID_task_schedule;
    static constexpr ompt_operation_t OMPT_ID_implicit_task =
        ROCPROFILER_OMPT_ID_implicit_task;
    static constexpr ompt_operation_t OMPT_ID_device_initialize =
        ROCPROFILER_OMPT_ID_device_initialize;
    static constexpr ompt_operation_t OMPT_ID_device_finalize =
        ROCPROFILER_OMPT_ID_device_finalize;
    static constexpr ompt_operation_t OMPT_ID_device_load =
        ROCPROFILER_OMPT_ID_device_load;
    static constexpr ompt_operation_t OMPT_ID_sync_region_wait =
        ROCPROFILER_OMPT_ID_sync_region_wait;
    static constexpr ompt_operation_t OMPT_ID_mutex_released =
        ROCPROFILER_OMPT_ID_mutex_released;
    static constexpr ompt_operation_t OMPT_ID_dependences =
        ROCPROFILER_OMPT_ID_dependences;
    static constexpr ompt_operation_t OMPT_ID_task_dependence =
        ROCPROFILER_OMPT_ID_task_dependence;
    static constexpr ompt_operation_t OMPT_ID_work   = ROCPROFILER_OMPT_ID_work;
    static constexpr ompt_operation_t OMPT_ID_masked = ROCPROFILER_OMPT_ID_masked;
    static constexpr ompt_operation_t OMPT_ID_sync_region =
        ROCPROFILER_OMPT_ID_sync_region;
    static constexpr ompt_operation_t OMPT_ID_lock_init = ROCPROFILER_OMPT_ID_lock_init;
    static constexpr ompt_operation_t OMPT_ID_lock_destroy =
        ROCPROFILER_OMPT_ID_lock_destroy;
    static constexpr ompt_operation_t OMPT_ID_mutex_acquire =
        ROCPROFILER_OMPT_ID_mutex_acquire;
    static constexpr ompt_operation_t OMPT_ID_mutex_acquired =
        ROCPROFILER_OMPT_ID_mutex_acquired;
    static constexpr ompt_operation_t OMPT_ID_nest_lock  = ROCPROFILER_OMPT_ID_nest_lock;
    static constexpr ompt_operation_t OMPT_ID_flush      = ROCPROFILER_OMPT_ID_flush;
    static constexpr ompt_operation_t OMPT_ID_cancel     = ROCPROFILER_OMPT_ID_cancel;
    static constexpr ompt_operation_t OMPT_ID_reduction  = ROCPROFILER_OMPT_ID_reduction;
    static constexpr ompt_operation_t OMPT_ID_dispatch   = ROCPROFILER_OMPT_ID_dispatch;
    static constexpr ompt_operation_t OMPT_ID_target_emi = ROCPROFILER_OMPT_ID_target_emi;
    static constexpr ompt_operation_t OMPT_ID_target_data_op_emi =
        ROCPROFILER_OMPT_ID_target_data_op_emi;
    static constexpr ompt_operation_t OMPT_ID_target_submit_emi =
        ROCPROFILER_OMPT_ID_target_submit_emi;
    static constexpr ompt_operation_t OMPT_ID_error = ROCPROFILER_OMPT_ID_error;
    static constexpr ompt_operation_t OMPT_ID_callback_functions =
        ROCPROFILER_OMPT_ID_callback_functions;
    static constexpr ompt_operation_t OMPT_ID_LAST = ROCPROFILER_OMPT_ID_LAST;

    using rccl_api_data    = rocprofiler_callback_tracing_rccl_api_data_t;
    using rccl_api_id_t    = rocprofiler_rccl_api_id_t;
    using nccl_data_type_t = ncclDataType_t;
    using nccl_comm_t      = ncclComm_t;
    using nccl_result_t    = ncclResult_t;

    // ─── NCCL data type constants ─────────────────────────────────────────────
    static constexpr nccl_result_t    NCCL_SUCCESS  = ncclSuccess;
    static constexpr nccl_data_type_t NCCL_INT8     = ncclInt8;
    static constexpr nccl_data_type_t NCCL_UINT8    = ncclUint8;
    static constexpr nccl_data_type_t NCCL_FLOAT16  = ncclFloat16;
    static constexpr nccl_data_type_t NCCL_BFLOAT16 = ncclBfloat16;
    static constexpr nccl_data_type_t NCCL_INT32    = ncclInt32;
    static constexpr nccl_data_type_t NCCL_UINT32   = ncclUint32;
    static constexpr nccl_data_type_t NCCL_FLOAT32  = ncclFloat32;
    static constexpr nccl_data_type_t NCCL_INT64    = ncclInt64;
    static constexpr nccl_data_type_t NCCL_UINT64   = ncclUint64;
    static constexpr nccl_data_type_t NCCL_FLOAT64  = ncclFloat64;
#    if defined(ncclFp8E4M3) && defined(ncclFp8E5M2)
    static constexpr nccl_data_type_t NCCL_FP8_E4M3 = ncclFp8E4M3;
    static constexpr nccl_data_type_t NCCL_FP8_E5M2 = ncclFp8E5M2;
#    endif

    // ─── RCCL API ID constants ─────────────────────────────────────────────────
    static constexpr rccl_api_id_t RCCL_API_ID_ncclAllGather =
        ROCPROFILER_RCCL_API_ID_ncclAllGather;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclAllToAll =
        ROCPROFILER_RCCL_API_ID_ncclAllToAll;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclAllReduce =
        ROCPROFILER_RCCL_API_ID_ncclAllReduce;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclGather =
        ROCPROFILER_RCCL_API_ID_ncclGather;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclRecv =
        ROCPROFILER_RCCL_API_ID_ncclRecv;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclReduce =
        ROCPROFILER_RCCL_API_ID_ncclReduce;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclBroadcast =
        ROCPROFILER_RCCL_API_ID_ncclBroadcast;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclReduceScatter =
        ROCPROFILER_RCCL_API_ID_ncclReduceScatter;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclSend =
        ROCPROFILER_RCCL_API_ID_ncclSend;
    using memory_alloc_record = rocprofiler_buffer_tracing_memory_allocation_record_t;
#endif

#if ROCPROFILER_VERSION >= 700
    using hip_stream_data        = rocprofiler_callback_tracing_hip_stream_data_t;
    using hip_stream_operation_t = rocprofiler_hip_stream_operation_t;
#endif

#if ROCPROFILER_VERSION >= 10000
    using kfd_page_fault_record   = rocprofiler_buffer_tracing_kfd_page_fault_record_t;
    using kfd_page_migrate_record = rocprofiler_buffer_tracing_kfd_page_migrate_record_t;
    using kfd_queue_record        = rocprofiler_buffer_tracing_kfd_queue_record_t;
    using kfd_event_queue_record  = rocprofiler_buffer_tracing_kfd_event_queue_record_t;
    using kfd_event_unmap_record =
        rocprofiler_buffer_tracing_kfd_event_unmap_from_gpu_record_t;
    using kfd_event_dropped_record =
        rocprofiler_buffer_tracing_kfd_event_dropped_events_record_t;
    using kfd_event_queue_operation_t = rocprofiler_kfd_event_queue_operation_t;
    using kfd_event_unmap_from_gpu_operation_t =
        rocprofiler_kfd_event_unmap_from_gpu_operation_t;
    using kfd_page_fault_operation_t   = rocprofiler_kfd_page_fault_operation_t;
    using kfd_page_migrate_operation_t = rocprofiler_kfd_page_migrate_operation_t;
    using kfd_queue_operation_t        = rocprofiler_kfd_queue_operation_t;
#endif

    // ─── Status constants ────────────────────────────────────────────────────────
    static constexpr status_t STATUS_SUCCESS = ROCPROFILER_STATUS_SUCCESS;
    static constexpr status_t STATUS_ERROR   = ROCPROFILER_STATUS_ERROR;
    static constexpr status_t STATUS_ERROR_BUFFER_BUSY =
        ROCPROFILER_STATUS_ERROR_BUFFER_BUSY;
    static constexpr status_t STATUS_ERROR_CONTEXT_ERROR =
        ROCPROFILER_STATUS_ERROR_CONTEXT_ERROR;
    static constexpr status_t STATUS_ERROR_HSA_NOT_LOADED =
        ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;
    static constexpr status_t STATUS_ERROR_INVALID_ARGUMENT =
        ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    // ─── Callback phase constants ────────────────────────────────────────────────
    static constexpr callback_phase_t CALLBACK_PHASE_ENTER =
        ROCPROFILER_CALLBACK_PHASE_ENTER;
    static constexpr callback_phase_t CALLBACK_PHASE_EXIT =
        ROCPROFILER_CALLBACK_PHASE_EXIT;
    static constexpr callback_phase_t CALLBACK_PHASE_NONE =
        ROCPROFILER_CALLBACK_PHASE_NONE;

    // ─── Agent type constants ────────────────────────────────────────────────────
    static constexpr agent_type_t AGENT_TYPE_CPU = ROCPROFILER_AGENT_TYPE_CPU;
    static constexpr agent_type_t AGENT_TYPE_GPU = ROCPROFILER_AGENT_TYPE_GPU;

    // ─── Agent version constants ─────────────────────────────────────────────────
    static constexpr agent_version_t AGENT_INFO_VERSION_0 =
        ROCPROFILER_AGENT_INFO_VERSION_0;

    // ─── Buffer category constants ───────────────────────────────────────────────
    static constexpr buffer_category_t BUFFER_CATEGORY_TRACING =
        ROCPROFILER_BUFFER_CATEGORY_TRACING;
    static constexpr buffer_policy_t BUFFER_POLICY_LOSSLESS =
        ROCPROFILER_BUFFER_POLICY_LOSSLESS;

    // ─── Runtime library flag constants ──────────────────────────────────────────
    static constexpr runtime_library_t LIBRARY        = ROCPROFILER_LIBRARY;
    static constexpr runtime_library_t HSA_LIBRARY    = ROCPROFILER_HSA_LIBRARY;
    static constexpr runtime_library_t HIP_LIBRARY    = ROCPROFILER_HIP_LIBRARY;
    static constexpr runtime_library_t MARKER_LIBRARY = ROCPROFILER_MARKER_LIBRARY;

    // ─── Callback tracing kind constants ─────────────────────────────────────────
    static constexpr callback_tracing_kind CALLBACK_TRACING_NONE =
        ROCPROFILER_CALLBACK_TRACING_NONE;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HSA_CORE_API =
        ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HSA_AMD_EXT_API =
        ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HSA_IMAGE_EXT_API =
        ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HSA_FINALIZE_EXT_API =
        ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HIP_RUNTIME_API =
        ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HIP_COMPILER_API =
        ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_CODE_OBJECT =
        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MARKER_CORE_API =
        ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MARKER_CONTROL_API =
        ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MARKER_NAME_API =
        ROCPROFILER_CALLBACK_TRACING_MARKER_NAME_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_SCRATCH_MEMORY =
        ROCPROFILER_CALLBACK_TRACING_SCRATCH_MEMORY;
    static constexpr callback_tracing_kind CALLBACK_TRACING_KERNEL_DISPATCH =
        ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MEMORY_COPY =
        ROCPROFILER_CALLBACK_TRACING_MEMORY_COPY;
    static constexpr callback_tracing_kind CALLBACK_TRACING_RCCL_API =
        ROCPROFILER_CALLBACK_TRACING_RCCL_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_LAST =
        ROCPROFILER_CALLBACK_TRACING_LAST;

#if ROCPROFILER_VERSION >= 600
    static constexpr callback_tracing_kind CALLBACK_TRACING_ROCDECODE_API =
        ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_OMPT =
        ROCPROFILER_CALLBACK_TRACING_OMPT;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MEMORY_ALLOCATION =
        ROCPROFILER_CALLBACK_TRACING_MEMORY_ALLOCATION;
    static constexpr callback_tracing_kind CALLBACK_TRACING_RUNTIME_INITIALIZATION =
        ROCPROFILER_CALLBACK_TRACING_RUNTIME_INITIALIZATION;
#endif

#if ROCPROFILER_VERSION >= 700
    static constexpr callback_tracing_kind CALLBACK_TRACING_ROCJPEG_API =
        ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HIP_STREAM =
        ROCPROFILER_CALLBACK_TRACING_HIP_STREAM;
#endif

    // ─── Buffer tracing kind constants ───────────────────────────────────────────
    static constexpr buffer_tracing_kind BUFFER_TRACING_HSA_CORE_API =
        ROCPROFILER_BUFFER_TRACING_HSA_CORE_API;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HSA_AMD_EXT_API =
        ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HSA_IMAGE_EXT_API =
        ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HSA_FINALIZE_EXT_API =
        ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HIP_RUNTIME_API =
        ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HIP_COMPILER_API =
        ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API;
    static constexpr buffer_tracing_kind BUFFER_TRACING_MARKER_CORE_API =
        ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KERNEL_DISPATCH =
        ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH;
    static constexpr buffer_tracing_kind BUFFER_TRACING_MEMORY_COPY =
        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY;
    static constexpr buffer_tracing_kind BUFFER_TRACING_SCRATCH_MEMORY =
        ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY;

#if ROCPROFILER_VERSION < 10000
    static constexpr buffer_tracing_kind BUFFER_TRACING_PAGE_MIGRATION =
        ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION;
#endif

#if ROCPROFILER_VERSION >= 600
    static constexpr buffer_tracing_kind BUFFER_TRACING_MEMORY_ALLOCATION =
        ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION;
#endif

#if ROCPROFILER_VERSION >= 10000
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_PAGE_FAULT =
        ROCPROFILER_BUFFER_TRACING_KFD_PAGE_FAULT;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_PAGE_MIGRATE =
        ROCPROFILER_BUFFER_TRACING_KFD_PAGE_MIGRATE;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_QUEUE =
        ROCPROFILER_BUFFER_TRACING_KFD_QUEUE;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_EVENT_QUEUE =
        ROCPROFILER_BUFFER_TRACING_KFD_EVENT_QUEUE;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU =
        ROCPROFILER_BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS =
        ROCPROFILER_BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS;
#endif

    // ─── Counter flag constants ───────────────────────────────────────────────────
    static constexpr counter_flag_t COUNTER_FLAG_NONE = ROCPROFILER_COUNTER_FLAG_NONE;

    // ─── Counter info version constants ──────────────────────────────────────────
    static constexpr counter_info_version_id_t COUNTER_INFO_VERSION_0 =
        ROCPROFILER_COUNTER_INFO_VERSION_0;
#if ROCPROFILER_VERSION >= 10000
    static constexpr counter_info_version_id_t COUNTER_INFO_VERSION_1 =
        ROCPROFILER_COUNTER_INFO_VERSION_1;
#endif

    // ─── Scratch memory operation constants ──────────────────────────────────────
    static constexpr scratch_memory_operation_t SCRATCH_MEMORY_ALLOC =
        ROCPROFILER_SCRATCH_MEMORY_ALLOC;

    // ─── Code object tracing operation constants ─────────────────────────────────
    static constexpr tracing_operation CODE_OBJECT_LOAD = ROCPROFILER_CODE_OBJECT_LOAD;
    static constexpr tracing_operation CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER =
        ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER;

    // ─── Kernel dispatch operation constants ─────────────────────────────────────
    static constexpr tracing_operation KERNEL_DISPATCH_COMPLETE =
        ROCPROFILER_KERNEL_DISPATCH_COMPLETE;

    // ─── External correlation request kind constants ──────────────────────────────
    static constexpr external_correlation_request_kind
        EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH =
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH;
    static constexpr external_correlation_request_kind
        EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY =
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY;

#if ROCPROFILER_VERSION >= 600
    static constexpr external_correlation_request_kind
        EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION =
            ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION;
#endif

    // ─── Marker API operation constants ──────────────────────────────────────────
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxMarkA =
        ROCPROFILER_MARKER_CORE_API_ID_roctxMarkA;
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxRangePushA =
        ROCPROFILER_MARKER_CORE_API_ID_roctxRangePushA;
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxRangePop =
        ROCPROFILER_MARKER_CORE_API_ID_roctxRangePop;
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxRangeStartA =
        ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStartA;
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxRangeStop =
        ROCPROFILER_MARKER_CORE_API_ID_roctxRangeStop;

    // ─── Marker control operation constants ───────────────────────────────────────
    static constexpr marker_control_op_t MARKER_CONTROL_API_ID_roctxProfilerPause =
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause;
    static constexpr marker_control_op_t MARKER_CONTROL_API_ID_roctxProfilerResume =
        ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume;

#if ROCPROFILER_VERSION >= 700
    // ─── HIP stream operation constants ──────────────────────────────────────────
    static constexpr hip_stream_operation_t HIP_STREAM_CREATE =
        ROCPROFILER_HIP_STREAM_CREATE;
    static constexpr hip_stream_operation_t HIP_STREAM_DESTROY =
        ROCPROFILER_HIP_STREAM_DESTROY;
    static constexpr hip_stream_operation_t HIP_STREAM_SET = ROCPROFILER_HIP_STREAM_SET;
#endif

#if ROCPROFILER_VERSION >= 10000
    // ─── KFD event queue operation constants ─────────────────────────────────────
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_NONE =
        ROCPROFILER_KFD_EVENT_QUEUE_NONE;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_SVM =
        ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SVM;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_USERPTR =
        ROCPROFILER_KFD_EVENT_QUEUE_EVICT_USERPTR;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_TTM =
        ROCPROFILER_KFD_EVENT_QUEUE_EVICT_TTM;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_SUSPEND =
        ROCPROFILER_KFD_EVENT_QUEUE_EVICT_SUSPEND;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT =
        ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE =
        ROCPROFILER_KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_RESTORE_RESCHEDULED =
        ROCPROFILER_KFD_EVENT_QUEUE_RESTORE_RESCHEDULED;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_RESTORE =
        ROCPROFILER_KFD_EVENT_QUEUE_RESTORE;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_LAST =
        ROCPROFILER_KFD_EVENT_QUEUE_LAST;

    // ─── KFD unmap-from-GPU operation constants ───────────────────────────────────
    static constexpr kfd_event_unmap_from_gpu_operation_t KFD_EVENT_UNMAP_FROM_GPU_NONE =
        ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_NONE;
    static constexpr kfd_event_unmap_from_gpu_operation_t
        KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY =
            ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY;
    static constexpr kfd_event_unmap_from_gpu_operation_t
        KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE =
            ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE;
    static constexpr kfd_event_unmap_from_gpu_operation_t
        KFD_EVENT_UNMAP_FROM_GPU_UNMAP_FROM_CPU =
            ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_UNMAP_FROM_CPU;
    static constexpr kfd_event_unmap_from_gpu_operation_t KFD_EVENT_UNMAP_FROM_GPU_LAST =
        ROCPROFILER_KFD_EVENT_UNMAP_FROM_GPU_LAST;

    // ─── KFD page fault operation constants ──────────────────────────────────────
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_NONE =
        ROCPROFILER_KFD_PAGE_FAULT_NONE;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_READ_FAULT_MIGRATED =
        ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_MIGRATED;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_READ_FAULT_UPDATED =
        ROCPROFILER_KFD_PAGE_FAULT_READ_FAULT_UPDATED;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_WRITE_FAULT_MIGRATED =
        ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_MIGRATED;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_WRITE_FAULT_UPDATED =
        ROCPROFILER_KFD_PAGE_FAULT_WRITE_FAULT_UPDATED;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_LAST =
        ROCPROFILER_KFD_PAGE_FAULT_LAST;

    // ─── KFD page migrate operation constants ─────────────────────────────────────
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_NONE =
        ROCPROFILER_KFD_PAGE_MIGRATE_NONE;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_PREFETCH =
        ROCPROFILER_KFD_PAGE_MIGRATE_PREFETCH;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_PAGEFAULT_GPU =
        ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_GPU;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_PAGEFAULT_CPU =
        ROCPROFILER_KFD_PAGE_MIGRATE_PAGEFAULT_CPU;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_TTM_EVICTION =
        ROCPROFILER_KFD_PAGE_MIGRATE_TTM_EVICTION;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_LAST =
        ROCPROFILER_KFD_PAGE_MIGRATE_LAST;

    // ─── KFD queue evict operation constants ──────────────────────────────────────
    static constexpr kfd_queue_operation_t KFD_QUEUE_NONE = ROCPROFILER_KFD_QUEUE_NONE;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_SVM =
        ROCPROFILER_KFD_QUEUE_EVICT_SVM;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_USERPTR =
        ROCPROFILER_KFD_QUEUE_EVICT_USERPTR;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_TTM =
        ROCPROFILER_KFD_QUEUE_EVICT_TTM;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_SUSPEND =
        ROCPROFILER_KFD_QUEUE_EVICT_SUSPEND;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_CRIU_CHECKPOINT =
        ROCPROFILER_KFD_QUEUE_EVICT_CRIU_CHECKPOINT;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_CRIU_RESTORE =
        ROCPROFILER_KFD_QUEUE_EVICT_CRIU_RESTORE;
    static constexpr kfd_queue_operation_t KFD_QUEUE_LAST = ROCPROFILER_KFD_QUEUE_LAST;
#endif

    // ─── SDK function wrappers ────────────────────────────────────────────────────

    static status_t get_version(std::uint32_t* major, std::uint32_t* minor,
                                std::uint32_t* patch)
    {
        return rocprofiler_get_version(major, minor, patch);
    }

    static status_t get_timestamp(timestamp_t* ts) noexcept
    {
        return rocprofiler_get_timestamp(ts);
    }

    static const char* get_status_string(status_t status) noexcept
    {
        return rocprofiler_get_status_string(status);
    }

    static status_t create_context(context_id* ctx)
    {
        return rocprofiler_create_context(ctx);
    }

    static status_t start_context(context_id ctx)
    {
        return rocprofiler_start_context(ctx);
    }

    static status_t stop_context(context_id ctx) { return rocprofiler_stop_context(ctx); }

    static status_t context_is_active(context_id ctx, int* out)
    {
        return rocprofiler_context_is_active(ctx, out);
    }

    static status_t context_is_valid(context_id ctx, int* out)
    {
        return rocprofiler_context_is_valid(ctx, out);
    }

    static status_t create_buffer(context_id ctx, size_t size, size_t watermark,
                                  buffer_policy_t policy, buffer_tracing_cb_t cb,
                                  void* cb_data, buffer_id* buf)
    {
        return rocprofiler_create_buffer(ctx, size, watermark, policy, cb, cb_data, buf);
    }

    static status_t destroy_buffer(buffer_id buf)
    {
        return rocprofiler_destroy_buffer(buf);
    }

    static status_t flush_buffer(buffer_id buf) { return rocprofiler_flush_buffer(buf); }

    static status_t create_callback_thread(callback_thread_id* thread)
    {
        return rocprofiler_create_callback_thread(thread);
    }

    static status_t assign_callback_thread(buffer_id buf, callback_thread_id thread)
    {
        return rocprofiler_assign_callback_thread(buf, thread);
    }

    static status_t query_available_agents(agent_version_t             version,
                                           query_available_agents_cb_t cb,
                                           size_t agent_size, void* user_data)
    {
        return rocprofiler_query_available_agents(version, cb, agent_size, user_data);
    }

    static status_t configure_callback_tracing_service(
        context_id ctx, callback_tracing_kind kind, tracing_operation* ops,
        size_t ops_count, callback_tracing_cb_t cb, void* cb_data)
    {
        return rocprofiler_configure_callback_tracing_service(ctx, kind, ops, ops_count,
                                                              cb, cb_data);
    }

    static status_t configure_buffer_tracing_service(context_id          ctx,
                                                     buffer_tracing_kind kind,
                                                     tracing_operation*  ops,
                                                     size_t ops_count, buffer_id buf)
    {
        return rocprofiler_configure_buffer_tracing_service(ctx, kind, ops, ops_count,
                                                            buf);
    }

    static status_t configure_external_correlation_id_request_service(
        context_id ctx, const external_correlation_request_kind* kinds, size_t count,
        external_correlation_id_request_cb_t cb, void* cb_data)
    {
        return rocprofiler_configure_external_correlation_id_request_service(
            ctx, kinds, count, cb, cb_data);
    }

    static status_t at_internal_thread_create(internal_thread_library_cb_t precreate,
                                              internal_thread_library_cb_t postcreate,
                                              runtime_library_t libs, void* user_data)
    {
        return rocprofiler_at_internal_thread_create(precreate, postcreate, libs,
                                                     user_data);
    }

    static status_t query_callback_op_name(callback_tracing_kind kind,
                                           tracing_operation op, const char** name,
                                           std::uint64_t* name_len)
    {
        return rocprofiler_query_callback_tracing_kind_operation_name(kind, op, name,
                                                                      name_len);
    }

    static status_t query_buffer_op_name(buffer_tracing_kind kind, tracing_operation op,
                                         const char** name, std::uint64_t* name_len)
    {
        return rocprofiler_query_buffer_tracing_kind_operation_name(kind, op, name,
                                                                    name_len);
    }

    static status_t iterate_callback_tracing_kind_operation_args(
        callback_tracing_record rec, callback_tracing_operation_args_cb_t cb,
        std::int32_t max_deref, void* user_data)
    {
        return rocprofiler_iterate_callback_tracing_kind_operation_args(
            rec, cb, max_deref, user_data);
    }

    static status_t iterate_agent_supported_counters(agent_id                id,
                                                     available_counters_cb_t cb,
                                                     void*                   user_data)
    {
        return rocprofiler_iterate_agent_supported_counters(id, cb, user_data);
    }

    static status_t iterate_counter_dimensions(counter_id                id,
                                               available_dimensions_cb_t cb,
                                               void*                     user_data)
    {
        return rocprofiler_iterate_counter_dimensions(id, cb, user_data);
    }

    static status_t query_counter_info(counter_id id, counter_info_version_id_t version,
                                       void* info)
    {
        return rocprofiler_query_counter_info(id, version, info);
    }

    static status_t query_record_counter_id(counter_instance_id_t id,
                                            counter_id*           counter_id_out)
    {
        return rocprofiler_query_record_counter_id(id, counter_id_out);
    }

    static status_t create_counter_config(agent_id id, counter_id* counters, size_t count,
                                          counter_config_id* config)
    {
#if ROCPROFILER_VERSION >= 10000
        return rocprofiler_create_counter_config(id, counters, count, config);
#else
        return rocprofiler_create_profile_config(id, counters, count, config);
#endif
    }

    static status_t configure_callback_dispatch_counting_service(
        context_id ctx, dispatch_counting_service_cb dispatch_cb, void* dispatch_data,
        dispatch_counting_record_cb record_cb, void* record_data)
    {
        return rocprofiler_configure_callback_dispatch_counting_service(
            ctx, dispatch_cb, dispatch_data, record_cb, record_data);
    }

    static status_t configure_device_counting_service(context_id ctx, buffer_id buf,
                                                      agent_id                     agent,
                                                      device_counting_service_cb_t cb,
                                                      void* user_data)
    {
        return rocprofiler_configure_device_counting_service(ctx, buf, agent, cb,
                                                             user_data);
    }

    static status_t sample_device_counting_service(context_id ctx, user_data_t user_data,
                                                   counter_flag_t  flags,
                                                   counter_record* output_records,
                                                   size_t*         rec_count)
    {
#if ROCPROFILER_VERSION >= 600
        return rocprofiler_sample_device_counting_service(ctx, user_data, flags,
                                                          output_records, rec_count);
#else
        // SDK < 0.6.0 (ROCm < 6.4) delivered records via the configured buffer callback;
        // the output_records/rec_count out-params did not exist yet.
        (void) output_records;
        (void) rec_count;
        return rocprofiler_sample_device_counting_service(ctx, user_data, flags);
#endif
    }

    // ─── Tracing name tables ──────────────────────────────────────────────────────
    // Name tables are exposed here so
    // rocprofsys::backends::rocprofiler_sdk::backend<Wrapper> (backend.hpp) routes all
    // SDK name-table access through this policy struct.

    using callback_name_info_t = rocprofiler::sdk::callback_name_info;
    using buffer_name_info_t   = rocprofiler::sdk::buffer_name_info;

    static callback_name_info_t get_callback_tracing_names()
    {
        return rocprofiler::sdk::get_callback_tracing_names();
    }

    static buffer_name_info_t get_buffer_tracing_names()
    {
        return rocprofiler::sdk::get_buffer_tracing_names();
    }
};

}  // namespace rocprofsys::rocprofiler_sdk
