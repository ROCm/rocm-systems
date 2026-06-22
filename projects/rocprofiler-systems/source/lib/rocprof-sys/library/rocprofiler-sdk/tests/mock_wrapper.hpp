// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gmock/gmock.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rocprofsys::mock::rocprofiler_sdk
{

// ─── Self-contained stub types ────────────────────────────────────────────────
//
// All types below replace rocprofiler-sdk C types.  No SDK header is included.
// Integer enum types carry the same integer values as the real SDK enums so
// that code comparing against backend::SOME_CONSTANT still works correctly.

struct handle_t
{
    std::uint64_t handle;
    bool          operator==(const handle_t&) const = default;
};

// Scalar alias types (replace rocprofiler_*_t enum typedefs)
using status_t                             = std::uint64_t;
using agent_type_t                         = int;
using agent_version_t                      = int;
using buffer_category_t                    = int;
using buffer_policy_t                      = int;
using runtime_library_t                    = int;
using callback_phase_t                     = int;
using tracing_operation                    = std::int32_t;
using callback_tracing_kind                = int;
using buffer_tracing_kind                  = int;
using external_correlation_request_kind    = int;
using counter_info_version_id_t            = int;
using scratch_memory_operation_t           = int;
using marker_op_t                          = int;
using marker_control_op_t                  = int;
using ompt_operation_t                     = int;
using rccl_api_id_t                        = int;
using nccl_data_type_t                     = int;
using nccl_result_t                        = int;
using hip_stream_operation_t               = int;
using kfd_page_fault_operation_t           = int;
using kfd_page_migrate_operation_t         = int;
using kfd_queue_operation_t                = int;
using kfd_event_queue_operation_t          = int;
using kfd_event_unmap_from_gpu_operation_t = int;
using counter_flag_t                       = int;

// Address type with .value member (replaces rocprofiler_address_t)
union address_t
{
    std::uint64_t handle{};
    std::uint64_t value;
    const void*   ptr;
};

// Agent stub (replaces rocprofiler_agent_t)
struct agent_t
{
    std::size_t  size = sizeof(agent_t);
    agent_type_t type = 0;
};

// Correlation ID stub (replaces rocprofiler_correlation_id_t)
struct correlation_id_t
{
    std::uint64_t internal = 0;
    struct
    {
        std::uint64_t value = 0;
    } external;
    std::uint64_t ancestor = 0;
};

// Record header stub (replaces rocprofiler_record_header_t)
struct record_header_t
{
    std::uint32_t category = 0;
    std::uint32_t kind     = 0;
    void*         payload  = nullptr;
};

// User data (struct instead of union for GMock copyability)
struct user_data_t
{
    std::uint64_t value                                = 0;
    bool          operator==(const user_data_t&) const = default;
};

using kernel_id_t = std::uint64_t;

// Dimension info stubs — field names match what backend.hpp accesses
struct dim_info_t
{
    const char*   dimension_name = nullptr;
    std::uint64_t index          = 0;
};

struct dim_instance_t
{
    std::uint64_t instance_id      = 0;
    std::uint64_t dimensions_count = 0;
    dim_info_t**  dimensions       = nullptr;
};

// Counter info stubs (match mock_sdk.hpp layout)
struct counter_info_v0_t
{
    const char* name        = nullptr;
    const char* description = nullptr;
    const char* block       = nullptr;
    const char* expression  = nullptr;
    int         is_constant = 0;
    int         is_derived  = 0;
};

struct counter_info_v1_t
{
    const char*      name                       = nullptr;
    const char*      description                = nullptr;
    const char*      block                      = nullptr;
    const char*      expression                 = nullptr;
    int              is_constant                = 0;
    int              is_derived                 = 0;
    std::uint64_t    dimensions_instances_count = 0;
    dim_instance_t** dimensions_instances       = nullptr;
};

// Counter record stub (replaces rocprofiler_counter_record_t)
struct counter_record_t
{
    std::uint64_t id            = 0;
    double        counter_value = 0.0;
    std::uint64_t dispatch_id   = 0;
};

// ─── KFD record stubs — only fields accessed by test/production code ──────────

struct kfd_page_fault_record_t
{
    std::uint64_t              size            = 0;
    buffer_tracing_kind        kind            = 0;
    kfd_page_fault_operation_t operation       = -1;
    std::uint64_t              start_timestamp = 0;
    std::uint64_t              end_timestamp   = 0;
    std::uint32_t              pid             = 0;
    handle_t                   agent_id        = {};
    address_t                  address         = {};
};

struct kfd_page_migrate_record_t
{
    std::uint64_t                size            = 0;
    buffer_tracing_kind          kind            = 0;
    kfd_page_migrate_operation_t operation       = -1;
    std::uint64_t                start_timestamp = 0;
    std::uint64_t                end_timestamp   = 0;
    std::uint32_t                pid             = 0;
    address_t                    start_address   = {};
    address_t                    end_address     = {};
    handle_t                     src_agent       = {};
    handle_t                     dst_agent       = {};
    handle_t                     prefetch_agent  = {};
    handle_t                     preferred_agent = {};
    std::int32_t                 error_code      = 0;
};

struct kfd_queue_record_t
{
    std::uint64_t         size            = 0;
    buffer_tracing_kind   kind            = 0;
    kfd_queue_operation_t operation       = -1;
    std::uint64_t         start_timestamp = 0;
    std::uint64_t         end_timestamp   = 0;
    std::uint32_t         pid             = 0;
    handle_t              agent_id        = {};
};

struct kfd_event_queue_record_t
{
    std::uint64_t               size      = 0;
    buffer_tracing_kind         kind      = 0;
    kfd_event_queue_operation_t operation = -1;
    std::uint64_t               timestamp = 0;
    std::uint32_t               pid       = 0;
    handle_t                    agent_id  = {};
};

struct kfd_event_unmap_record_t
{
    std::uint64_t                        size          = 0;
    buffer_tracing_kind                  kind          = 0;
    kfd_event_unmap_from_gpu_operation_t operation     = -1;
    std::uint64_t                        timestamp     = 0;
    std::uint32_t                        pid           = 0;
    handle_t                             agent_id      = {};
    address_t                            start_address = {};
    address_t                            end_address   = {};
};

struct kfd_event_dropped_record_t
{
    std::uint64_t       size      = 0;
    buffer_tracing_kind kind      = 0;
    int                 operation = -1;
    std::uint64_t       timestamp = 0;
    std::uint32_t       pid       = 0;
    std::uint32_t       count     = 0;
};

// Legacy page-migration record (ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION was removed
// in SDK 1.x; retained for source compatibility)
struct page_migration_record_t
{
    std::uint64_t                size            = 0;
    buffer_tracing_kind          kind            = 0;
    kfd_page_migrate_operation_t operation       = -1;
    std::uint64_t                start_timestamp = 0;
    std::uint64_t                end_timestamp   = 0;
    std::uint32_t                pid             = 0;
    address_t                    start_address   = {};
    address_t                    end_address     = {};
};

// Callback tracing record stub (replaces rocprofiler_callback_tracing_record_t)
struct callback_tracing_record_t
{
    handle_t              context_id     = {};
    std::uint64_t         thread_id      = 0;
    correlation_id_t      correlation_id = {};
    callback_tracing_kind kind           = 0;
    tracing_operation     operation      = 0;
    callback_phase_t      phase          = 0;
    void*                 payload        = nullptr;
};

// Opaque record stubs
struct kernel_dispatch_record_t
{
    std::uint64_t size = 0;
};
struct kernel_dispatch_data_t
{};
struct memory_copy_record_t
{};
struct scratch_memory_record_t
{};
struct code_object_load_data_t
{};
struct kernel_symbol_data_t
{};
struct marker_payload_t
{};
struct ompt_data_t
{};
struct rccl_api_data_t
{};
struct hip_stream_data_t
{};
struct memory_alloc_record_t
{};
struct dispatch_counting_data_t
{};

// Client types
struct client_id_t
{
    const char*   name = nullptr;
    std::uint64_t id   = 0;
};
using client_finalize_t = void (*)(client_id_t*, void*);
using client_detach_t   = void (*)(client_id_t*, void*);

// Opaque NCCL communicator (replaces ncclComm_t)
struct nccl_comm_struct
{};
using nccl_comm_t = nccl_comm_struct*;

// Callback / iterator function pointer types
using buffer_tracing_cb_t                  = void*;
using callback_tracing_cb_t                = void*;
using external_correlation_id_request_cb_t = void*;
using internal_thread_library_cb_t         = void*;
using query_available_agents_cb_t          = void*;
using callback_tracing_operation_args_cb_t = void*;
using available_counters_cb_t              = void*;
using available_dimensions_cb_t            = void*;
using device_counting_agent_cb_t           = void*;
using device_counting_service_cb_t         = void*;
using dispatch_counting_service_cb         = void*;
using dispatch_counting_record_cb          = void*;

// Name-info stub (return type for get_callback/buffer_tracing_names)
struct name_info_t
{};

// ─── gmock_wrapper ────────────────────────────────────────────────────────────

class gmock_wrapper
{
public:
    MOCK_METHOD(status_t, get_version, (std::uint32_t*, std::uint32_t*, std::uint32_t*) );
    MOCK_METHOD(status_t, get_timestamp, (std::uint64_t*) );
    MOCK_METHOD(const char*, get_status_string, (status_t));

    MOCK_METHOD(status_t, create_context, (handle_t*) );
    MOCK_METHOD(status_t, start_context, (handle_t));
    MOCK_METHOD(status_t, stop_context, (handle_t));
    MOCK_METHOD(status_t, context_is_active, (handle_t, int*) );
    MOCK_METHOD(status_t, context_is_valid, (handle_t, int*) );

    MOCK_METHOD(status_t, create_buffer,
                (handle_t, std::size_t, std::size_t, buffer_policy_t, buffer_tracing_cb_t,
                 void*, handle_t*) );
    MOCK_METHOD(status_t, destroy_buffer, (handle_t));
    MOCK_METHOD(status_t, flush_buffer, (handle_t));
    MOCK_METHOD(status_t, create_callback_thread, (handle_t*) );
    MOCK_METHOD(status_t, assign_callback_thread, (handle_t, handle_t));

    MOCK_METHOD(status_t, query_available_agents,
                (agent_version_t, query_available_agents_cb_t, std::size_t, void*) );

    MOCK_METHOD(status_t, configure_callback_tracing_service,
                (handle_t, callback_tracing_kind, tracing_operation*, std::size_t,
                 callback_tracing_cb_t, void*) );
    MOCK_METHOD(status_t, configure_buffer_tracing_service,
                (handle_t, buffer_tracing_kind, tracing_operation*, std::size_t,
                 handle_t));
    MOCK_METHOD(status_t, configure_external_correlation_id_request_service,
                (handle_t, const external_correlation_request_kind*, std::size_t,
                 external_correlation_id_request_cb_t, void*) );
    MOCK_METHOD(status_t, at_internal_thread_create,
                (internal_thread_library_cb_t, internal_thread_library_cb_t,
                 runtime_library_t, void*) );
    MOCK_METHOD(status_t, configure_callback_dispatch_counting_service,
                (handle_t, dispatch_counting_service_cb, void*,
                 dispatch_counting_record_cb, void*) );
    MOCK_METHOD(status_t, configure_device_counting_service,
                (handle_t, handle_t, handle_t, device_counting_service_cb_t, void*) );
    MOCK_METHOD(status_t, sample_device_counting_service,
                (handle_t, user_data_t, counter_flag_t, counter_record_t*,
                 std::size_t*) );

    MOCK_METHOD(status_t, iterate_agent_supported_counters,
                (handle_t, available_counters_cb_t, void*) );
    MOCK_METHOD(status_t, iterate_counter_dimensions,
                (handle_t, available_dimensions_cb_t, void*) );
    MOCK_METHOD(status_t, query_counter_info,
                (handle_t, counter_info_version_id_t, void*) );
    MOCK_METHOD(status_t, query_record_counter_id, (std::uint64_t, handle_t*) );
    MOCK_METHOD(status_t, create_counter_config,
                (handle_t, handle_t*, std::size_t, handle_t*) );

    MOCK_METHOD(status_t, query_callback_op_name,
                (callback_tracing_kind, tracing_operation, const char**,
                 std::uint64_t*) );
    MOCK_METHOD(status_t, query_buffer_op_name,
                (buffer_tracing_kind, tracing_operation, const char**, std::uint64_t*) );
    MOCK_METHOD(status_t, iterate_callback_tracing_kind_operation_args,
                (callback_tracing_record_t, callback_tracing_operation_args_cb_t,
                 std::int32_t, void*) );

    MOCK_METHOD(name_info_t, get_callback_tracing_names, ());
    MOCK_METHOD(name_info_t, get_buffer_tracing_names, ());
};

inline std::unique_ptr<gmock_wrapper> g_mock_wrapper;

// ─── backend ──────────────────────────────────────────────────────────────────

struct backend
{
    // ─── Compile-time SDK version ─────────────────────────────────────────────
    static constexpr std::uint32_t compile_time_version = 10301U;

    // ─── Scalar handle / ID types ─────────────────────────────────────────────
    using status_t = ::rocprofsys::mock::rocprofiler_sdk::status_t;

    // handle_t is defined at namespace scope; alias it here so backend::handle_t works.
    using handle_t = ::rocprofsys::mock::rocprofiler_sdk::handle_t;

    using context_id            = handle_t;
    using buffer_id             = handle_t;
    using agent_id              = handle_t;
    using queue_id              = handle_t;
    using timestamp_t           = std::uint64_t;
    using thread_id             = std::uint64_t;
    using counter_id            = handle_t;
    using dispatch_id_t         = std::uint64_t;
    using counter_instance_id_t = std::uint64_t;
    using callback_thread_id    = handle_t;

    // ─── Agent / device types ─────────────────────────────────────────────────
    using agent_t         = ::rocprofsys::mock::rocprofiler_sdk::agent_t;
    using agent_type_t    = ::rocprofsys::mock::rocprofiler_sdk::agent_type_t;
    using agent_version_t = ::rocprofsys::mock::rocprofiler_sdk::agent_version_t;

    // ─── Enum / flag types ────────────────────────────────────────────────────
    using buffer_category_t = ::rocprofsys::mock::rocprofiler_sdk::buffer_category_t;
    using buffer_policy_t   = ::rocprofsys::mock::rocprofiler_sdk::buffer_policy_t;
    using runtime_library_t = ::rocprofsys::mock::rocprofiler_sdk::runtime_library_t;
    using callback_phase_t  = ::rocprofsys::mock::rocprofiler_sdk::callback_phase_t;
    using tracing_operation = ::rocprofsys::mock::rocprofiler_sdk::tracing_operation;
    using callback_tracing_kind =
        ::rocprofsys::mock::rocprofiler_sdk::callback_tracing_kind;
    using buffer_tracing_kind = ::rocprofsys::mock::rocprofiler_sdk::buffer_tracing_kind;
    using external_correlation_request_kind =
        ::rocprofsys::mock::rocprofiler_sdk::external_correlation_request_kind;
    using counter_info_version_id_t =
        ::rocprofsys::mock::rocprofiler_sdk::counter_info_version_id_t;
    using counter_info_v0_t = ::rocprofsys::mock::rocprofiler_sdk::counter_info_v0_t;
    using counter_info_v1_t = ::rocprofsys::mock::rocprofiler_sdk::counter_info_v1_t;
    using scratch_memory_operation_t =
        ::rocprofsys::mock::rocprofiler_sdk::scratch_memory_operation_t;
    using marker_op_t         = ::rocprofsys::mock::rocprofiler_sdk::marker_op_t;
    using marker_control_op_t = ::rocprofsys::mock::rocprofiler_sdk::marker_control_op_t;

    // ─── Client / registration types ──────────────────────────────────────────
    using client_id_t       = ::rocprofsys::mock::rocprofiler_sdk::client_id_t;
    using client_finalize_t = ::rocprofsys::mock::rocprofiler_sdk::client_finalize_t;
    using client_detach_t   = ::rocprofsys::mock::rocprofiler_sdk::client_detach_t;

    // ─── Correlation types ────────────────────────────────────────────────────
    using correlation_id_t = ::rocprofsys::mock::rocprofiler_sdk::correlation_id_t;

    // ─── Buffer/callback tracing record types ─────────────────────────────────
    using record_header_t = ::rocprofsys::mock::rocprofiler_sdk::record_header_t;
    using user_data_t     = ::rocprofsys::mock::rocprofiler_sdk::user_data_t;
    using kernel_id_t     = ::rocprofsys::mock::rocprofiler_sdk::kernel_id_t;
    using kernel_dispatch_record =
        ::rocprofsys::mock::rocprofiler_sdk::kernel_dispatch_record_t;
    using kernel_dispatch_data =
        ::rocprofsys::mock::rocprofiler_sdk::kernel_dispatch_data_t;
    using dimension_info_t   = ::rocprofsys::mock::rocprofiler_sdk::dim_info_t;
    using memory_copy_record = ::rocprofsys::mock::rocprofiler_sdk::memory_copy_record_t;
    using scratch_memory_record =
        ::rocprofsys::mock::rocprofiler_sdk::scratch_memory_record_t;
    using callback_tracing_record =
        ::rocprofsys::mock::rocprofiler_sdk::callback_tracing_record_t;
    using code_object_load_data =
        ::rocprofsys::mock::rocprofiler_sdk::code_object_load_data_t;
    using kernel_symbol_data = ::rocprofsys::mock::rocprofiler_sdk::kernel_symbol_data_t;
    using marker_payload_t   = ::rocprofsys::mock::rocprofiler_sdk::marker_payload_t;

    // ─── Callback / iterator function pointer types ───────────────────────────
    using buffer_tracing_cb_t = ::rocprofsys::mock::rocprofiler_sdk::buffer_tracing_cb_t;
    using callback_tracing_cb_t =
        ::rocprofsys::mock::rocprofiler_sdk::callback_tracing_cb_t;
    using external_correlation_id_request_cb_t =
        ::rocprofsys::mock::rocprofiler_sdk::external_correlation_id_request_cb_t;
    using internal_thread_library_cb_t =
        ::rocprofsys::mock::rocprofiler_sdk::internal_thread_library_cb_t;
    using query_available_agents_cb_t =
        ::rocprofsys::mock::rocprofiler_sdk::query_available_agents_cb_t;
    using callback_tracing_operation_args_cb_t =
        ::rocprofsys::mock::rocprofiler_sdk::callback_tracing_operation_args_cb_t;
    using available_counters_cb_t =
        ::rocprofsys::mock::rocprofiler_sdk::available_counters_cb_t;
    using available_dimensions_cb_t =
        ::rocprofsys::mock::rocprofiler_sdk::available_dimensions_cb_t;
    using device_counting_agent_cb_t =
        ::rocprofsys::mock::rocprofiler_sdk::device_counting_agent_cb_t;
    using device_counting_service_cb_t =
        ::rocprofsys::mock::rocprofiler_sdk::device_counting_service_cb_t;
    using counter_flag_t = ::rocprofsys::mock::rocprofiler_sdk::counter_flag_t;

    using counter_config_id = handle_t;
    using counter_record    = ::rocprofsys::mock::rocprofiler_sdk::counter_record_t;
    using dispatch_counting_data =
        ::rocprofsys::mock::rocprofiler_sdk::dispatch_counting_data_t;
    using dispatch_counting_service_cb =
        ::rocprofsys::mock::rocprofiler_sdk::dispatch_counting_service_cb;
    using dispatch_counting_record_cb =
        ::rocprofsys::mock::rocprofiler_sdk::dispatch_counting_record_cb;

    using stream_id = handle_t;

    using page_migration_record =
        ::rocprofsys::mock::rocprofiler_sdk::page_migration_record_t;

    using ompt_data_t      = ::rocprofsys::mock::rocprofiler_sdk::ompt_data_t;
    using ompt_operation_t = ::rocprofsys::mock::rocprofiler_sdk::ompt_operation_t;

    // ─── OMPT operation constants ─────────────────────────────────────────────
    static constexpr ompt_operation_t OMPT_ID_NONE               = -1;
    static constexpr ompt_operation_t OMPT_ID_thread_begin       = 0;
    static constexpr ompt_operation_t OMPT_ID_thread_end         = 1;
    static constexpr ompt_operation_t OMPT_ID_parallel_begin     = 2;
    static constexpr ompt_operation_t OMPT_ID_parallel_end       = 3;
    static constexpr ompt_operation_t OMPT_ID_task_create        = 4;
    static constexpr ompt_operation_t OMPT_ID_task_schedule      = 5;
    static constexpr ompt_operation_t OMPT_ID_implicit_task      = 6;
    static constexpr ompt_operation_t OMPT_ID_device_initialize  = 7;
    static constexpr ompt_operation_t OMPT_ID_device_finalize    = 8;
    static constexpr ompt_operation_t OMPT_ID_device_load        = 9;
    static constexpr ompt_operation_t OMPT_ID_sync_region_wait   = 10;
    static constexpr ompt_operation_t OMPT_ID_mutex_released     = 11;
    static constexpr ompt_operation_t OMPT_ID_dependences        = 12;
    static constexpr ompt_operation_t OMPT_ID_task_dependence    = 13;
    static constexpr ompt_operation_t OMPT_ID_work               = 14;
    static constexpr ompt_operation_t OMPT_ID_masked             = 15;
    static constexpr ompt_operation_t OMPT_ID_sync_region        = 16;
    static constexpr ompt_operation_t OMPT_ID_lock_init          = 17;
    static constexpr ompt_operation_t OMPT_ID_lock_destroy       = 18;
    static constexpr ompt_operation_t OMPT_ID_mutex_acquire      = 19;
    static constexpr ompt_operation_t OMPT_ID_mutex_acquired     = 20;
    static constexpr ompt_operation_t OMPT_ID_nest_lock          = 21;
    static constexpr ompt_operation_t OMPT_ID_flush              = 22;
    static constexpr ompt_operation_t OMPT_ID_cancel             = 23;
    static constexpr ompt_operation_t OMPT_ID_reduction          = 24;
    static constexpr ompt_operation_t OMPT_ID_dispatch           = 25;
    static constexpr ompt_operation_t OMPT_ID_target_emi         = 26;
    static constexpr ompt_operation_t OMPT_ID_target_data_op_emi = 27;
    static constexpr ompt_operation_t OMPT_ID_target_submit_emi  = 28;
    static constexpr ompt_operation_t OMPT_ID_error              = 29;
    static constexpr ompt_operation_t OMPT_ID_callback_functions = 30;
    static constexpr ompt_operation_t OMPT_ID_LAST               = 31;

    using rccl_api_data    = ::rocprofsys::mock::rocprofiler_sdk::rccl_api_data_t;
    using rccl_api_id_t    = ::rocprofsys::mock::rocprofiler_sdk::rccl_api_id_t;
    using nccl_data_type_t = ::rocprofsys::mock::rocprofiler_sdk::nccl_data_type_t;
    using nccl_comm_t      = ::rocprofsys::mock::rocprofiler_sdk::nccl_comm_t;
    using nccl_result_t    = ::rocprofsys::mock::rocprofiler_sdk::nccl_result_t;

    // ─── NCCL data type constants ─────────────────────────────────────────────
    static constexpr nccl_result_t    NCCL_SUCCESS  = 0;
    static constexpr nccl_data_type_t NCCL_INT8     = 0;
    static constexpr nccl_data_type_t NCCL_UINT8    = 1;
    static constexpr nccl_data_type_t NCCL_FLOAT16  = 6;
    static constexpr nccl_data_type_t NCCL_BFLOAT16 = 9;
    static constexpr nccl_data_type_t NCCL_INT32    = 2;
    static constexpr nccl_data_type_t NCCL_UINT32   = 3;
    static constexpr nccl_data_type_t NCCL_FLOAT32  = 7;
    static constexpr nccl_data_type_t NCCL_INT64    = 4;
    static constexpr nccl_data_type_t NCCL_UINT64   = 5;
    static constexpr nccl_data_type_t NCCL_FLOAT64  = 8;
    static constexpr nccl_data_type_t NCCL_FP8_E4M3 = 10;
    static constexpr nccl_data_type_t NCCL_FP8_E5M2 = 11;

    // ─── RCCL API ID constants ────────────────────────────────────────────────
    static constexpr rccl_api_id_t RCCL_API_ID_ncclAllGather     = 0;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclAllToAll      = 2;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclAllReduce     = 1;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclGather        = 5;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclRecv          = 10;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclReduce        = 6;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclBroadcast     = 4;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclReduceScatter = 7;
    static constexpr rccl_api_id_t RCCL_API_ID_ncclSend          = 9;

    using memory_alloc_record =
        ::rocprofsys::mock::rocprofiler_sdk::memory_alloc_record_t;
    using hip_stream_data = ::rocprofsys::mock::rocprofiler_sdk::hip_stream_data_t;
    using hip_stream_operation_t =
        ::rocprofsys::mock::rocprofiler_sdk::hip_stream_operation_t;

    using kfd_page_fault_record =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_page_fault_record_t;
    using kfd_page_migrate_record =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_page_migrate_record_t;
    using kfd_queue_record = ::rocprofsys::mock::rocprofiler_sdk::kfd_queue_record_t;
    using kfd_event_queue_record =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_event_queue_record_t;
    using kfd_event_unmap_record =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_event_unmap_record_t;
    using kfd_event_dropped_record =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_event_dropped_record_t;
    using kfd_event_queue_operation_t =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_event_queue_operation_t;
    using kfd_event_unmap_from_gpu_operation_t =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_event_unmap_from_gpu_operation_t;
    using kfd_page_fault_operation_t =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_page_fault_operation_t;
    using kfd_page_migrate_operation_t =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_page_migrate_operation_t;
    using kfd_queue_operation_t =
        ::rocprofsys::mock::rocprofiler_sdk::kfd_queue_operation_t;

    // ─── Status constants ─────────────────────────────────────────────────────
    static constexpr status_t STATUS_SUCCESS                = 0;
    static constexpr status_t STATUS_ERROR                  = 1;
    static constexpr status_t STATUS_ERROR_BUFFER_BUSY      = 14;
    static constexpr status_t STATUS_ERROR_HSA_NOT_LOADED   = 22;
    static constexpr status_t STATUS_ERROR_INVALID_ARGUMENT = 19;

    // ─── Callback phase constants ─────────────────────────────────────────────
    static constexpr callback_phase_t CALLBACK_PHASE_ENTER = 1;
    static constexpr callback_phase_t CALLBACK_PHASE_EXIT  = 2;
    static constexpr callback_phase_t CALLBACK_PHASE_NONE  = 0;

    // ─── Agent type constants ─────────────────────────────────────────────────
    static constexpr agent_type_t AGENT_TYPE_CPU = 1;
    static constexpr agent_type_t AGENT_TYPE_GPU = 2;

    // ─── Agent version constants ──────────────────────────────────────────────
    static constexpr agent_version_t AGENT_INFO_VERSION_0 = 1;

    // ─── Buffer category constants ────────────────────────────────────────────
    static constexpr buffer_category_t BUFFER_CATEGORY_TRACING = 1;
    static constexpr buffer_policy_t   BUFFER_POLICY_LOSSLESS  = 2;

    // ─── Runtime library flag constants ───────────────────────────────────────
    static constexpr runtime_library_t LIBRARY        = 1;
    static constexpr runtime_library_t HSA_LIBRARY    = 2;
    static constexpr runtime_library_t HIP_LIBRARY    = 4;
    static constexpr runtime_library_t MARKER_LIBRARY = 8;

    // ─── Callback tracing kind constants ──────────────────────────────────────
    static constexpr callback_tracing_kind CALLBACK_TRACING_NONE                 = 0;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HSA_CORE_API         = 1;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HSA_AMD_EXT_API      = 2;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HSA_IMAGE_EXT_API    = 3;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HSA_FINALIZE_EXT_API = 4;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HIP_RUNTIME_API      = 5;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HIP_COMPILER_API     = 6;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MARKER_CORE_API      = 7;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MARKER_CONTROL_API   = 8;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MARKER_NAME_API      = 9;
    static constexpr callback_tracing_kind CALLBACK_TRACING_CODE_OBJECT          = 10;
    static constexpr callback_tracing_kind CALLBACK_TRACING_SCRATCH_MEMORY       = 11;
    static constexpr callback_tracing_kind CALLBACK_TRACING_KERNEL_DISPATCH      = 12;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MEMORY_COPY          = 13;
    static constexpr callback_tracing_kind CALLBACK_TRACING_RCCL_API             = 14;
    static constexpr callback_tracing_kind CALLBACK_TRACING_LAST                 = 22;

    static constexpr callback_tracing_kind CALLBACK_TRACING_ROCDECODE_API          = 18;
    static constexpr callback_tracing_kind CALLBACK_TRACING_OMPT                   = 15;
    static constexpr callback_tracing_kind CALLBACK_TRACING_MEMORY_ALLOCATION      = 16;
    static constexpr callback_tracing_kind CALLBACK_TRACING_RUNTIME_INITIALIZATION = 17;

    static constexpr callback_tracing_kind CALLBACK_TRACING_ROCJPEG_API = 19;
    static constexpr callback_tracing_kind CALLBACK_TRACING_HIP_STREAM  = 20;

    // ─── Buffer tracing kind constants ────────────────────────────────────────
    static constexpr buffer_tracing_kind BUFFER_TRACING_HSA_CORE_API         = 1;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HSA_AMD_EXT_API      = 2;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HSA_IMAGE_EXT_API    = 3;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HSA_FINALIZE_EXT_API = 4;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HIP_RUNTIME_API      = 5;
    static constexpr buffer_tracing_kind BUFFER_TRACING_HIP_COMPILER_API     = 6;
    static constexpr buffer_tracing_kind BUFFER_TRACING_MARKER_CORE_API      = 7;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KERNEL_DISPATCH      = 11;
    static constexpr buffer_tracing_kind BUFFER_TRACING_MEMORY_COPY          = 10;
    static constexpr buffer_tracing_kind BUFFER_TRACING_SCRATCH_MEMORY       = 12;

    // Legacy combined page-migration (removed in SDK 1.x; assigned non-conflicting value)
    static constexpr buffer_tracing_kind BUFFER_TRACING_PAGE_MIGRATION = 50;

    static constexpr buffer_tracing_kind BUFFER_TRACING_MEMORY_ALLOCATION = 16;

    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_PAGE_FAULT           = 30;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_PAGE_MIGRATE         = 29;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_QUEUE                = 31;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_EVENT_QUEUE          = 26;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_EVENT_UNMAP_FROM_GPU = 27;
    static constexpr buffer_tracing_kind BUFFER_TRACING_KFD_EVENT_DROPPED_EVENTS = 28;

    // ─── Counter flag constants ───────────────────────────────────────────────
    static constexpr counter_flag_t COUNTER_FLAG_NONE = 0;

    // ─── Counter info version constants ───────────────────────────────────────
    static constexpr counter_info_version_id_t COUNTER_INFO_VERSION_0 = 1;
    static constexpr counter_info_version_id_t COUNTER_INFO_VERSION_1 = 2;

    // ─── Scratch memory operation constants ───────────────────────────────────
    static constexpr scratch_memory_operation_t SCRATCH_MEMORY_ALLOC = 1;

    // ─── Code object tracing operation constants ──────────────────────────────
    static constexpr tracing_operation CODE_OBJECT_LOAD                          = 1;
    static constexpr tracing_operation CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER = 2;

    // ─── Kernel dispatch operation constants ──────────────────────────────────
    static constexpr tracing_operation KERNEL_DISPATCH_COMPLETE = 2;

    // ─── External correlation request kind constants ───────────────────────────
    static constexpr external_correlation_request_kind
        EXTERNAL_CORRELATION_REQUEST_KERNEL_DISPATCH = 11;
    static constexpr external_correlation_request_kind
        EXTERNAL_CORRELATION_REQUEST_MEMORY_COPY = 10;
    static constexpr external_correlation_request_kind
        EXTERNAL_CORRELATION_REQUEST_MEMORY_ALLOCATION = 15;

    // ─── Marker API operation constants ───────────────────────────────────────
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxMarkA       = 0;
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxRangePushA  = 1;
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxRangePop    = 2;
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxRangeStartA = 3;
    static constexpr marker_op_t MARKER_CORE_API_ID_roctxRangeStop   = 4;

    // ─── Marker control operation constants ───────────────────────────────────
    static constexpr marker_control_op_t MARKER_CONTROL_API_ID_roctxProfilerPause  = 0;
    static constexpr marker_control_op_t MARKER_CONTROL_API_ID_roctxProfilerResume = 1;

    // ─── HIP stream operation constants ───────────────────────────────────────
    static constexpr hip_stream_operation_t HIP_STREAM_CREATE  = 1;
    static constexpr hip_stream_operation_t HIP_STREAM_DESTROY = 2;
    static constexpr hip_stream_operation_t HIP_STREAM_SET     = 3;

    // ─── KFD event queue operation constants ──────────────────────────────────
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_NONE          = -1;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_SVM     = 0;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_USERPTR = 1;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_TTM     = 2;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_SUSPEND = 3;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_CRIU_CHECKPOINT =
        4;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_EVICT_CRIU_RESTORE  = 5;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_RESTORE_RESCHEDULED = 6;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_RESTORE             = 7;
    static constexpr kfd_event_queue_operation_t KFD_EVENT_QUEUE_LAST                = 8;

    // ─── KFD unmap-from-GPU operation constants ───────────────────────────────
    static constexpr kfd_event_unmap_from_gpu_operation_t KFD_EVENT_UNMAP_FROM_GPU_NONE =
        -1;
    static constexpr kfd_event_unmap_from_gpu_operation_t
        KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY = 0;
    static constexpr kfd_event_unmap_from_gpu_operation_t
        KFD_EVENT_UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE = 1;
    static constexpr kfd_event_unmap_from_gpu_operation_t
        KFD_EVENT_UNMAP_FROM_GPU_UNMAP_FROM_CPU = 2;
    static constexpr kfd_event_unmap_from_gpu_operation_t KFD_EVENT_UNMAP_FROM_GPU_LAST =
        3;

    // ─── KFD page fault operation constants ───────────────────────────────────
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_NONE                 = -1;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_READ_FAULT_MIGRATED  = 0;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_READ_FAULT_UPDATED   = 1;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_WRITE_FAULT_MIGRATED = 2;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_WRITE_FAULT_UPDATED  = 3;
    static constexpr kfd_page_fault_operation_t KFD_PAGE_FAULT_LAST                 = 4;

    // ─── KFD page migrate operation constants ─────────────────────────────────
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_NONE          = -1;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_PREFETCH      = 0;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_PAGEFAULT_GPU = 1;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_PAGEFAULT_CPU = 2;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_TTM_EVICTION  = 3;
    static constexpr kfd_page_migrate_operation_t KFD_PAGE_MIGRATE_LAST          = 4;

    // ─── KFD queue evict operation constants ──────────────────────────────────
    static constexpr kfd_queue_operation_t KFD_QUEUE_NONE                  = -1;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_SVM             = 0;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_USERPTR         = 1;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_TTM             = 2;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_SUSPEND         = 3;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_CRIU_CHECKPOINT = 4;
    static constexpr kfd_queue_operation_t KFD_QUEUE_EVICT_CRIU_RESTORE    = 5;
    static constexpr kfd_queue_operation_t KFD_QUEUE_LAST                  = 6;

    // ─── SDK function wrappers — all delegate to g_mock_wrapper ──────────────

    static status_t get_version(std::uint32_t* major, std::uint32_t* minor,
                                std::uint32_t* patch)
    {
        return g_mock_wrapper->get_version(major, minor, patch);
    }

    static status_t get_timestamp(timestamp_t* ts) noexcept
    {
        return g_mock_wrapper->get_timestamp(ts);
    }

    static const char* get_status_string(status_t status) noexcept
    {
        return g_mock_wrapper->get_status_string(status);
    }

    static status_t create_context(context_id* ctx)
    {
        return g_mock_wrapper->create_context(ctx);
    }

    static status_t start_context(context_id ctx)
    {
        return g_mock_wrapper->start_context(ctx);
    }

    static status_t stop_context(context_id ctx)
    {
        return g_mock_wrapper->stop_context(ctx);
    }

    static status_t context_is_active(context_id ctx, int* out)
    {
        return g_mock_wrapper->context_is_active(ctx, out);
    }

    static status_t context_is_valid(context_id ctx, int* out)
    {
        return g_mock_wrapper->context_is_valid(ctx, out);
    }

    static status_t create_buffer(context_id ctx, size_t size, size_t watermark,
                                  buffer_policy_t policy, buffer_tracing_cb_t cb,
                                  void* cb_data, buffer_id* buf)
    {
        return g_mock_wrapper->create_buffer(ctx, size, watermark, policy, cb, cb_data,
                                             buf);
    }

    static status_t destroy_buffer(buffer_id buf)
    {
        return g_mock_wrapper->destroy_buffer(buf);
    }

    static status_t flush_buffer(buffer_id buf)
    {
        return g_mock_wrapper->flush_buffer(buf);
    }

    static status_t create_callback_thread(callback_thread_id* thread)
    {
        return g_mock_wrapper->create_callback_thread(thread);
    }

    static status_t assign_callback_thread(buffer_id buf, callback_thread_id thread)
    {
        return g_mock_wrapper->assign_callback_thread(buf, thread);
    }

    static status_t query_available_agents(agent_version_t             version,
                                           query_available_agents_cb_t cb,
                                           size_t agent_size, void* user_data)
    {
        return g_mock_wrapper->query_available_agents(version, cb, agent_size, user_data);
    }

    static status_t configure_callback_tracing_service(
        context_id ctx, callback_tracing_kind kind, tracing_operation* ops,
        size_t ops_count, callback_tracing_cb_t cb, void* cb_data)
    {
        return g_mock_wrapper->configure_callback_tracing_service(ctx, kind, ops,
                                                                  ops_count, cb, cb_data);
    }

    static status_t configure_buffer_tracing_service(context_id          ctx,
                                                     buffer_tracing_kind kind,
                                                     tracing_operation*  ops,
                                                     size_t ops_count, buffer_id buf)
    {
        return g_mock_wrapper->configure_buffer_tracing_service(ctx, kind, ops, ops_count,
                                                                buf);
    }

    static status_t configure_external_correlation_id_request_service(
        context_id ctx, const external_correlation_request_kind* kinds, size_t count,
        external_correlation_id_request_cb_t cb, void* cb_data)
    {
        return g_mock_wrapper->configure_external_correlation_id_request_service(
            ctx, kinds, count, cb, cb_data);
    }

    static status_t at_internal_thread_create(internal_thread_library_cb_t precreate,
                                              internal_thread_library_cb_t postcreate,
                                              runtime_library_t libs, void* user_data)
    {
        return g_mock_wrapper->at_internal_thread_create(precreate, postcreate, libs,
                                                         user_data);
    }

    static status_t query_callback_op_name(callback_tracing_kind kind,
                                           tracing_operation op, const char** name,
                                           std::uint64_t* name_len)
    {
        return g_mock_wrapper->query_callback_op_name(kind, op, name, name_len);
    }

    static status_t query_buffer_op_name(buffer_tracing_kind kind, tracing_operation op,
                                         const char** name, std::uint64_t* name_len)
    {
        return g_mock_wrapper->query_buffer_op_name(kind, op, name, name_len);
    }

    static status_t iterate_callback_tracing_kind_operation_args(
        callback_tracing_record rec, callback_tracing_operation_args_cb_t cb,
        std::int32_t max_deref, void* user_data)
    {
        return g_mock_wrapper->iterate_callback_tracing_kind_operation_args(
            rec, cb, max_deref, user_data);
    }

    static status_t iterate_agent_supported_counters(agent_id                id,
                                                     available_counters_cb_t cb,
                                                     void*                   user_data)
    {
        return g_mock_wrapper->iterate_agent_supported_counters(id, cb, user_data);
    }

    static status_t iterate_counter_dimensions(counter_id                id,
                                               available_dimensions_cb_t cb,
                                               void*                     user_data)
    {
        return g_mock_wrapper->iterate_counter_dimensions(id, cb, user_data);
    }

    static status_t query_counter_info(counter_id id, counter_info_version_id_t version,
                                       void* info)
    {
        return g_mock_wrapper->query_counter_info(id, version, info);
    }

    static status_t query_record_counter_id(counter_instance_id_t id,
                                            counter_id*           counter_id_out)
    {
        return g_mock_wrapper->query_record_counter_id(id, counter_id_out);
    }

    static status_t create_counter_config(agent_id id, counter_id* counters, size_t count,
                                          counter_config_id* config)
    {
        return g_mock_wrapper->create_counter_config(id, counters, count, config);
    }

    static status_t configure_callback_dispatch_counting_service(
        context_id ctx, dispatch_counting_service_cb dispatch_cb, void* dispatch_data,
        dispatch_counting_record_cb record_cb, void* record_data)
    {
        return g_mock_wrapper->configure_callback_dispatch_counting_service(
            ctx, dispatch_cb, dispatch_data, record_cb, record_data);
    }

    static status_t configure_device_counting_service(context_id ctx, buffer_id buf,
                                                      agent_id                     agent,
                                                      device_counting_service_cb_t cb,
                                                      void* user_data)
    {
        return g_mock_wrapper->configure_device_counting_service(ctx, buf, agent, cb,
                                                                 user_data);
    }

    static status_t sample_device_counting_service(context_id ctx, user_data_t user_data,
                                                   counter_flag_t  flags,
                                                   counter_record* output_records,
                                                   size_t*         rec_count)
    {
        return g_mock_wrapper->sample_device_counting_service(ctx, user_data, flags,
                                                              output_records, rec_count);
    }

    static name_info_t get_callback_tracing_names()
    {
        return g_mock_wrapper->get_callback_tracing_names();
    }

    static name_info_t get_buffer_tracing_names()
    {
        return g_mock_wrapper->get_buffer_tracing_names();
    }
};

}  // namespace rocprofsys::mock::rocprofiler_sdk
