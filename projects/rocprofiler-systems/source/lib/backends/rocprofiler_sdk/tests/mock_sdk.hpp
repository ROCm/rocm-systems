// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gmock/gmock.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rocprofsys::backends::rocprofiler_sdk::testing
{

// ─── Stub types ───────────────────────────────────────────────────────────────
//
// Minimal stand-ins for the rocprofiler-sdk C types.  They satisfy every
// declaration in backend<Sdk> without pulling in any ROCm headers.
//
// Note: backend<Sdk> method bodies are only instantiated when called.
// Types used exclusively inside uninstantiated bodies (counter_info_v0/v1_t,
// compile_time_version, COUNTER_INFO_VERSION_*) are intentionally omitted.

using status_t                                            = int;
static constexpr status_t k_status_success                = 0;
static constexpr status_t k_status_error                  = -1;
static constexpr status_t k_status_buffer_busy            = -2;
static constexpr status_t k_status_hsa_not_loaded         = -3;
static constexpr status_t k_status_error_invalid_argument = -4;

struct context_id
{
    std::uint64_t handle{};
    bool          operator==(const context_id&) const = default;
};
struct buffer_id
{
    std::uint64_t handle{};
    bool          operator==(const buffer_id&) const = default;
};
struct agent_id
{
    std::uint64_t handle{};
    bool          operator==(const agent_id&) const = default;
};
struct counter_id
{
    std::uint64_t handle{};
    bool          operator==(const counter_id&) const = default;
};
struct counter_config_id
{
    std::uint64_t handle{};
};
struct callback_thread_id
{
    std::uint64_t handle{};
    bool          operator==(const callback_thread_id&) const = default;
};
// Real SDK defines rocprofiler_counter_instance_id_t as plain std::uint64_t;
// match that so backend<mock_sdk> type-checks cleanly.
using counter_instance_id = std::uint64_t;
struct counter_record
{
    counter_instance_id id;
    double              counter_value{};
    bool                operator==(const counter_record&) const = default;
};
struct user_data
{
    std::uint64_t value{};
    bool          operator==(const user_data&) const = default;
};
struct timestamp
{
    std::uint64_t value{};
    bool          operator==(const timestamp&) const = default;
};

using counter_flag_t   = std::uint32_t;
using runtime_library  = std::uint32_t;
using buffer_policy    = std::uint32_t;
using tracing_kind_cb  = int;
using tracing_kind_buf = int;
using tracing_op       = int;
using ext_corr_kind    = int;
using counter_info_ver = int;

// Stubs for query_counter_details — match the field layout backend.hpp accesses.
struct dim_info
{
    const char*   dimension_name = nullptr;
    std::uint64_t index          = 0;
};

struct dim_instance
{
    std::uint64_t instance_id      = 0;
    std::uint64_t dimensions_count = 0;
    dim_info**    dimensions       = nullptr;
};

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
    const char*    name                       = nullptr;
    const char*    description                = nullptr;
    const char*    block                      = nullptr;
    const char*    expression                 = nullptr;
    int            is_constant                = 0;
    int            is_derived                 = 0;
    std::uint64_t  dimensions_instances_count = 0;
    dim_instance** dimensions_instances       = nullptr;
};

// Callback function pointer types — tests never invoke these, so void* suffices.
using buffer_tracing_cb_t        = void*;
using callback_tracing_cb_t      = void*;
using ext_correlation_req_cb_t   = void*;
using internal_thread_cb_t       = void*;
using tracing_op_args_cb_t       = void*;
using available_counters_cb_t    = void*;
using available_dimensions_cb_t  = void*;
using device_counting_agent_cb_t = void*;
using device_counting_svc_cb_t   = void*;
using dispatch_counting_svc_cb   = void*;
using dispatch_counting_rec_cb   = void*;

struct callback_tracing_record_t
{};

// ─── gmock_sdk ────────────────────────────────────────────────────────────────
//
// Non-static GMock class.  mock_sdk's static stubs delegate here so
// EXPECT_CALL can observe and control every SDK call.

class gmock_sdk
{
public:
    MOCK_METHOD(status_t, create_context, (context_id * ctx));
    MOCK_METHOD(status_t, start_context, (context_id ctx));
    MOCK_METHOD(status_t, stop_context, (context_id ctx));

    MOCK_METHOD(status_t, sample_device_counting_service,
                (context_id ctx, user_data ud, counter_flag_t flags,
                 counter_record* output_records, size_t* record_count));

    MOCK_METHOD(status_t, iterate_agent_supported_counters,
                (agent_id ag, available_counters_cb_t cb, void* user_data));

    MOCK_METHOD(status_t, create_counter_config,
                (agent_id ag, counter_id* counters, size_t count,
                 counter_config_id* config));

    MOCK_METHOD(status_t, configure_device_counting_service,
                (context_id ctx, buffer_id buf, agent_id ag, device_counting_svc_cb_t cb,
                 void* user_data));

    MOCK_METHOD(status_t, query_record_counter_id,
                (counter_instance_id id, counter_id* out));

    MOCK_METHOD(status_t, query_counter_info,
                (counter_id id, counter_info_ver version, void* info));

    MOCK_METHOD(status_t, create_buffer,
                (context_id ctx, size_t size, size_t watermark, buffer_policy policy,
                 buffer_tracing_cb_t cb, void* cb_data, buffer_id* buf));

    MOCK_METHOD(status_t, flush_buffer, (buffer_id buf));
    MOCK_METHOD(status_t, destroy_buffer, (buffer_id buf));

    MOCK_METHOD(status_t, create_callback_thread, (callback_thread_id * thread));
    MOCK_METHOD(status_t, assign_callback_thread,
                (buffer_id buf, callback_thread_id thread));

    MOCK_METHOD(status_t, configure_callback_tracing_service,
                (context_id ctx, tracing_kind_cb kind, tracing_op* ops, size_t ops_count,
                 callback_tracing_cb_t cb, void* cb_data));

    MOCK_METHOD(status_t, configure_buffer_tracing_service,
                (context_id ctx, tracing_kind_buf kind, tracing_op* ops, size_t ops_count,
                 buffer_id buf));

    MOCK_METHOD(status_t, configure_external_correlation_id_request_service,
                (context_id ctx, const ext_corr_kind* kinds, size_t count,
                 ext_correlation_req_cb_t cb, void* cb_data));

    MOCK_METHOD(status_t, configure_callback_dispatch_counting_service,
                (context_id ctx, dispatch_counting_svc_cb dispatch_cb,
                 void* dispatch_data, dispatch_counting_rec_cb record_cb,
                 void* record_data));

    MOCK_METHOD(status_t, at_internal_thread_create,
                (internal_thread_cb_t pre, internal_thread_cb_t post,
                 runtime_library libs, void* user_data));

    MOCK_METHOD(status_t, context_is_active, (context_id ctx, int* out));
    MOCK_METHOD(status_t, context_is_valid, (context_id ctx, int* out));

    MOCK_METHOD(status_t, query_callback_op_name,
                (tracing_kind_cb kind, tracing_op op, const char** name,
                 std::uint64_t* name_len));

    MOCK_METHOD(status_t, query_buffer_op_name,
                (tracing_kind_buf kind, tracing_op op, const char** name,
                 std::uint64_t* name_len));

    MOCK_METHOD(status_t, iterate_callback_tracing_kind_operation_args,
                (callback_tracing_record_t rec, tracing_op_args_cb_t cb,
                 std::int32_t max_deref, void* user_data));

    MOCK_METHOD(status_t, iterate_counter_dimensions,
                (counter_id id, available_dimensions_cb_t cb, void* user_data));

    MOCK_METHOD(status_t, get_version,
                (std::uint32_t * major, std::uint32_t* minor, std::uint32_t* patch));
    MOCK_METHOD(status_t, get_timestamp, (timestamp * ts));
    MOCK_METHOD(const char*, get_status_string, (status_t s));
};

// Global singleton — GMock objects are non-copyable, so they live on the heap.
//
// SAFETY: Every static stub in mock_sdk dereferences g_mock_sdk without a null
// check.  Always use this inside a backend_test fixture whose SetUp() sets
// g_mock_sdk and TearDown() resets it.  Calling any backend<mock_sdk> function
// outside that fixture will crash with a null pointer dereference.
inline std::unique_ptr<gmock_sdk> g_mock_sdk;

// ─── mock_sdk ─────────────────────────────────────────────────────────────────
//
// The Sdk policy type for backend<mock_sdk>.  Exposes stub type aliases,
// compile-time constants, and static stubs that forward every SDK call to
// g_mock_sdk so EXPECT_CALL can intercept them.

struct mock_sdk
{
    // ── Type aliases (match the names backend<Sdk> uses as Sdk::name) ────────
    using status_t                             = testing::status_t;
    using context_id                           = testing::context_id;
    using agent_id                             = testing::agent_id;
    using buffer_id                            = testing::buffer_id;
    using counter_id                           = testing::counter_id;
    using counter_config_id                    = testing::counter_config_id;
    using counter_record                       = testing::counter_record;
    using counter_instance_id_t                = testing::counter_instance_id;
    using counter_flag_t                       = testing::counter_flag_t;
    using user_data_t                          = testing::user_data;
    using timestamp_t                          = testing::timestamp;
    using available_counters_cb_t              = testing::available_counters_cb_t;
    using device_counting_agent_cb_t           = testing::device_counting_agent_cb_t;
    using device_counting_service_cb_t         = testing::device_counting_svc_cb_t;
    using buffer_policy_t                      = testing::buffer_policy;
    using buffer_tracing_cb_t                  = testing::buffer_tracing_cb_t;
    using callback_tracing_cb_t                = testing::callback_tracing_cb_t;
    using callback_tracing_kind                = testing::tracing_kind_cb;
    using buffer_tracing_kind                  = testing::tracing_kind_buf;
    using tracing_operation                    = testing::tracing_op;
    using callback_thread_id                   = testing::callback_thread_id;
    using runtime_library_t                    = testing::runtime_library;
    using external_correlation_request_kind    = testing::ext_corr_kind;
    using external_correlation_id_request_cb_t = testing::ext_correlation_req_cb_t;
    using internal_thread_library_cb_t         = testing::internal_thread_cb_t;
    using callback_tracing_record              = testing::callback_tracing_record_t;
    using callback_tracing_operation_args_cb_t = testing::tracing_op_args_cb_t;
    using available_dimensions_cb_t            = testing::available_dimensions_cb_t;
    using counter_info_version_id_t            = testing::counter_info_ver;
    using counter_info_v0_t                    = testing::counter_info_v0_t;
    using counter_info_v1_t                    = testing::counter_info_v1_t;
    using dispatch_counting_service_cb         = testing::dispatch_counting_svc_cb;
    using dispatch_counting_record_cb          = testing::dispatch_counting_rec_cb;

    // compile_time_version >= 10000 selects the v1 branch in query_counter_details.
    static constexpr std::uint32_t compile_time_version = 10100u;

    // ── Status constants ──────────────────────────────────────────────────────
    static constexpr status_t STATUS_SUCCESS              = k_status_success;
    static constexpr status_t STATUS_ERROR                = k_status_error;
    static constexpr status_t STATUS_ERROR_BUFFER_BUSY    = k_status_buffer_busy;
    static constexpr status_t STATUS_ERROR_HSA_NOT_LOADED = k_status_hsa_not_loaded;
    static constexpr status_t STATUS_ERROR_INVALID_ARGUMENT =
        k_status_error_invalid_argument;

    // ── Counter constants ─────────────────────────────────────────────────────
    static constexpr counter_flag_t            COUNTER_FLAG_NONE      = 0;
    static constexpr counter_info_version_id_t COUNTER_INFO_VERSION_0 = 0;
    static constexpr counter_info_version_id_t COUNTER_INFO_VERSION_1 = 1;

    // ── Static forwarding stubs ───────────────────────────────────────────────

    static status_t create_context(context_id* ctx)
    {
        return g_mock_sdk->create_context(ctx);
    }

    static status_t start_context(context_id ctx)
    {
        return g_mock_sdk->start_context(ctx);
    }

    static status_t stop_context(context_id ctx) { return g_mock_sdk->stop_context(ctx); }

    static status_t sample_device_counting_service(context_id ctx, user_data_t ud,
                                                   counter_flag_t  flags,
                                                   counter_record* output_records,
                                                   size_t*         record_count)
    {
        return g_mock_sdk->sample_device_counting_service(ctx, ud, flags, output_records,
                                                          record_count);
    }

    static status_t iterate_agent_supported_counters(agent_id                ag,
                                                     available_counters_cb_t cb,
                                                     void*                   user_data)
    {
        return g_mock_sdk->iterate_agent_supported_counters(ag, cb, user_data);
    }

    static status_t create_counter_config(agent_id ag, counter_id* counters, size_t count,
                                          counter_config_id* config)
    {
        return g_mock_sdk->create_counter_config(ag, counters, count, config);
    }

    static status_t configure_device_counting_service(context_id ctx, buffer_id buf,
                                                      agent_id                     ag,
                                                      device_counting_service_cb_t cb,
                                                      void* user_data)
    {
        return g_mock_sdk->configure_device_counting_service(ctx, buf, ag, cb, user_data);
    }

    static status_t query_record_counter_id(counter_instance_id_t id, counter_id* out)
    {
        return g_mock_sdk->query_record_counter_id(id, out);
    }

    static status_t query_counter_info(counter_id id, counter_info_version_id_t version,
                                       void* info)
    {
        return g_mock_sdk->query_counter_info(id, version, info);
    }

    static status_t create_buffer(context_id ctx, size_t size, size_t watermark,
                                  buffer_policy_t policy, buffer_tracing_cb_t cb,
                                  void* cb_data, buffer_id* buf)
    {
        return g_mock_sdk->create_buffer(ctx, size, watermark, policy, cb, cb_data, buf);
    }

    static status_t flush_buffer(buffer_id buf) { return g_mock_sdk->flush_buffer(buf); }

    static status_t destroy_buffer(buffer_id buf)
    {
        return g_mock_sdk->destroy_buffer(buf);
    }

    static status_t create_callback_thread(callback_thread_id* thread)
    {
        return g_mock_sdk->create_callback_thread(thread);
    }

    static status_t assign_callback_thread(buffer_id buf, callback_thread_id thread)
    {
        return g_mock_sdk->assign_callback_thread(buf, thread);
    }

    static status_t configure_callback_tracing_service(
        context_id ctx, callback_tracing_kind kind, tracing_operation* ops,
        size_t ops_count, callback_tracing_cb_t cb, void* cb_data)
    {
        return g_mock_sdk->configure_callback_tracing_service(ctx, kind, ops, ops_count,
                                                              cb, cb_data);
    }

    static status_t configure_buffer_tracing_service(context_id          ctx,
                                                     buffer_tracing_kind kind,
                                                     tracing_operation*  ops,
                                                     size_t ops_count, buffer_id buf)
    {
        return g_mock_sdk->configure_buffer_tracing_service(ctx, kind, ops, ops_count,
                                                            buf);
    }

    static status_t configure_external_correlation_id_request_service(
        context_id ctx, const external_correlation_request_kind* kinds, size_t count,
        external_correlation_id_request_cb_t cb, void* cb_data)
    {
        return g_mock_sdk->configure_external_correlation_id_request_service(
            ctx, kinds, count, cb, cb_data);
    }

    static status_t configure_callback_dispatch_counting_service(
        context_id ctx, dispatch_counting_service_cb dispatch_cb, void* dispatch_data,
        dispatch_counting_record_cb record_cb, void* record_data)
    {
        return g_mock_sdk->configure_callback_dispatch_counting_service(
            ctx, dispatch_cb, dispatch_data, record_cb, record_data);
    }

    static status_t at_internal_thread_create(internal_thread_library_cb_t pre,
                                              internal_thread_library_cb_t post,
                                              runtime_library_t libs, void* user_data)
    {
        return g_mock_sdk->at_internal_thread_create(pre, post, libs, user_data);
    }

    static status_t context_is_active(context_id ctx, int* out)
    {
        return g_mock_sdk->context_is_active(ctx, out);
    }

    static status_t context_is_valid(context_id ctx, int* out)
    {
        return g_mock_sdk->context_is_valid(ctx, out);
    }

    static status_t query_callback_op_name(callback_tracing_kind kind,
                                           tracing_operation op, const char** name,
                                           std::uint64_t* name_len)
    {
        return g_mock_sdk->query_callback_op_name(kind, op, name, name_len);
    }

    static status_t query_buffer_op_name(buffer_tracing_kind kind, tracing_operation op,
                                         const char** name, std::uint64_t* name_len)
    {
        return g_mock_sdk->query_buffer_op_name(kind, op, name, name_len);
    }

    static status_t iterate_callback_tracing_kind_operation_args(
        callback_tracing_record rec, callback_tracing_operation_args_cb_t cb,
        std::int32_t max_deref, void* user_data)
    {
        return g_mock_sdk->iterate_callback_tracing_kind_operation_args(
            rec, cb, max_deref, user_data);
    }

    static status_t iterate_counter_dimensions(counter_id                id,
                                               available_dimensions_cb_t cb,
                                               void*                     user_data)
    {
        return g_mock_sdk->iterate_counter_dimensions(id, cb, user_data);
    }

    static status_t get_version(std::uint32_t* major, std::uint32_t* minor,
                                std::uint32_t* patch) noexcept
    {
        return g_mock_sdk->get_version(major, minor, patch);
    }

    static status_t get_timestamp(timestamp_t* ts) noexcept
    {
        return g_mock_sdk->get_timestamp(ts);
    }

    static const char* get_status_string(status_t s) noexcept
    {
        return g_mock_sdk->get_status_string(s);
    }
};

}  // namespace rocprofsys::backends::rocprofiler_sdk::testing
