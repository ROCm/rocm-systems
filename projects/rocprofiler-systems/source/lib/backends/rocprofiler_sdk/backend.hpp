// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rocprofsys::backends::rocprofiler_sdk
{

template <typename Sdk>
inline void
sdk_check(typename Sdk::status_t status)
{
    if(status != Sdk::STATUS_SUCCESS)
    {
        throw std::runtime_error{ std::string{ Sdk::get_status_string(status) } };
    }
}

/// Device-counting service backend, parameterized over the rocprofiler-sdk
/// abstraction layer.
///
/// @tparam Sdk  A type whose static interface matches rocprofiler_sdk::backend.
///              All SDK function calls are routed through Sdk, keeping this
///              struct free of direct SDK dependencies and fully mockable.
template <typename Sdk>
struct backend
{
    // ─── Type aliases from Sdk ────────────────────────────────────────────────────
    using status_t                            = Sdk::status_t;
    using context_id_t                        = Sdk::context_id;
    using agent_id_t                          = Sdk::agent_id;
    using buffer_id_t                         = Sdk::buffer_id;
    using counter_id_t                        = Sdk::counter_id;
    using counter_config_id_t                 = Sdk::counter_config_id;
    using counter_record_t                    = Sdk::counter_record;
    using counter_instance_id_t               = Sdk::counter_instance_id_t;
    using counter_flag_t                      = Sdk::counter_flag_t;
    using user_data_t                         = Sdk::user_data_t;
    using available_counters_cb_t             = Sdk::available_counters_cb_t;
    using device_counting_agent_cb_t          = Sdk::device_counting_agent_cb_t;
    using device_counting_service_cb_t        = Sdk::device_counting_service_cb_t;
    using buffer_policy_t                     = Sdk::buffer_policy_t;
    using buffer_tracing_cb_t                 = Sdk::buffer_tracing_cb_t;
    using callback_tracing_cb_t               = Sdk::callback_tracing_cb_t;
    using callback_tracing_kind_t             = Sdk::callback_tracing_kind;
    using buffer_tracing_kind_t               = Sdk::buffer_tracing_kind;
    using tracing_operation_t                 = Sdk::tracing_operation;
    using callback_thread_id_t                = Sdk::callback_thread_id;
    using runtime_library_t                   = Sdk::runtime_library_t;
    using external_correlation_request_kind_t = Sdk::external_correlation_request_kind;
    using external_correlation_id_request_cb_t =
        Sdk::external_correlation_id_request_cb_t;
    using internal_thread_library_cb_t = Sdk::internal_thread_library_cb_t;
    using callback_tracing_record_t    = Sdk::callback_tracing_record;
    using callback_tracing_operation_args_cb_t =
        Sdk::callback_tracing_operation_args_cb_t;
    using available_dimensions_cb_t      = Sdk::available_dimensions_cb_t;
    using counter_info_version_id_t      = Sdk::counter_info_version_id_t;
    using timestamp_t                    = Sdk::timestamp_t;
    using dispatch_counting_service_cb_t = Sdk::dispatch_counting_service_cb;
    using dispatch_counting_record_cb_t  = Sdk::dispatch_counting_record_cb;

    // ─── Constants from Sdk ───────────────────────────────────────────────────────
    static constexpr counter_flag_t flag_none       = Sdk::COUNTER_FLAG_NONE;
    static constexpr status_t       status_success  = Sdk::STATUS_SUCCESS;
    static constexpr status_t       status_error    = Sdk::STATUS_ERROR;
    static constexpr status_t status_hsa_not_loaded = Sdk::STATUS_ERROR_HSA_NOT_LOADED;

    // ─── Helper ───────────────────────────────────────────────────────────────────
    static agent_id_t make_agent_id(std::uint64_t handle) { return agent_id_t{ handle }; }

    // ─── Pure SDK delegations ─────────────────────────────────────────────────────

    static status_t create_context(context_id_t* ctx) { return Sdk::create_context(ctx); }

    static status_t start_context(context_id_t ctx) { return Sdk::start_context(ctx); }

    static status_t stop_context(context_id_t ctx) { return Sdk::stop_context(ctx); }

    static status_t sample_device_counting_service(context_id_t      ctx,
                                                   user_data_t       user_data,
                                                   counter_flag_t    flags,
                                                   counter_record_t* output_records,
                                                   size_t*           record_count)
    {
        return Sdk::sample_device_counting_service(ctx, user_data, flags, output_records,
                                                   record_count);
    }

    static status_t iterate_agent_supported_counters(agent_id_t              agent_id,
                                                     available_counters_cb_t callback,
                                                     void*                   user_data)
    {
        return Sdk::iterate_agent_supported_counters(agent_id, callback, user_data);
    }

    static status_t create_counter_config(agent_id_t           agent_id,
                                          counter_id_t*        counters_list,
                                          size_t               counters_count,
                                          counter_config_id_t* config_id)
    {
        return Sdk::create_counter_config(agent_id, counters_list, counters_count,
                                          config_id);
    }

    static status_t configure_device_counting_service(context_id_t ctx, buffer_id_t buf,
                                                      agent_id_t                   agent,
                                                      device_counting_service_cb_t cb,
                                                      void* user_data)
    {
        return Sdk::configure_device_counting_service(ctx, buf, agent, cb, user_data);
    }

    // ─── Business logic ───────────────────────────────────────────────────────────

    /// Adapts the record-level API: extracts the instance id from a full counter
    /// record and delegates to Sdk::query_record_counter_id.
    static status_t query_record_counter_id(counter_record_t record,
                                            counter_id_t*    counter_id)
    {
        return Sdk::query_record_counter_id(record.id, counter_id);
    }

    /// Queries the SDK for counter info and builds the SDK-agnostic counter_metadata
    /// representation. Uses SDK version v1 info (with per-instance dimensions) when
    /// available, falling back to v0 otherwise.
    static std::vector<counter_metadata> query_counter_details(counter_id_t counter_id)
    {
        auto safe_str = [](const char* s) {
            return s ? std::string{ s } : std::string{};
        };

        if constexpr(Sdk::compile_time_version >= 10000)
        {
            typename Sdk::counter_info_v1_t info{};
            if(Sdk::query_counter_info(counter_id, Sdk::COUNTER_INFO_VERSION_1, &info) !=
                   Sdk::STATUS_SUCCESS ||
               info.name == nullptr)
                return {};

            auto result = std::vector<counter_metadata>{};
            result.reserve(info.dimensions_instances_count);

            for(std::uint64_t i = 0; i < info.dimensions_instances_count; ++i)
            {
                const auto* dim_inst = info.dimensions_instances[i];
                auto        dims     = std::vector<dimension_position>{};
                dims.reserve(dim_inst->dimensions_count);
                for(std::uint64_t d = 0; d < dim_inst->dimensions_count; ++d)
                {
                    dims.push_back(
                        { std::string{ dim_inst->dimensions[d]->dimension_name },
                          dim_inst->dimensions[d]->index });
                }
                result.push_back(counter_metadata{
                    dim_inst->instance_id, std::string{ info.name },
                    safe_str(info.description), safe_str(info.block),
                    safe_str(info.expression), static_cast<bool>(info.is_constant),
                    static_cast<bool>(info.is_derived), std::move(dims) });
            }
            return result;
        }
        else
        {
            typename Sdk::counter_info_v0_t info{};
            if(Sdk::query_counter_info(counter_id, Sdk::COUNTER_INFO_VERSION_0, &info) !=
                   Sdk::STATUS_SUCCESS ||
               info.name == nullptr)
                return {};

            return { counter_metadata{ counter_id.handle,
                                       std::string{ info.name },
                                       safe_str(info.description),
                                       safe_str(info.block),
                                       safe_str(info.expression),
                                       static_cast<bool>(info.is_constant),
                                       static_cast<bool>(info.is_derived),
                                       {} } };
        }
    }

    // ─── Buffer and callback thread management ────────────────────────────────────

    static void create_buffer(context_id_t ctx, size_t size, size_t watermark,
                              buffer_policy_t policy, buffer_tracing_cb_t cb, void* data,
                              buffer_id_t* buf)
    {
        sdk_check<Sdk>(Sdk::create_buffer(ctx, size, watermark, policy, cb, data, buf));
    }

    static void flush_buffer(buffer_id_t buf)
    {
        auto status = Sdk::flush_buffer(buf);
        if(status != Sdk::STATUS_ERROR_BUFFER_BUSY)
        {
            sdk_check<Sdk>(status);
        }
    }

    static void destroy_buffer(buffer_id_t buf)
    {
        while(Sdk::destroy_buffer(buf) == Sdk::STATUS_ERROR_BUFFER_BUSY)
        {
            std::this_thread::yield();
        }
    }

    static void create_callback_thread(callback_thread_id_t* thread)
    {
        sdk_check<Sdk>(Sdk::create_callback_thread(thread));
    }

    static void assign_callback_thread(buffer_id_t buf, callback_thread_id_t thread)
    {
        sdk_check<Sdk>(Sdk::assign_callback_thread(buf, thread));
    }

    // ─── Tracing service configuration ───────────────────────────────────────────

    static void configure_callback_tracing_service(
        context_id_t ctx, callback_tracing_kind_t kind, tracing_operation_t* ops,
        size_t ops_count, callback_tracing_cb_t cb, void* cb_data)
    {
        sdk_check<Sdk>(Sdk::configure_callback_tracing_service(ctx, kind, ops, ops_count,
                                                               cb, cb_data));
    }

    static void configure_buffer_tracing_service(context_id_t          ctx,
                                                 buffer_tracing_kind_t kind,
                                                 tracing_operation_t*  ops,
                                                 size_t ops_count, buffer_id_t buf)
    {
        sdk_check<Sdk>(
            Sdk::configure_buffer_tracing_service(ctx, kind, ops, ops_count, buf));
    }

    static void configure_external_correlation_id_request_service(
        context_id_t ctx, const external_correlation_request_kind_t* kinds, size_t count,
        external_correlation_id_request_cb_t cb, void* cb_data)
    {
        sdk_check<Sdk>(Sdk::configure_external_correlation_id_request_service(
            ctx, kinds, count, cb, cb_data));
    }

    static void configure_callback_dispatch_counting_service(
        context_id_t ctx, dispatch_counting_service_cb_t dispatch_cb, void* dispatch_data,
        dispatch_counting_record_cb_t record_cb, void* record_data)
    {
        sdk_check<Sdk>(Sdk::configure_callback_dispatch_counting_service(
            ctx, dispatch_cb, dispatch_data, record_cb, record_data));
    }

    static void at_internal_thread_create(internal_thread_library_cb_t pre,
                                          internal_thread_library_cb_t post,
                                          runtime_library_t libs, void* user_data)
    {
        sdk_check<Sdk>(Sdk::at_internal_thread_create(pre, post, libs, user_data));
    }

    // ─── Context queries — return bool, never throw ───────────────────────────────

    static bool context_is_active(context_id_t ctx)
    {
        int  out  = 0;
        auto errc = Sdk::context_is_active(ctx, &out);
        return (errc == Sdk::STATUS_SUCCESS && out > 0);
    }

    static bool context_is_valid(context_id_t ctx)
    {
        int  out  = 0;
        auto errc = Sdk::context_is_valid(ctx, &out);
        return (errc == Sdk::STATUS_SUCCESS && out > 0);
    }

    // ─── Tracing queries ──────────────────────────────────────────────────────────

    static void query_callback_op_name(callback_tracing_kind_t kind,
                                       tracing_operation_t op, const char** name,
                                       std::uint64_t* name_len)
    {
        sdk_check<Sdk>(Sdk::query_callback_op_name(kind, op, name, name_len));
    }

    static void query_buffer_op_name(buffer_tracing_kind_t kind, tracing_operation_t op,
                                     const char** name, std::uint64_t* name_len)
    {
        sdk_check<Sdk>(Sdk::query_buffer_op_name(kind, op, name, name_len));
    }

    static void iterate_callback_tracing_kind_operation_args(
        callback_tracing_record_t rec, callback_tracing_operation_args_cb_t cb,
        std::int32_t max_deref, void* user_data)
    {
        sdk_check<Sdk>(Sdk::iterate_callback_tracing_kind_operation_args(
            rec, cb, max_deref, user_data));
    }

    static void iterate_counter_dimensions(counter_id_t id, available_dimensions_cb_t cb,
                                           void* user_data)
    {
        sdk_check<Sdk>(Sdk::iterate_counter_dimensions(id, cb, user_data));
    }

    static void query_counter_info(counter_id_t id, counter_info_version_id_t version,
                                   void* info)
    {
        sdk_check<Sdk>(Sdk::query_counter_info(id, version, info));
    }

    // ─── Timestamp and status — noexcept, never throw ────────────────────────────

    static timestamp_t get_timestamp() noexcept
    {
        timestamp_t ts{};
        Sdk::get_timestamp(&ts);
        return ts;
    }

    static const char* get_status_string(status_t status) noexcept
    {
        return Sdk::get_status_string(status);
    }
};

template <typename Sdk>
struct backend_factory
{
    using backend_t = backend<Sdk>;

    static std::shared_ptr<backend_t> create_backend()
    {
        return std::make_shared<backend_t>();
    }
};

}  // namespace rocprofsys::backends::rocprofiler_sdk
