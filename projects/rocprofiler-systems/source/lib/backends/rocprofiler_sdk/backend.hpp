// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rocprofsys::backends::rocprofiler_sdk
{

template <typename Wrapper>
inline void
sdk_check(typename Wrapper::status_t status)
{
    if(status != Wrapper::STATUS_SUCCESS)
    {
        const char* msg = Wrapper::get_status_string(status);
        throw std::runtime_error{ std::string{ "rocprofiler-sdk error: " } +
                                  (msg != nullptr ? msg : "<unknown status>") };
    }
}

/// Device-counting service backend, parameterized over the rocprofiler-sdk
/// abstraction layer.
///
/// @tparam Wrapper  A type whose static interface matches rocprofiler_sdk::backend.
///                  All SDK function calls are routed through Wrapper, keeping this
///              struct free of direct SDK dependencies and fully mockable.
template <typename Wrapper>
struct backend
{
    using status_t                     = Wrapper::status_t;
    using context_id_t                 = Wrapper::context_id;
    using agent_id_t                   = Wrapper::agent_id;
    using buffer_id_t                  = Wrapper::buffer_id;
    using counter_id_t                 = Wrapper::counter_id;
    using counter_config_id_t          = Wrapper::counter_config_id;
    using counter_record_t             = Wrapper::counter_record;
    using counter_instance_id_t        = Wrapper::counter_instance_id_t;
    using counter_flag_t               = Wrapper::counter_flag_t;
    using user_data_t                  = Wrapper::user_data_t;
    using available_counters_cb_t      = Wrapper::available_counters_cb_t;
    using device_counting_agent_cb_t   = Wrapper::device_counting_agent_cb_t;
    using device_counting_service_cb_t = Wrapper::device_counting_service_cb_t;
    using buffer_policy_t              = Wrapper::buffer_policy_t;
    using buffer_tracing_cb_t          = Wrapper::buffer_tracing_cb_t;
    using callback_tracing_cb_t        = Wrapper::callback_tracing_cb_t;
    using callback_tracing_kind_t      = Wrapper::callback_tracing_kind;
    using buffer_tracing_kind_t        = Wrapper::buffer_tracing_kind;
    using tracing_operation_t          = Wrapper::tracing_operation;
    using callback_thread_id_t         = Wrapper::callback_thread_id;
    using runtime_library_t            = Wrapper::runtime_library_t;
    using external_correlation_request_kind_t =
        Wrapper::external_correlation_request_kind;
    using external_correlation_id_request_cb_t =
        Wrapper::external_correlation_id_request_cb_t;
    using internal_thread_library_cb_t = Wrapper::internal_thread_library_cb_t;
    using callback_tracing_record_t    = Wrapper::callback_tracing_record;
    using callback_tracing_operation_args_cb_t =
        Wrapper::callback_tracing_operation_args_cb_t;
    using available_dimensions_cb_t      = Wrapper::available_dimensions_cb_t;
    using counter_info_version_id_t      = Wrapper::counter_info_version_id_t;
    using timestamp_t                    = Wrapper::timestamp_t;
    using dispatch_counting_service_cb_t = Wrapper::dispatch_counting_service_cb;
    using dispatch_counting_record_cb_t  = Wrapper::dispatch_counting_record_cb;

    static constexpr counter_flag_t flag_none      = Wrapper::COUNTER_FLAG_NONE;
    static constexpr status_t       status_success = Wrapper::STATUS_SUCCESS;
    static constexpr status_t       status_error   = Wrapper::STATUS_ERROR;
    static constexpr status_t       status_hsa_not_loaded =
        Wrapper::STATUS_ERROR_HSA_NOT_LOADED;

    static agent_id_t make_agent_id(std::uint64_t handle) { return agent_id_t{ handle }; }

    static status_t create_context(context_id_t* ctx)
    {
        return Wrapper::create_context(ctx);
    }

    static status_t start_context(context_id_t ctx)
    {
        return Wrapper::start_context(ctx);
    }

    static status_t stop_context(context_id_t ctx) { return Wrapper::stop_context(ctx); }

    static status_t sample_device_counting_service(context_id_t      ctx,
                                                   user_data_t       user_data,
                                                   counter_flag_t    flags,
                                                   counter_record_t* output_records,
                                                   size_t*           record_count)
    {
        return Wrapper::sample_device_counting_service(ctx, user_data, flags,
                                                       output_records, record_count);
    }

    static status_t iterate_agent_supported_counters(agent_id_t              agent_id,
                                                     available_counters_cb_t callback,
                                                     void*                   user_data)
    {
        return Wrapper::iterate_agent_supported_counters(agent_id, callback, user_data);
    }

    static status_t create_counter_config(agent_id_t           agent_id,
                                          counter_id_t*        counters_list,
                                          size_t               counters_count,
                                          counter_config_id_t* config_id)
    {
        return Wrapper::create_counter_config(agent_id, counters_list, counters_count,
                                              config_id);
    }

    static status_t configure_device_counting_service(context_id_t ctx, buffer_id_t buf,
                                                      agent_id_t                   agent,
                                                      device_counting_service_cb_t cb,
                                                      void* user_data)
    {
        return Wrapper::configure_device_counting_service(ctx, buf, agent, cb, user_data);
    }

    static status_t query_record_counter_id(counter_record_t record,
                                            counter_id_t*    counter_id)
    {
        if constexpr(Wrapper::compile_time_version >= 10000)
        {
            if(counter_id == nullptr) return Wrapper::STATUS_ERROR_INVALID_ARGUMENT;
            counter_id->handle = record.id;
            return status_success;
        }
        else
        {
            return Wrapper::query_record_counter_id(record.id, counter_id);
        }
    }

    /// Queries the SDK for counter info and builds the SDK-agnostic counter_metadata
    /// representation. Uses SDK version v1 info (with per-instance dimensions) when
    /// available, falling back to v0 otherwise.
    static std::vector<counter_metadata> query_counter_details(counter_id_t counter_id)
    {
        auto safe_str = [](const char* s) {
            return s ? std::string{ s } : std::string{};
        };

        if constexpr(Wrapper::compile_time_version >= 10000)
        {
            typename Wrapper::counter_info_v1_t info{};
            if(Wrapper::query_counter_info(counter_id, Wrapper::COUNTER_INFO_VERSION_1,
                                           &info) != Wrapper::STATUS_SUCCESS ||
               info.name == nullptr)
                return {};

            auto result   = std::vector<counter_metadata>{};
            auto name_str = std::string{ info.name };
            auto desc_str = safe_str(info.description);
            auto blk_str  = safe_str(info.block);
            auto expr_str = safe_str(info.expression);
            result.reserve(info.dimensions_instances_count);

            for(std::uint64_t i = 0; i < info.dimensions_instances_count; ++i)
            {
                const auto* dim_inst = info.dimensions_instances[i];
                auto        dims     = std::vector<dimension_position>{};
                dims.reserve(dim_inst->dimensions_count);
                for(std::uint64_t d = 0; d < dim_inst->dimensions_count; ++d)
                {
                    const auto* dim = dim_inst->dimensions[d];
                    dims.push_back({ safe_str(dim->dimension_name), dim->index });
                }
                result.push_back(counter_metadata{
                    dim_inst->instance_id, name_str, desc_str, blk_str, expr_str,
                    static_cast<bool>(info.is_constant),
                    static_cast<bool>(info.is_derived), std::move(dims) });
            }
            return result;
        }
        else
        {
            typename Wrapper::counter_info_v0_t info{};
            if(Wrapper::query_counter_info(counter_id, Wrapper::COUNTER_INFO_VERSION_0,
                                           &info) != Wrapper::STATUS_SUCCESS ||
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

    static void create_buffer(context_id_t ctx, size_t size, size_t watermark,
                              buffer_policy_t policy, buffer_tracing_cb_t cb, void* data,
                              buffer_id_t* buf)
    {
        sdk_check<Wrapper>(
            Wrapper::create_buffer(ctx, size, watermark, policy, cb, data, buf));
    }

    static void flush_buffer(buffer_id_t buf)
    {
        auto status = Wrapper::flush_buffer(buf);
        if(status != Wrapper::STATUS_ERROR_BUFFER_BUSY)
        {
            sdk_check<Wrapper>(status);
        }
    }

    static void destroy_buffer(buffer_id_t buf)
    {
        while(Wrapper::destroy_buffer(buf) == Wrapper::STATUS_ERROR_BUFFER_BUSY)
        {
            std::this_thread::yield();
        }
    }

    static void create_callback_thread(callback_thread_id_t* thread)
    {
        sdk_check<Wrapper>(Wrapper::create_callback_thread(thread));
    }

    static void assign_callback_thread(buffer_id_t buf, callback_thread_id_t thread)
    {
        sdk_check<Wrapper>(Wrapper::assign_callback_thread(buf, thread));
    }

    static void configure_callback_tracing_service(
        context_id_t ctx, callback_tracing_kind_t kind, tracing_operation_t* ops,
        size_t ops_count, callback_tracing_cb_t cb, void* cb_data)
    {
        sdk_check<Wrapper>(Wrapper::configure_callback_tracing_service(
            ctx, kind, ops, ops_count, cb, cb_data));
    }

    static void configure_buffer_tracing_service(context_id_t          ctx,
                                                 buffer_tracing_kind_t kind,
                                                 tracing_operation_t*  ops,
                                                 size_t ops_count, buffer_id_t buf)
    {
        sdk_check<Wrapper>(
            Wrapper::configure_buffer_tracing_service(ctx, kind, ops, ops_count, buf));
    }

    static void configure_external_correlation_id_request_service(
        context_id_t ctx, const external_correlation_request_kind_t* kinds, size_t count,
        external_correlation_id_request_cb_t cb, void* cb_data)
    {
        sdk_check<Wrapper>(Wrapper::configure_external_correlation_id_request_service(
            ctx, kinds, count, cb, cb_data));
    }

    static void configure_callback_dispatch_counting_service(
        context_id_t ctx, dispatch_counting_service_cb_t dispatch_cb, void* dispatch_data,
        dispatch_counting_record_cb_t record_cb, void* record_data)
    {
        sdk_check<Wrapper>(Wrapper::configure_callback_dispatch_counting_service(
            ctx, dispatch_cb, dispatch_data, record_cb, record_data));
    }

    static void at_internal_thread_create(internal_thread_library_cb_t pre,
                                          internal_thread_library_cb_t post,
                                          runtime_library_t libs, void* user_data)
    {
        sdk_check<Wrapper>(
            Wrapper::at_internal_thread_create(pre, post, libs, user_data));
    }

    static bool context_is_active(context_id_t ctx) noexcept
    {
        return query_context_flag<Wrapper::context_is_active>(ctx);
    }

    static bool context_is_valid(context_id_t ctx) noexcept
    {
        return query_context_flag<Wrapper::context_is_valid>(ctx);
    }

private:
    template <auto WrapperFn>
    static bool query_context_flag(context_id_t ctx) noexcept
    {
        int out = 0;
        return (WrapperFn(ctx, &out) == Wrapper::STATUS_SUCCESS && out > 0);
    }

public:
    static void query_callback_op_name(callback_tracing_kind_t kind,
                                       tracing_operation_t op, const char** name,
                                       std::uint64_t* name_len)
    {
        sdk_check<Wrapper>(Wrapper::query_callback_op_name(kind, op, name, name_len));
    }

    static void query_buffer_op_name(buffer_tracing_kind_t kind, tracing_operation_t op,
                                     const char** name, std::uint64_t* name_len)
    {
        sdk_check<Wrapper>(Wrapper::query_buffer_op_name(kind, op, name, name_len));
    }

    static void iterate_callback_tracing_kind_operation_args(
        callback_tracing_record_t rec, callback_tracing_operation_args_cb_t cb,
        std::int32_t max_deref, void* user_data)
    {
        sdk_check<Wrapper>(Wrapper::iterate_callback_tracing_kind_operation_args(
            rec, cb, max_deref, user_data));
    }

    static void iterate_counter_dimensions(counter_id_t id, available_dimensions_cb_t cb,
                                           void* user_data)
    {
        sdk_check<Wrapper>(Wrapper::iterate_counter_dimensions(id, cb, user_data));
    }

    static void query_counter_info(counter_id_t id, counter_info_version_id_t version,
                                   void* info)
    {
        sdk_check<Wrapper>(Wrapper::query_counter_info(id, version, info));
    }

    static status_t get_version(std::uint32_t* major, std::uint32_t* minor,
                                std::uint32_t* patch)
    {
        return Wrapper::get_version(major, minor, patch);
    }

    static timestamp_t get_timestamp() noexcept
    {
        timestamp_t ts{};
        // Return code is always Wrapper::STATUS_SUCCESS.
        (void) Wrapper::get_timestamp(&ts);
        return ts;
    }

    static const char* get_status_string(status_t status) noexcept
    {
        return Wrapper::get_status_string(status);
    }
};

template <typename Wrapper>
struct backend_factory
{
    using backend_t = backend<Wrapper>;

    static std::shared_ptr<backend_t> create_backend()
    {
        return std::make_shared<backend_t>();
    }
};

}  // namespace rocprofsys::backends::rocprofiler_sdk
