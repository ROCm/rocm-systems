// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::backends::rocprofiler_sdk
{

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
    using status_t                     = Sdk::status_t;
    using context_id_t                 = Sdk::context_id;
    using agent_id_t                   = Sdk::agent_id;
    using buffer_id_t                  = Sdk::buffer_id;
    using counter_id_t                 = Sdk::counter_id;
    using counter_config_id_t          = Sdk::counter_config_id;
    using counter_record_t             = Sdk::counter_record;
    using counter_instance_id_t        = Sdk::counter_instance_id_t;
    using counter_flag_t               = Sdk::counter_flag_t;
    using user_data_t                  = Sdk::user_data_t;
    using available_counters_cb_t      = Sdk::available_counters_cb_t;
    using device_counting_agent_cb_t   = Sdk::device_counting_agent_cb_t;
    using device_counting_service_cb_t = Sdk::device_counting_service_cb_t;

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
