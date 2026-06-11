// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "library/pmc/collectors/gpu_perf_counter/types.hpp"
#include <cstdint>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/version.h>
#if ROCPROFILER_VERSION >= 10000
#    include <rocprofiler-sdk/counter_config.h>
#else
#    include <rocprofiler-sdk/profile_config.h>
using rocprofiler_counter_config_id_t        = rocprofiler_profile_config_id_t;
using rocprofiler_counter_record_t           = rocprofiler_record_counter_t;
using rocprofiler_device_counting_agent_cb_t = rocprofiler_agent_set_profile_callback_t;
using rocprofiler_device_counting_service_cb_t =
    rocprofiler_device_counting_service_callback_t;
#endif

namespace rocprofsys::pmc::drivers::rocprofiler_sdk
{

using collectors::gpu_perf_counter::counter_metadata;
using collectors::gpu_perf_counter::dimension_position;

struct driver
{
    using counter_config_id_t          = rocprofiler_counter_config_id_t;
    using counter_id_t                 = rocprofiler_counter_id_t;
    using counter_record_t             = rocprofiler_counter_record_t;
    using device_counting_agent_cb_t   = rocprofiler_device_counting_agent_cb_t;
    using device_counting_service_cb_t = rocprofiler_device_counting_service_cb_t;
    using available_counters_cb_t      = rocprofiler_available_counters_cb_t;
    using context_id_t                 = rocprofiler_context_id_t;
    using agent_id_t                   = rocprofiler_agent_id_t;
    using buffer_id_t                  = rocprofiler_buffer_id_t;
    using status_t                     = rocprofiler_status_t;
    using counter_instance_id_t        = rocprofiler_counter_instance_id_t;
    using user_data_t                  = rocprofiler_user_data_t;
    using counter_flag_t               = rocprofiler_counter_flag_t;

    static constexpr counter_flag_t flag_none      = ROCPROFILER_COUNTER_FLAG_NONE;
    static constexpr status_t       status_success = ROCPROFILER_STATUS_SUCCESS;
    static constexpr status_t       status_error   = ROCPROFILER_STATUS_ERROR;
    static constexpr status_t       status_hsa_not_loaded =
        ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;

    static agent_id_t make_agent_id(std::uint64_t handle) { return agent_id_t{ handle }; }

    static status_t create_context(context_id_t* context)
    {
        return rocprofiler_create_context(context);
    }

    static status_t start_context(context_id_t context)
    {
        return rocprofiler_start_context(context);
    }

    static status_t stop_context(context_id_t context)
    {
        return rocprofiler_stop_context(context);
    }

    static status_t sample_device_counting_service(context_id_t      context,
                                                   user_data_t       user_data,
                                                   counter_flag_t    flags,
                                                   counter_record_t* output_records,
                                                   size_t*           record_count)
    {
        return rocprofiler_sample_device_counting_service(context, user_data, flags,
                                                          output_records, record_count);
    }

    static status_t query_record_counter_id(counter_record_t record,
                                            counter_id_t*    counter_id)
    {
#if ROCPROFILER_VERSION < 10000
        return rocprofiler_query_record_counter_id(record.id, counter_id);
#else
        if(counter_id == nullptr)
        {
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        }

        counter_id->handle = record.id;
        return ROCPROFILER_STATUS_SUCCESS;
#endif
    }

    static std::vector<counter_metadata> query_counter_details(counter_id_t counter_id)
    {
        auto safe_str = [](const char* str) {
            return str != nullptr ? std::string{ str } : std::string{};
        };

#if ROCPROFILER_VERSION >= 10000
        rocprofiler_counter_info_v1_t info{};
        auto                          status = rocprofiler_query_counter_info(
            counter_id, ROCPROFILER_COUNTER_INFO_VERSION_1, &info);
        if(status != ROCPROFILER_STATUS_SUCCESS || info.name == nullptr) return {};

        auto result = std::vector<counter_metadata>{};
        result.reserve(info.dimensions_instances_count);

        for(std::uint64_t i = 0; i < info.dimensions_instances_count; ++i)
        {
            const auto* dim_inst = info.dimensions_instances[i];
            auto        dims     = std::vector<dimension_position>{};
            dims.reserve(dim_inst->dimensions_count);
            for(std::uint64_t d = 0; d < dim_inst->dimensions_count; ++d)
            {
                dims.push_back({ std::string{ dim_inst->dimensions[d]->dimension_name },
                                 dim_inst->dimensions[d]->index });
            }
            result.push_back(counter_metadata{
                dim_inst->instance_id, std::string{ info.name },
                safe_str(info.description), safe_str(info.block),
                safe_str(info.expression), static_cast<bool>(info.is_constant),
                static_cast<bool>(info.is_derived), std::move(dims) });
        }
        return result;
#else
        rocprofiler_counter_info_v0_t info{};
        auto                          status = rocprofiler_query_counter_info(
            counter_id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info);
        if(status != ROCPROFILER_STATUS_SUCCESS || info.name == nullptr) return {};

        return { counter_metadata{ counter_id.handle,
                                   std::string{ info.name },
                                   safe_str(info.description),
                                   safe_str(info.block),
                                   safe_str(info.expression),
                                   static_cast<bool>(info.is_constant),
                                   static_cast<bool>(info.is_derived),
                                   {} } };
#endif
    }

    static status_t iterate_agent_supported_counters(agent_id_t              agent_id,
                                                     available_counters_cb_t callback,
                                                     void*                   user_data)
    {
        return rocprofiler_iterate_agent_supported_counters(agent_id, callback,
                                                            user_data);
    }

    static status_t create_counter_config(agent_id_t           agent_id,
                                          counter_id_t*        counters_list,
                                          size_t               counters_count,
                                          counter_config_id_t* config_id)
    {
#if ROCPROFILER_VERSION >= 10000
        return rocprofiler_create_counter_config(agent_id, counters_list, counters_count,
                                                 config_id);
#else
        return rocprofiler_create_profile_config(agent_id, counters_list, counters_count,
                                                 config_id);
#endif
    }

    static status_t configure_device_counting_service(
        context_id_t context_id, buffer_id_t buffer_id, agent_id_t agent_id,
        device_counting_service_cb_t callback, void* user_data)
    {
        return rocprofiler_configure_device_counting_service(
            context_id, buffer_id, agent_id, callback, user_data);
    }
};

struct driver_factory
{
    using driver_t = driver;

    static std::shared_ptr<driver_t> create_driver()
    {
        return std::make_shared<driver_t>();
    }
};

}  // namespace rocprofsys::pmc::drivers::rocprofiler_sdk
