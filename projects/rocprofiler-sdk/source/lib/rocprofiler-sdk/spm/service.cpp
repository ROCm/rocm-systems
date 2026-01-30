// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/spm/asynchandler.hpp"
#include "lib/rocprofiler-sdk/spm/dlsym.hpp"

#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <glog/logging.h>
#include <hsa/hsa_api_trace.h>
#include <cstdint>

namespace rocprofiler
{
namespace spm
{
bool
is_dlsym_valid()
{
    static bool valid = Dlsym().valid();
    return valid;
}
}  // namespace spm
}  // namespace rocprofiler
extern "C" {

/**
 * @brief Create Profile Configuration.
 *
 * @param [in] agent Agent identifier
 * @param [in] counters_list List of GPU counters
 * @param [in] counters_count Size of counters list
 * @param [in/out] config_id Identifier for GPU counters group. If an existing
                   profile is supplied, that profiles counters will be copied
                   over to a new profile (returned via this id).
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_spm_create_counter_config(rocprofiler_agent_id_t               agent_id,
                                      rocprofiler_counter_id_t*            counters_list,
                                      size_t                               counters_count,
                                      rocprofiler_spm_configuration_t*     parameters,
                                      rocprofiler_spm_counter_config_id_t* config_id)
{
    if(!rocprofiler::spm::is_dlsym_valid()) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    if(!rocprofiler::spm::is_spm_explicitly_enabled())
        return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    std::unordered_set<uint64_t> already_added;
    const auto*                  agent = ::rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    std::shared_ptr<rocprofiler::spm::spm_counter_config> config =
        std::make_shared<rocprofiler::spm::spm_counter_config>();

    auto        metrics_map = rocprofiler::counters::loadMetrics();
    const auto& id_map      = metrics_map->id_to_metric;

    for(size_t i = 0; i < counters_count; i++)
    {
        auto& counter_id       = counters_list[i];
        auto  base_metric_id   = rocprofiler::counters::get_base_metric_from_counter_id(counter_id);
        const auto* metric_ptr = rocprofiler::common::get_val(id_map, base_metric_id);

        if(!metric_ptr) return ROCPROFILER_STATUS_ERROR_COUNTER_NOT_FOUND;
        // Don't add duplicates
        if(!already_added.emplace(metric_ptr->id()).second) continue;

        if(!rocprofiler::counters::checkValidMetric(std::string(agent->name), *metric_ptr) ||
           !metric_ptr->spm_support())
        {
            return ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT;
        }
        config->metrics.push_back(*metric_ptr);
    }
    
    if(parameters)
    {
       config->timeout     = parameters->timeout;
       config->buffer_size = parameters->buffer_size;
       config->sample_freq = parameters->frequency;
    }
    
    if(config_id->handle != 0)
    {
        // Copy existing counters from previous config
        if(auto existing = rocprofiler::spm::get_spm_counter_config(*config_id))
        {
            for(const auto& metric : existing->metrics)
            {
                if(!already_added.emplace(metric.id()).second) continue;
                config->metrics.push_back(metric);
            }
            if(existing->sample_freq != config->sample_freq)
                config->sample_freq = existing->sample_freq;
            if(existing->buffer_size != config->buffer_size)
                config->buffer_size = existing->buffer_size;
            if(existing->timeout != config->timeout) config->timeout = existing->timeout;
        }
    }

    config->agent = agent;
    if(auto status = rocprofiler::spm::create_spm_counter_profile(config);
       status != ROCPROFILER_STATUS_SUCCESS)
    {
        return ROCPROFILER_STATUS_ERROR_EXCEEDS_HW_LIMIT;
    }
    *config_id = config->id;

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_spm_destroy_counter_config(rocprofiler_spm_counter_config_id_t config_id)
{
    rocprofiler::spm::destroy_spm_counter_profile(config_id.handle);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_configure_callback_spm_dispatch_service(
    rocprofiler_context_id_t                       context_id,
    rocprofiler_spm_dispatch_counting_service_cb_t dispatch_callback,
    void*                                          dispatch_callback_args,
    rocprofiler_spm_dispatch_counting_record_cb_t  record_callback,
    void*                                          record_callback_args)
{
    if(!rocprofiler::spm::is_dlsym_valid()) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    if(!rocprofiler::spm::is_spm_explicitly_enabled())
        return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    return rocprofiler::spm::configure_callback_spm_dispatch(context_id,
                                                             dispatch_callback,
                                                             dispatch_callback_args,
                                                             record_callback,
                                                             record_callback_args);
}

/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in] agent
 * @param [out] counters_list
 * @param [out] counters_count
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_iterate_spm_supported_counters(rocprofiler_agent_id_t              agent_id,
                                           rocprofiler_available_counters_cb_t cb,
                                           void*                               user_data)
{
    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    auto metrics = rocprofiler::counters::getMetricsForAgent(agent);

    auto ids = std::vector<rocprofiler_counter_id_t>{};

    for(const auto& m : metrics)
    {
        if(m.spm_support())
        {
            // Create agent-encoded counter ID using the agent's logical_node_id
            rocprofiler_counter_id_t counter_id{.handle = 0};
            rocprofiler::counters::set_base_metric_in_counter_id(counter_id, m.id());
            rocprofiler::counters::set_agent_in_counter_id(counter_id, agent->logical_node_id);
            ids.push_back(counter_id);
        }
    }
    if(ids.empty()) return ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED;

    return cb(agent_id, ids.data(), ids.size(), user_data);
}
}
