// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/id_decode.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/spm/interface.hpp"

#include <rocprofiler-sdk/experimental/spm.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <memory>
#include <vector>

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
rocprofiler_spm_create_counter_config(rocprofiler_agent_id_t           agent_id,
                                      rocprofiler_counter_id_t*        counters_list,
                                      size_t                           counters_count,
                                      rocprofiler_spm_parameters_t**   parameters,
                                      size_t                           parameters_count,
                                      rocprofiler_counter_config_id_t* config_id)
{
    const auto* sym = rocprofiler::spm::construct_spm_interface();
    if(!sym) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

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
           !rocprofiler::counters::has_spm_support(*metric_ptr, agent->id))
        {
            return ROCPROFILER_STATUS_ERROR_METRIC_NOT_VALID_FOR_AGENT;
        }
        config->metrics.push_back(*metric_ptr);
    }

    for(size_t i = 0; i < parameters_count; i++)
    {
        config->spm_parameters.emplace_back(
            rocprofiler_spm_parameters_t{.size  = sizeof(rocprofiler_spm_parameters_t),
                                         .type  = CHECK_NOTNULL(parameters[i])->type,
                                         .value = CHECK_NOTNULL(parameters[i])->value});
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
        }
    }

    config->agent = agent;
    if(auto status = rocprofiler::spm::create_spm_counter_profile(config);
       status != ROCPROFILER_STATUS_SUCCESS)
    {
        return status;
    }
    *config_id = config->id;

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_spm_destroy_counter_config(rocprofiler_counter_config_id_t config_id)
{
    return rocprofiler::spm::destroy_spm_counter_profile(config_id);
}

rocprofiler_status_t
rocprofiler_spm_configure_callback_dispatch_service(
    rocprofiler_context_id_t                       context_id,
    rocprofiler_spm_dispatch_counting_service_cb_t dispatch_callback,
    void*                                          dispatch_callback_args,
    rocprofiler_spm_dispatch_counting_record_cb_t  record_callback,
    void*                                          record_callback_args)
{
    const auto* sym = rocprofiler::spm::construct_spm_interface();
    if(!sym) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    if(!rocprofiler::spm::is_spm_explicitly_enabled())
        return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    return rocprofiler::spm::configure_callback_spm_dispatch(context_id,
                                                             dispatch_callback,
                                                             dispatch_callback_args,
                                                             record_callback,
                                                             record_callback_args);
}

/**
 * @brief Query Agent Counters Availability.
 *
 * @param [in]  agent_id agent for which the supported counters are queried
 * @param [in]  callback to return available counters
 * @param [in]  user_data data to be passed to the callback
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_spm_iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                                 rocprofiler_available_counters_cb_t cb,
                                                 void*                               user_data)
{
    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    const auto* sym = rocprofiler::spm::construct_spm_interface();
    if(!sym) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    auto metrics = rocprofiler::counters::getMetricsForAgent(agent);

    auto ids = std::vector<rocprofiler_counter_id_t>{};

    for(const auto& m : metrics)
    {
        if(rocprofiler::counters::has_spm_support(m, agent->id))
        {
            rocprofiler_counter_id_t counter_id{.handle = 0};
            rocprofiler::counters::set_base_metric_in_counter_id(counter_id, m.id());
            ids.push_back(counter_id);
        }
    }
    if(ids.empty()) return ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED;

    return cb(agent_id, ids.data(), ids.size(), user_data);
}

/**
 * @brief Configure buffered dispatch profile Counting Service.
 *        Collects the counters in dispatch packets and stores them
 *        in buffer_id. The buffer may contain packets from more than
 *        one dispatch (denoted by dispatch id). Will trigger the
 *        callback based on the parameters setup in buffer_id_t.
 *
 * @param [in] context_id context to configure spm buffer service
 * @param [in] buffer_id buffer to use for the counting service
 * @param [in] callback  callback to be invoked when a kernel is dispatched
 * @param [in] callback_data_args  callback data passed to the callback
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t
rocprofiler_spm_configure_buffer_dispatch_service(
    rocprofiler_context_id_t                       context_id,
    rocprofiler_buffer_id_t                        buffer_id,
    rocprofiler_spm_dispatch_counting_service_cb_t callback,
    void*                                          callback_data_args)
{
    auto* ctx_p = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx_p) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    const auto* sym = rocprofiler::spm::construct_spm_interface();
    if(!sym) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    if(!rocprofiler::spm::is_spm_explicitly_enabled())
        return ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED;

    // checking if the buffer is registered
    auto const* buff = rocprofiler::buffer::get_buffer(buffer_id);
    if(!buff) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    auto& ctx = *ctx_p;

    if(ctx.pc_sampler) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
    if(ctx.dispatch_counter_collection) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
    if(ctx.device_counter_collection) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
    if(!ctx.dispatch_spm)
        ctx.dispatch_spm =
            std::make_unique<rocprofiler::context::spm_dispatch_counter_collection_service>();
    auto& cb = *ctx.dispatch_spm->callbacks.emplace_back(
        std::make_shared<rocprofiler::spm::spm_counter_callback_info>());

    cb.user_cb          = callback;
    cb.callback_args    = callback_data_args;
    cb.context          = context_id;
    cb.buffer           = buffer_id;
    cb.internal_context = ctx_p;

    return ROCPROFILER_STATUS_SUCCESS;
}

using spm_config_vec_t = std::vector<std::unique_ptr<rocprofiler_spm_available_configuration_t>>;

rocprofiler_spm_parameter_type_t
get_type(aqlprofile_spm_parameter_type_t src)
{
    switch(src)
    {
        case AQLPROFILE_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL:
            return ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES;
        default: break;
    }
    return ROCPROFILER_SPM_PARAMETER_TYPE_NONE;
}

hsa_status_t
query_cb(const aqlprofile_spm_available_configuration_t* config,
         size_t                                          configs_size,
         void*                                           userdata)
{
    auto& configs_supported = *reinterpret_cast<spm_config_vec_t*>(userdata);
    for(size_t itr = 0; itr < configs_size; itr++)
    {
        auto cfg = rocprofiler_spm_available_configuration_t{};
        cfg.size = sizeof(rocprofiler_spm_available_configuration_t);
        cfg.type = get_type(config[itr].type);
        switch(cfg.type)
        {
            case ROCPROFILER_SPM_PARAMETER_TYPE_SAMPLE_INTERVAL_SCLK_CYCLES:
                cfg.interval.min_interval = config[itr].interval.min_interval;
                cfg.interval.max_interval = config[itr].interval.max_interval;
                break;
            case ROCPROFILER_SPM_PARAMETER_TYPE_NONE:
            case ROCPROFILER_SPM_PARAMETER_TYPE_LAST: continue;
        }
        configs_supported.emplace_back(
            std::make_unique<rocprofiler_spm_available_configuration_t>(cfg));
    }
    return HSA_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_spm_query_agent_configurations(rocprofiler_agent_id_t                        agent_id,
                                           rocprofiler_spm_available_configurations_cb_t cb,
                                           void*                                         user_data)
{
    const auto* sym = rocprofiler::spm::construct_spm_interface();
    if(!sym) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    const auto* aql_agent = rocprofiler::agent::get_aql_agent(agent_id);
    if(!aql_agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    spm_config_vec_t configs_supported{};
    auto status = sym->spm_query_agent_configurations(*aql_agent, query_cb, &configs_supported);

    if(status == HSA_STATUS_SUCCESS)
    {
        std::vector<const rocprofiler_spm_available_configuration_t*> config_ptrs;
        config_ptrs.reserve(configs_supported.size());
        for(auto& cfg : configs_supported)
            config_ptrs.push_back(cfg.get());

        cb(config_ptrs.data(), config_ptrs.size(), user_data);
        return ROCPROFILER_STATUS_SUCCESS;
    }
    else
        return ROCPROFILER_STATUS_ERROR;
}
}
