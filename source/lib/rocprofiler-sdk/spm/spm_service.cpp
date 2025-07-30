// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <glog/logging.h>
#include <hsa/hsa_api_trace.h>
#include <rocprofiler-sdk/experimental/spm/capture.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <cstdint>

#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/spm/spm_dlsym.hpp"

namespace rocprofiler
{
namespace SPM
{
bool
is_dlsym_valid()
{
    static bool valid = Dlsym().valid();
    return valid;
}

bool
build_pack(spm_parameter_pack&          pack,
           rocprofiler_agent_id_t       agent_id,
           rocprofiler_spm_parameter_t* parameters,
           size_t                       parameter_count,
           rocprofiler_counter_id_t*    counters_list,
           size_t                       counters_count)
{
    for(size_t p = 0; p < parameter_count; p++)
    {
        const rocprofiler_spm_parameter_t& param = parameters[p];

        switch(param.type)
        {
            case ROCPROFILER_SPM_PARAMETER_TIMEOUT_MS: pack.timeout = param.value; break;
            case ROCPROFILER_SPM_PARAMETER_BUFFER_SIZE: pack.buffer_size = param.value; break;
            case ROCPROFILER_SPM_PARAMETER_SAMPLE_FREQUENCY: pack.sample_freq = param.value; break;
            case ROCPROFILER_SPM_PARAMETER_LAST: return false;
            default: return false;
        }
    }

    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return false;

    const auto* metricset = rocprofiler::counters::getSupportedSPMCounters(agent->name);
    const auto& id_map    = rocprofiler::counters::loadMetrics()->id_to_metric;

    for(size_t i = 0; i < counters_count; i++)
    {
        // Check for unsupported metrics
        if(metricset->find(counters_list[i].handle) == metricset->end()) return false;

        auto it = id_map.find(counters_list[i].handle);
        if(it == id_map.end()) return false;
        pack.metrics.push_back(it->second);
    }

    return pack.valid();
}
};  // namespace SPM
};  // namespace rocprofiler

extern "C" {
rocprofiler_status_t
rocprofiler_configure_spm_agent_service(rocprofiler_context_id_t        context_id,
                                        rocprofiler_agent_id_t          agent_id,
                                        rocprofiler_counter_id_t*       counters_list,
                                        size_t                          counters_count,
                                        rocprofiler_spm_parameter_t*    parameters,
                                        size_t                          parameter_count,
                                        rocprofiler_spm_data_callback_t data_fn,
                                        rocprofiler_user_data_t         user_data)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if(!data_fn) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    if(!rocprofiler::SPM::is_dlsym_valid()) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    if(!ctx->agent_spm) ctx->agent_spm = std::make_unique<rocprofiler::SPM::SPMAgentManager>();

    auto pack      = rocprofiler::SPM::spm_parameter_pack{};
    pack.data_fn   = data_fn;
    pack.user_data = user_data;

    if(!rocprofiler::SPM::build_pack(
           pack, agent_id, parameters, parameter_count, counters_list, counters_count))
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    if(!ctx->agent_spm->add_agent(agent_id, pack))
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_configure_spm_dispatch_service(rocprofiler_context_id_t            context_id,
                                           rocprofiler_agent_id_t              agent_id,
                                           rocprofiler_counter_id_t*           counters_list,
                                           size_t                              counters_count,
                                           rocprofiler_spm_parameter_t*        parameters,
                                           size_t                              parameter_count,
                                           rocprofiler_spm_dispatch_callback_t dispatch_fn,
                                           rocprofiler_spm_data_callback_t     data_fn,
                                           void*                               config_userdata)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    if(!data_fn || !dispatch_fn) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    if(!rocprofiler::SPM::is_dlsym_valid()) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    if(!ctx->dispatch_spm)
        ctx->dispatch_spm = std::make_unique<rocprofiler::SPM::SPMDispatchManager>();

    auto pack = rocprofiler::SPM::spm_parameter_pack{};

    pack.data_fn         = data_fn;
    pack.dispatch_fn     = dispatch_fn;
    pack.config_userdata = config_userdata;

    if(!rocprofiler::SPM::build_pack(
           pack, agent_id, parameters, parameter_count, counters_list, counters_count))
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    if(!ctx->dispatch_spm->add_agent(agent_id, pack))
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;

    return ROCPROFILER_STATUS_SUCCESS;
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
    if(!rocprofiler::SPM::is_dlsym_valid()) return ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI;

    const auto* agent = rocprofiler::agent::get_agent(agent_id);
    if(!agent) return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;

    const auto* id_set = rocprofiler::counters::getSupportedSPMCounters(agent->name);
    if(id_set->empty()) return ROCPROFILER_STATUS_ERROR_AGENT_ARCH_NOT_SUPPORTED;

    std::vector<rocprofiler_counter_id_t> ids{};
    ids.reserve(id_set->size());

    for(auto id : *id_set)
        ids.push_back(rocprofiler_counter_id_t{.handle = id});

    return cb(agent_id, ids.data(), ids.size(), user_data);
}
}
