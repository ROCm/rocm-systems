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
#include <rocprofiler-sdk/rocprofiler.h>
#include <cstdint>

#include "lib/rocprofiler-sdk/aql/helpers.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "rocprofiler-sdk/amd_detail/thread_trace.h"

using DispatchThreadTracer = rocprofiler::thread_trace::DispatchThreadTracer;
using AgentThreadTracer    = rocprofiler::thread_trace::AgentThreadTracer;

extern "C" {
rocprofiler_status_t
rocprofiler_configure_dispatch_thread_trace_service(
    rocprofiler_context_id_t               context_id,
    rocprofiler_att_parameter_t*           parameters,
    size_t                                 num_parameters,
    rocprofiler_att_dispatch_callback_t    dispatch_callback,
    rocprofiler_att_shader_data_callback_t shader_callback,
    void*                                  callback_userdata)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;
    if(ctx->dispatch_thread_trace) return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
    if(ctx->agent_thread_trace) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    auto pack = rocprofiler::thread_trace::thread_trace_parameter_pack{};

    pack.context_id        = context_id;
    pack.dispatch_cb_fn    = dispatch_callback;
    pack.shader_cb_fn      = shader_callback;
    pack.callback_userdata = callback_userdata;

    if(pack.dispatch_cb_fn == nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto id_map = rocprofiler::counters::getPerfCountersIdMap();
    for(size_t p = 0; p < num_parameters; p++)
    {
        const rocprofiler_att_parameter_t& param = parameters[p];
        if(param.type > ROCPROFILER_ATT_PARAMETER_LAST)
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

        switch(param.type)
        {
            case ROCPROFILER_ATT_PARAMETER_TARGET_CU: pack.target_cu = param.value; break;
            case ROCPROFILER_ATT_PARAMETER_SHADER_ENGINE_MASK:
                pack.shader_engine_mask = param.value;
                break;
            case ROCPROFILER_ATT_PARAMETER_BUFFER_SIZE: pack.buffer_size = param.value; break;
            case ROCPROFILER_ATT_PARAMETER_SIMD_SELECT: pack.simd_select = param.value; break;
            case ROCPROFILER_ATT_PARAMETER_PERFCOUNTER:
            {
                auto event_it = id_map.find(param.counter_id.handle);
                if(event_it != id_map.end())
                    pack.perfcounters.push_back({event_it->second, param.simd_mask});
            }
            break;
            case ROCPROFILER_ATT_PARAMETER_PERFCOUNTERS_CTRL:
                pack.perfcounter_ctrl = param.value;
                break;
            case ROCPROFILER_ATT_PARAMETER_LAST: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        }
    }

    if(!pack.are_params_valid()) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    ctx->dispatch_thread_trace = std::make_unique<DispatchThreadTracer>(pack);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_configure_agent_thread_trace_service(
    rocprofiler_context_id_t               context_id,
    rocprofiler_att_parameter_t*           parameters,
    size_t                                 num_parameters,
    rocprofiler_agent_id_t                 agent,
    rocprofiler_att_shader_data_callback_t shader_callback,
    void*                                  callback_userdata)
{
    if(rocprofiler::registration::get_init_status() > -1)
        return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;

    auto* ctx = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_STARTED;
    if(ctx->dispatch_thread_trace) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    if(!ctx->agent_thread_trace) ctx->agent_thread_trace = std::make_unique<AgentThreadTracer>();

    auto pack = rocprofiler::thread_trace::thread_trace_parameter_pack{};

    pack.context_id        = context_id;
    pack.shader_cb_fn      = shader_callback;
    pack.callback_userdata = callback_userdata;

    auto id_map = rocprofiler::counters::getPerfCountersIdMap();
    for(size_t p = 0; p < num_parameters; p++)
    {
        const rocprofiler_att_parameter_t& param = parameters[p];
        if(param.type > ROCPROFILER_ATT_PARAMETER_LAST)
            return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

        switch(param.type)
        {
            case ROCPROFILER_ATT_PARAMETER_TARGET_CU: pack.target_cu = param.value; break;
            case ROCPROFILER_ATT_PARAMETER_SHADER_ENGINE_MASK:
                pack.shader_engine_mask = param.value;
                break;
            case ROCPROFILER_ATT_PARAMETER_BUFFER_SIZE: pack.buffer_size = param.value; break;
            case ROCPROFILER_ATT_PARAMETER_SIMD_SELECT: pack.simd_select = param.value; break;
            case ROCPROFILER_ATT_PARAMETER_PERFCOUNTER:
            {
                auto event_it = id_map.find(param.counter_id.handle);
                if(event_it != id_map.end())
                    pack.perfcounters.push_back({event_it->second, param.simd_mask});
            }
            break;
            case ROCPROFILER_ATT_PARAMETER_PERFCOUNTERS_CTRL:
                pack.perfcounter_ctrl = param.value;
                break;
            case ROCPROFILER_ATT_PARAMETER_LAST: return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
        }
    }

    if(!pack.are_params_valid()) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    assert(ctx->agent_thread_trace);
    ctx->agent_thread_trace->add_agent(agent, pack);
    return ROCPROFILER_STATUS_SUCCESS;
}
}
