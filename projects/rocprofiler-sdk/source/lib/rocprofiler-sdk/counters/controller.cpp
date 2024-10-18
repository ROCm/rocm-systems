// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/counters/controller.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"

namespace rocprofiler
{
namespace counters
{
CounterController::CounterController()
{
    // Pre-read metrics map file to catch faliures during initial setup.
    rocprofiler::counters::getMetricIdMap();
}

// Adds a counter collection profile to our global cache.
// Note: these profiles can be used across multiple contexts
//       and are independent of the context.
uint64_t
CounterController::add_profile(std::shared_ptr<profile_config>&& config)
{
    static std::atomic<uint64_t> profile_val = 1;
    uint64_t                     ret         = 0;
    _configs.wlock([&](auto& data) {
        config->id = rocprofiler_profile_config_id_t{.handle = profile_val};
        data.emplace(profile_val, std::move(config));
        ret = profile_val;
        profile_val++;
    });
    return ret;
}

void
CounterController::destroy_profile(uint64_t id)
{
    _configs.wlock([&](auto& data) { data.erase(id); });
}

rocprofiler_status_t
CounterController::configure_agent_collection(rocprofiler_context_id_t context_id,
                                              rocprofiler_buffer_id_t  buffer_id,
                                              rocprofiler_agent_id_t   agent_id,
                                              rocprofiler_device_counting_service_callback_t cb,
                                              void* user_data)
{
    auto* ctx_p = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx_p) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    auto& ctx = *ctx_p;

    if(ctx.counter_collection) return ROCPROFILER_STATUS_ERROR_AGENT_DISPATCH_CONFLICT;

    // FIXME: Due to the clock gating issue, counter collection and PC sampling service
    // cannot coexist in the same context for now.
    if(ctx.pc_sampler) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;

    if(!rocprofiler::buffer::get_buffer(buffer_id.handle))
    {
        return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;
    }

    if(!ctx.device_counter_collection)
    {
        ctx.device_counter_collection =
            std::make_unique<rocprofiler::context::device_counting_service>();
    }

    if(ctx.device_counter_collection->conf_agents.emplace(agent_id.handle).second == false)
    {
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    ctx.device_counter_collection->agent_data.emplace_back();
    ctx.device_counter_collection->agent_data.back().callback_data =
        rocprofiler_user_data_t{.ptr = user_data};
    ctx.device_counter_collection->agent_data.back().agent_id = agent_id;
    ctx.device_counter_collection->agent_data.back().cb       = cb;
    ctx.device_counter_collection->agent_data.back().buffer   = buffer_id;

    return ROCPROFILER_STATUS_SUCCESS;
}

// Setup the counter collection service. counter_callback_info is created here
// to contain the counters that need to be collected (specified in profile_id) and
// the AQL packet generator for injecting packets. Note: the service is created
// in the stop state.
rocprofiler_status_t
CounterController::configure_dispatch(
    rocprofiler_context_id_t                         context_id,
    rocprofiler_buffer_id_t                          buffer,
    rocprofiler_dispatch_counting_service_callback_t callback,
    void*                                            callback_args,
    rocprofiler_profile_counting_record_callback_t   record_callback,
    void*                                            record_callback_args)
{
    auto* ctx_p = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx_p) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    auto& ctx = *ctx_p;

    if(ctx.device_counter_collection) return ROCPROFILER_STATUS_ERROR_AGENT_DISPATCH_CONFLICT;

    // FIXME: Due to the clock gating issue, counter collection and PC sampling service
    // cannot coexist in the same context for now.
    if(ctx.pc_sampler) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;

    if(!ctx.counter_collection)
    {
        ctx.counter_collection =
            std::make_unique<rocprofiler::context::dispatch_counter_collection_service>();
    }

    auto& cb =
        *ctx.counter_collection->callbacks.emplace_back(std::make_shared<counter_callback_info>());

    cb.user_cb       = callback;
    cb.callback_args = callback_args;
    cb.context       = context_id;
    if(buffer.handle != 0)
    {
        cb.buffer = buffer;
    }
    cb.internal_context     = ctx_p;
    cb.record_callback      = record_callback;
    cb.record_callback_args = record_callback_args;

    return ROCPROFILER_STATUS_SUCCESS;
}

std::shared_ptr<profile_config>
CounterController::get_profile_cfg(rocprofiler_profile_config_id_t id)
{
    std::shared_ptr<profile_config> cfg;
    _configs.rlock([&](const auto& map) { cfg = map.at(id.handle); });
    return cfg;
}

CounterController&
get_controller()
{
    static CounterController controller;
    return controller;
}

rocprofiler_status_t
create_counter_profile(std::shared_ptr<profile_config> config)
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    if(status = counters::counter_callback_info::setup_profile_config(config);
       status != ROCPROFILER_STATUS_SUCCESS)
    {
        return status;
    }

    if(status = config->pkt_generator->can_collect(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        return status;
    }

    get_controller().add_profile(std::move(config));

    return status;
}

void
destroy_counter_profile(uint64_t id)
{
    get_controller().destroy_profile(id);
}

std::shared_ptr<profile_config>
get_profile_config(rocprofiler_profile_config_id_t id)
{
    try
    {
        return get_controller().get_profile_cfg(id);
    } catch(std::out_of_range&)
    {
        return nullptr;
    }
}
}  // namespace counters
}  // namespace rocprofiler
