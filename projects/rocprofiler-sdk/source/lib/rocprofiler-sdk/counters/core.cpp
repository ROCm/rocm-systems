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

#include "lib/rocprofiler-sdk/counters/core.hpp"

#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/aql/packet_construct.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/dispatch_handlers.hpp"
#include "lib/rocprofiler-sdk/counters/sample_processing.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>

namespace rocprofiler
{
namespace counters
{
rocprofiler_status_t
counter_callback_info::setup_counter_config(std::shared_ptr<counter_config>& profile)
{
    if(profile->pkt_generator || !profile->reqired_hw_counters.empty())
    {
        return ROCPROFILER_STATUS_SUCCESS;
    }

    // Sets up the packet generator for the profile. This must be delayed until after HSA is loaded.
    // This call needs to be thread protected in that only one thread must be setting up profile at
    // the same time.

    auto& config     = *profile;
    auto  agent_name = std::string(config.agent->name);
    for(const auto& metric : config.metrics)
    {
        const auto asts = get_ast_map();
        auto       req_counters =
            get_required_hardware_counters(asts->arch_to_counter_asts, agent_name, metric);

        if(!req_counters)
        {
            ROCP_ERROR << fmt::format("Could not find counter {}", metric.name());
            return ROCPROFILER_STATUS_ERROR_PROFILE_COUNTER_NOT_FOUND;
        }

        // Special metrics are those that are not hw counters but other
        // constants like MAX_WAVE_SIZE
        for(const auto& req_metric : *req_counters)
        {
            if(req_metric.constant().empty())
            {
                config.reqired_hw_counters.insert(req_metric);
            }
            else
            {
                config.required_special_counters.insert(req_metric);
            }
        }

        const auto* agent_map =
            rocprofiler::common::get_val(asts->arch_to_counter_asts, agent_name);
        if(!agent_map)
        {
            ROCP_ERROR << fmt::format("Could not build AST for {}", agent_name);
            return ROCPROFILER_STATUS_ERROR_AST_GENERATION_FAILED;
        }

        const auto* counter_ast = rocprofiler::common::get_val(*agent_map, metric.name());
        if(!counter_ast)
        {
            ROCP_ERROR << fmt::format("Could not find AST for {}", metric.name());
            return ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND;
        }
        config.asts.push_back(*counter_ast);

        try
        {
            config.asts.back().set_dimensions(config.agent->id);
        } catch(std::runtime_error& e)
        {
            ROCP_ERROR << metric.name() << " has improper dimensions"
                       << " " << e.what();
            return ROCPROFILER_STATUS_ERROR_AST_NOT_FOUND;
        }
    }

    profile->pkt_generator = std::make_unique<rocprofiler::aql::CounterPacketConstruct>(
        config.agent->id,
        std::vector<counters::Metric>{profile->reqired_hw_counters.begin(),
                                      profile->reqired_hw_counters.end()});
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
counter_callback_info::get_packet(std::unique_ptr<rocprofiler::hsa::AQLPacket>& ret_pkt,
                                  std::shared_ptr<counter_config>&              profile)
{
    rocprofiler_status_t status;
    // Check packet cache
    profile->packets.wlock([&](auto& pkt_vector) {
        status = counter_callback_info::setup_counter_config(profile);
        if(!pkt_vector.empty() && status == ROCPROFILER_STATUS_SUCCESS)
        {
            ret_pkt = std::move(pkt_vector.back());
            pkt_vector.pop_back();
        }
    });

    if(status != ROCPROFILER_STATUS_SUCCESS) return status;
    if(!ret_pkt)
    {
        // If we do not have a packet in the cache, create one.
        ret_pkt = profile->pkt_generator->construct_packet(
            CHECK_NOTNULL(hsa::get_queue_controller())->get_core_table(),
            CHECK_NOTNULL(hsa::get_queue_controller())->get_ext_table());
    }

    ret_pkt->clear();
    packet_return_map.wlock([&](auto& data) { data.emplace(ret_pkt.get(), profile); });

    return ROCPROFILER_STATUS_SUCCESS;
}

void
start_context(const context::context* ctx)
{
    if(!ctx || !ctx->dispatch_counter_collection) return;

    auto* controller = hsa::get_queue_controller();

    bool already_enabled = true;
    CHECK_NOTNULL(controller)->enable_serialization();
    ctx->dispatch_counter_collection->enabled.wlock([&](auto& enabled) {
        if(enabled) return;
        already_enabled = false;
        enabled         = true;
    });
    if(!already_enabled)
    {
        // #8586: per-queue callback registration removed; counter instrumentation now
        // flows through counters::write_hook. Keep develop's callback-thread startup.
        callback_thread_start();
    }
}

void
stop_context(const context::context* ctx)
{
    if(!ctx || !ctx->dispatch_counter_collection) return;

    auto* controller = hsa::get_queue_controller();

    ctx->dispatch_counter_collection->enabled.wlock([&](auto& enabled) {
        if(!enabled) return;
        enabled = false;
    });

    if(controller)
    {
        controller->disable_serialization();

        // #8586: per-queue counter callbacks were removed; counter instrumentation now
        // runs via counters::write_hook, which no-ops once dispatch_counter_collection is
        // disabled above. The attach-mode detach-drain teardown is therefore implicit.
    }

    callback_thread_stop();
}

rocprofiler_status_t
configure_agent_collection(rocprofiler_context_id_t                 context_id,
                           rocprofiler_buffer_id_t                  buffer_id,
                           rocprofiler_agent_id_t                   agent_id,
                           rocprofiler_device_counting_service_cb_t cb,
                           void*                                    user_data)
{
    return get_controller().configure_agent_collection(
        context_id, buffer_id, agent_id, cb, user_data);
}

rocprofiler_status_t
configure_buffered_dispatch(rocprofiler_context_id_t                   context_id,
                            rocprofiler_buffer_id_t                    buffer,
                            rocprofiler_dispatch_counting_service_cb_t callback,
                            void*                                      callback_args)
{
    CHECK_NE(buffer.handle, 0ULL);
    return get_controller().configure_dispatch(
        context_id, buffer, callback, callback_args, nullptr, nullptr);
}

rocprofiler_status_t
configure_callback_dispatch(rocprofiler_context_id_t                   context_id,
                            rocprofiler_dispatch_counting_service_cb_t callback,
                            void*                                      callback_data_args,
                            rocprofiler_dispatch_counting_record_cb_t  record_callback,
                            void*                                      record_callback_args)
{
    return get_controller().configure_dispatch(context_id,
                                               {.handle = 0},
                                               callback,
                                               callback_data_args,
                                               record_callback,
                                               record_callback_args);
}

}  // namespace counters
}  // namespace rocprofiler
