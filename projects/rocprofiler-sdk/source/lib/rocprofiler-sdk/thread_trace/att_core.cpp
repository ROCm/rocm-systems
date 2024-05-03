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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_HSA(fn, message)                                                                     \
    {                                                                                              \
        auto _status = (fn);                                                                       \
        if(_status != HSA_STATUS_SUCCESS)                                                          \
        {                                                                                          \
            LOG(ERROR) << "HSA Err: " << _status << '\n';                                          \
            throw std::runtime_error(message);                                                     \
        }                                                                                          \
    }

namespace rocprofiler
{
using AQLPacketOwner = std::unique_ptr<hsa::AQLPacket>;
using inst_pkt_t     = common::container::small_vector<std::pair<AQLPacketOwner, int64_t>, 4>;
using corr_id_map_t  = hsa::Queue::queue_info_session_t::external_corr_id_map_t;

struct cbdata_t
{
    void*                                  tool_userdata;
    rocprofiler_att_shader_data_callback_t cb_fn;
    std::vector<uint8_t>*                  memory_space;
};

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
AQLPacketOwner
pre_kernel_call(ThreadTracer&                  tracer,
                const hsa::Queue&              queue,
                const hsa::rocprofiler_packet& kern_pkt,
                uint64_t                       kernel_id,
                const corr_id_map_t&           extern_corr_ids,
                const context::correlation_id* corr_id)
{
    (void) extern_corr_ids;
    (void) corr_id;

    rocprofiler_correlation_id_t temp_corr_id;
    temp_corr_id.internal       = 0;
    temp_corr_id.external.value = 0;
    temp_corr_id.external.ptr   = nullptr;

    auto control_flags = tracer.params->dispatch_cb_fn(queue.get_id(),
                                                       queue.get_agent().get_rocp_agent(),
                                                       temp_corr_id,
                                                       &kern_pkt.kernel_dispatch,
                                                       kernel_id,
                                                       tracer.params->callback_userdata);

    if(control_flags == ROCPROFILER_ATT_CONTROL_NONE) return nullptr;

    assert(control_flags == ROCPROFILER_ATT_CONTROL_START_AND_STOP && "Error: Not implemented");

    uint64_t                    agent = queue.get_agent().get_hsa_agent().handle;
    std::lock_guard<std::mutex> lk(tracer.trace_resources_mut);

    try
    {
        auto moved = std::move(tracer.resources.at(agent));
        tracer.resources.erase(agent);
        return moved;
    } catch(std::out_of_range& e)
    {
        LOG(WARNING) << "Attempt to initialize ATT without allocated resources!\n";
        return nullptr;
    }
}

hsa_status_t
thread_trace_callback(uint32_t shader, void* buffer, uint64_t size, void* callback_data)
{
    void*                 tool_userdata = static_cast<cbdata_t*>(callback_data)->tool_userdata;
    auto                  callback_fn   = *static_cast<cbdata_t*>(callback_data)->cb_fn;
    std::vector<uint8_t>& cpu_data      = *static_cast<cbdata_t*>(callback_data)->memory_space;

    // TODO(gbaraldi): Handle parallel callbacks
    static std::mutex           mut;
    std::lock_guard<std::mutex> lk(mut);

    if(size > cpu_data.size()) cpu_data.resize(size + cpu_data.size());

    auto status = hsa::get_queue_controller()->get_core_table().hsa_memory_copy_fn(
        cpu_data.data(), buffer, size);
    if(status != HSA_STATUS_SUCCESS)
    {
        LOG(WARNING) << "Failed to copy hsa memory!";
        return HSA_STATUS_SUCCESS;
    }

    callback_fn(shader, cpu_data.data(), size, tool_userdata);
    return HSA_STATUS_SUCCESS;
}

void
post_kernel_call(ThreadTracer& tracer, inst_pkt_t& aql)
{
    std::vector<uint8_t> cpu_data{};
    auto pair = cbdata_t{tracer.params->callback_userdata, tracer.params->shader_cb_fn, &cpu_data};

    for(auto& aql_pkt : aql)
    {
        auto* pkt = dynamic_cast<hsa::TraceAQLPacket*>(aql_pkt.first.get());
        if(!pkt) continue;

        auto status = aqlprofile_att_iterate_data(pkt->GetHandle(), thread_trace_callback, &pair);
        CHECK_HSA(status, "Failed to iterate ATT data");

        std::lock_guard<std::mutex> lk(tracer.trace_resources_mut);
        if(tracer.agent_active_queues.find(pkt->GetAgent()) != tracer.agent_active_queues.end())
            tracer.resources[pkt->GetAgent()] = std::move(aql_pkt.first);
    }
}

common::Synchronized<std::optional<int64_t>> client;

void
ThreadTracer::start_context()
{
    // Only one thread should be attempting to enable/disable this context
    client.wlock([&](auto& client_id) {
        if(client_id) return;

        client_id = hsa::get_queue_controller()->add_callback(
            std::nullopt,
            [=](const hsa::Queue&                                               q,
                const hsa::rocprofiler_packet&                                  kern_pkt,
                rocprofiler_kernel_id_t                                         kernel_id,
                rocprofiler_dispatch_id_t                                       dispatch_id,
                rocprofiler_user_data_t*                                        user_data,
                const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
                const context::correlation_id*                                  corr_id) {
                return pre_kernel_call(*this, q, kern_pkt, kernel_id, extern_corr_ids, corr_id);
                (void) user_data;
                (void) dispatch_id;
            },
            [=](const hsa::Queue&                       q,
                hsa::rocprofiler_packet                 kern_pkt,
                const hsa::Queue::queue_info_session_t& session,
                inst_pkt_t&                             aql) {
                post_kernel_call(*this, aql);
                (void) session;
                (void) kern_pkt;
                (void) q;
            });
    });
}

void
ThreadTracer::stop_context()
{
    client.wlock([&](auto& client_id) {
        if(!client_id) return;

        // Remove our callbacks from HSA's queue controller
        hsa::get_queue_controller()->remove_callback(*client_id);
        client_id = std::nullopt;
    });
}

void
ThreadTracer::resource_init(const hsa::AgentCache& cache,
                            const CoreApiTable&    coreapi,
                            const AmdExtTable&     ext)
{
    uint64_t                    agent = cache.get_hsa_agent().handle;
    std::lock_guard<std::mutex> lk(trace_resources_mut);

    if(agent_active_queues.find(agent) != agent_active_queues.end())
    {
        agent_active_queues.at(agent).fetch_add(1);
        return;
    }

    auto factory     = aql::ThreadTraceAQLPacketFactory(cache, this->params, coreapi, ext);
    resources[agent] = factory.construct_packet();
    agent_active_queues[agent] = 1;
}

void
ThreadTracer::resource_deinit(const hsa::AgentCache& cache)
{
    uint64_t                    agent = cache.get_hsa_agent().handle;
    std::lock_guard<std::mutex> lk(trace_resources_mut);

    try
    {
        if(agent_active_queues.at(agent).fetch_add(-1) > 1) return;
    } catch(std::out_of_range&)
    {}

    agent_active_queues.erase(agent);
    resources.erase(agent);
}

}  // namespace rocprofiler
