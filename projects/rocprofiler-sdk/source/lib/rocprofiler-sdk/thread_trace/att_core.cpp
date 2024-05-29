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
            ROCP_ERROR << "HSA Err: " << _status << '\n';                                          \
            throw std::runtime_error(message);                                                     \
        }                                                                                          \
    }

constexpr size_t ROCPROFILER_QUEUE_SIZE = 64;

namespace rocprofiler
{
struct cbdata_t
{
    void*                                  tool_userdata;
    rocprofiler_att_shader_data_callback_t cb_fn;
    rocprofiler_correlation_id_t           corr_id;
};

common::Synchronized<std::optional<int64_t>> client;

bool
AgentThreadTracer::Submit(hsa_ext_amd_aql_pm4_packet_t* packet)
{
    const uint64_t write_idx = add_write_index_relaxed_fn(queue, 1);

    size_t index      = (write_idx % queue->size) * sizeof(hsa_ext_amd_aql_pm4_packet_t);
    auto*  queue_slot = reinterpret_cast<uint32_t*>(size_t(queue->base_address) + index);  // NOLINT

    const auto* slot_data = reinterpret_cast<const uint32_t*>(packet);

    memcpy(&queue_slot[1], &slot_data[1], sizeof(hsa_ext_amd_aql_pm4_packet_t) - sizeof(uint32_t));
    auto* header = reinterpret_cast<std::atomic<uint32_t>*>(queue_slot);

    header->store(slot_data[0], std::memory_order_release);
    signal_store_screlease_fn(queue->doorbell_signal, write_idx);

    int loops = 0;
    while(load_read_index_relaxed_fn(queue) <= write_idx)
    {
        loops++;
        usleep(1);
        if(loops > 10000)  // Add loop limit to prevent hang. TODO: Remove once stability proven
        {
            ROCP_ERROR << "Codeobj packet submission failed!";
            return false;
        }
    }
    return true;
}

AgentThreadTracer::AgentThreadTracer(thread_trace_parameter_pack _params,
                                     const hsa::AgentCache&      cache,
                                     const CoreApiTable&         coreapi,
                                     const AmdExtTable&          ext)
: params(std::move(_params))
{
    factory = std::make_unique<aql::ThreadTraceAQLPacketFactory>(cache, this->params, coreapi, ext);
    cached_resources = factory->construct_packet();

    auto status = coreapi.hsa_queue_create_fn(cache.get_hsa_agent(),
                                              ROCPROFILER_QUEUE_SIZE,
                                              HSA_QUEUE_TYPE_SINGLE,
                                              nullptr,
                                              nullptr,
                                              UINT32_MAX,
                                              UINT32_MAX,
                                              &this->queue);
    if(status != HSA_STATUS_SUCCESS)
    {
        ROCP_ERROR << "Failed to create thread trace async queue";
        this->queue = nullptr;
    }

    queue_destroy_fn           = coreapi.hsa_queue_destroy_fn;
    signal_store_screlease_fn  = coreapi.hsa_signal_store_screlease_fn;
    add_write_index_relaxed_fn = coreapi.hsa_queue_add_write_index_relaxed_fn;
    load_read_index_relaxed_fn = coreapi.hsa_queue_load_read_index_relaxed_fn;
}

AgentThreadTracer::~AgentThreadTracer()
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);

    if(active_resources.packet != nullptr)
        ROCP_WARNING << "Thread tracer being destroyed with thread trace active";

    if(!this->queue) return;

    auto* packet = static_cast<hsa::TraceControlAQLPacket*>(active_resources.packet.get());
    if(packet)
    {
        packet->clear();
        packet->populate_after();

        for(auto& after_packet : packet->after_krn_pkt)
            Submit(&after_packet);
    }

    if(queue_destroy_fn) queue_destroy_fn(this->queue);
}

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
std::unique_ptr<hsa::AQLPacket>
AgentThreadTracer::pre_kernel_call(rocprofiler_att_control_flags_t control_flags,
                                   rocprofiler_queue_id_t          queue_id,
                                   rocprofiler_correlation_id_t    corr_id)
{
    if(control_flags == ROCPROFILER_ATT_CONTROL_NONE) return nullptr;

    std::unique_lock<std::mutex> lk(trace_resources_mut);

    if(control_flags == ROCPROFILER_ATT_CONTROL_STOP)
    {
        if(active_resources.packet == nullptr)
        {
            ROCP_ERROR << "Attempt at stopping a thread trace that has not started!\n";
            return nullptr;
        }

        active_resources.packet->clear();
        active_resources.packet->populate_after();
        data_is_ready.fetch_add(1);
        return std::move(active_resources.packet);
    }

    if(active_resources.packet != nullptr)
    {
        ROCP_ERROR << "Attempt at starting a thread trace while another was active!\n";
        return nullptr;
    }
    else
    {
        active_resources.corr_id  = corr_id;
        active_resources.queue_id = queue_id;
    }

    if(cached_resources == nullptr)
    {
        ROCP_ERROR << "Attempt to initialize ATT without allocated resources!\n";
        return nullptr;
    }

    cached_resources->clear();
    cached_resources->populate_before();

    if((control_flags & ROCPROFILER_ATT_CONTROL_STOP) != 0)
    {
        cached_resources->populate_after();
        data_is_ready.fetch_add(1);
    }

    return std::move(cached_resources);
}

hsa_status_t
thread_trace_callback(uint32_t shader, void* buffer, uint64_t size, void* callback_data)
{
    void* tool_userdata = static_cast<cbdata_t*>(callback_data)->tool_userdata;
    auto  callback_fn   = *static_cast<cbdata_t*>(callback_data)->cb_fn;

    callback_fn(shader, buffer, size, tool_userdata);
    return HSA_STATUS_SUCCESS;
}

void
AgentThreadTracer::post_kernel_call(std::unique_ptr<hsa::AQLPacket>&& aql)
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);

    active_resources.packet = std::move(aql);

    if(!active_resources.packet || data_is_ready.load() < 1) return;
    auto* pkt = static_cast<hsa::TraceControlAQLPacket*>(active_resources.packet.get());

    for(auto& record : remaining_codeobj_record)
    {
        if(!record.bUnload)
            pkt->add_codeobj(record.id, record.addr, record.size);
        else
            pkt->remove_codeobj(record.id);
    }
    remaining_codeobj_record.clear();

    cbdata_t cb_dt{};

    cb_dt.corr_id       = active_resources.corr_id;
    cb_dt.tool_userdata = params.callback_userdata;
    cb_dt.cb_fn         = params.shader_cb_fn;

    auto status = aqlprofile_att_iterate_data(pkt->GetHandle(), thread_trace_callback, &cb_dt);
    CHECK_HSA(status, "Failed to iterate ATT data");

    data_is_ready.fetch_sub(1);
    cached_resources = std::move(active_resources.packet);
}

void
AgentThreadTracer::load_codeobj(code_object_id_t id, uint64_t addr, uint64_t size)
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);

    if(auto* pkt = static_cast<hsa::TraceControlAQLPacket*>(cached_resources.get()))
    {
        pkt->add_codeobj(id, addr, size);
        return;
    }

    remaining_codeobj_record.push_back({id, addr, size, false});

    if(!queue) return;

    auto packet   = factory->construct_load_marker_packet(id, addr, size);
    bool bSuccess = Submit(&packet->packet);

    if(!bSuccess)  // If something went wrong, don't delete packet to avoid CP memory access fault
        packet.release();
}

void
AgentThreadTracer::unload_codeobj(code_object_id_t id)
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);

    if(auto* pkt = static_cast<hsa::TraceControlAQLPacket*>(cached_resources.get()))
    {
        pkt->remove_codeobj(id);
        return;
    }

    remaining_codeobj_record.push_back({id, 0, 0, true});

    if(!queue) return;

    auto packet   = factory->construct_unload_marker_packet(id);
    bool bSuccess = Submit(&packet->packet);

    if(!bSuccess)  // If something went wrong, don't delete packet to avoid CP memory access fault
        packet.release();
}

// TODO: make this a wrapper on HSA load instead of registering
void
GlobalThreadTracer::codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                             rocprofiler_user_data_t* /* user_data */,
                                             void* callback_data)
{
    if(!callback_data) return;
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_LOAD) return;

    auto* rec = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);
    assert(rec);

    GlobalThreadTracer& tracer = *static_cast<GlobalThreadTracer*>(callback_data);
    auto                agent  = rec->hsa_agent;

    std::shared_lock<std::shared_mutex> lk(tracer.agents_map_mut);

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
    {
        try
        {
            tracer.loaded_codeobjs.at(rec->hsa_agent).erase(rec->code_object_id);
        } catch(std::exception& e)
        {
            ROCP_WARNING << "Codeobj unload called for invalid ID " << rec->code_object_id;
        }
    }
    else
    {
        tracer.loaded_codeobjs[agent][rec->code_object_id] = {rec->load_delta, rec->load_size};
    }

    auto tracer_it = tracer.agents.find(agent);
    if(tracer_it == tracer.agents.end()) return;

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        tracer_it->second->load_codeobj(rec->code_object_id, rec->load_delta, rec->load_size);
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        tracer_it->second->unload_codeobj(rec->code_object_id);
}

void
GlobalThreadTracer::resource_init(const hsa::AgentCache& cache,
                                  const CoreApiTable&    coreapi,
                                  const AmdExtTable&     ext)
{
    auto                                agent = cache.get_hsa_agent();
    std::unique_lock<std::shared_mutex> lk(agents_map_mut);

    auto agent_it = agents.find(agent);
    if(agent_it != agents.end())
    {
        agent_it->second->active_queues.fetch_add(1);
        return;
    }

    auto new_tracer = std::make_unique<AgentThreadTracer>(this->params, cache, coreapi, ext);
    new_tracer->active_queues.store(1);
    agents.emplace(agent, std::move(new_tracer));
}

void
GlobalThreadTracer::resource_deinit(const hsa::AgentCache& cache)
{
    std::unique_lock<std::shared_mutex> lk(agents_map_mut);

    auto agent_it = agents.find(cache.get_hsa_agent());
    if(agent_it == agents.end()) return;

    if(agent_it->second->active_queues.fetch_sub(1) > 1) return;

    agents.erase(cache.get_hsa_agent());
}

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
std::unique_ptr<hsa::AQLPacket>
GlobalThreadTracer::pre_kernel_call(const hsa::Queue&              queue,
                                    rocprofiler_kernel_id_t        kernel_id,
                                    const context::correlation_id* corr_id)
{
    rocprofiler_correlation_id_t rocprof_corr_id =
        rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};

    if(corr_id) rocprof_corr_id.internal = corr_id->internal;
    // TODO: Get external

    auto control_flags = params.dispatch_cb_fn(queue.get_id(),
                                               queue.get_agent().get_rocp_agent(),
                                               rocprof_corr_id,
                                               kernel_id,
                                               params.callback_userdata);

    if(control_flags == ROCPROFILER_ATT_CONTROL_NONE) return nullptr;

    std::shared_lock<std::shared_mutex> lk(agents_map_mut);

    auto it = agents.find(queue.get_agent().get_hsa_agent());
    assert(it != agents.end() && it->second != nullptr);

    auto packet = it->second->pre_kernel_call(control_flags, queue.get_id(), rocprof_corr_id);
    if(packet != nullptr) post_move_data.fetch_add(1);
    return packet;
}

void
GlobalThreadTracer::post_kernel_call(GlobalThreadTracer::inst_pkt_t& aql)
{
    if(post_move_data.load() < 1) return;

    for(auto& aql_pkt : aql)
    {
        auto* pkt = dynamic_cast<hsa::TraceControlAQLPacket*>(aql_pkt.first.get());
        if(!pkt) continue;

        std::shared_lock<std::shared_mutex> lk(agents_map_mut);
        post_move_data.fetch_sub(1);

        auto it = agents.find(pkt->GetAgent());
        if(it != agents.end() && it->second != nullptr)
            it->second->post_kernel_call(std::move(aql_pkt.first));
    }
}

void
GlobalThreadTracer::start_context()
{
    if(codeobj_client_ctx.handle != 0)
    {
        auto status = rocprofiler_start_context(codeobj_client_ctx);
        if(status != ROCPROFILER_STATUS_SUCCESS) throw std::exception();
    }

    // Only one thread should be attempting to enable/disable this context
    client.wlock([&](auto& client_id) {
        if(client_id) return;

        client_id = hsa::get_queue_controller()->add_callback(
            std::nullopt,
            [=](const hsa::Queue& q,
                const hsa::rocprofiler_packet& /* kern_pkt */,
                rocprofiler_kernel_id_t kernel_id,
                rocprofiler_dispatch_id_t /* dispatch_id */,
                rocprofiler_user_data_t* /* user_data */,
                const corr_id_map_t& /* extern_corr_ids */,
                const context::correlation_id* corr_id) {
                return this->pre_kernel_call(q, kernel_id, corr_id);
            },
            [=](const hsa::Queue& /* q */,
                hsa::rocprofiler_packet /* kern_pkt */,
                const hsa::Queue::queue_info_session_t& /* session */,
                inst_pkt_t& aql) { this->post_kernel_call(aql); });
    });
}

void
GlobalThreadTracer::stop_context()
{
    client.wlock([&](auto& client_id) {
        if(!client_id) return;

        // Remove our callbacks from HSA's queue controller
        hsa::get_queue_controller()->remove_callback(*client_id);
        client_id = std::nullopt;
    });
}

}  // namespace rocprofiler
