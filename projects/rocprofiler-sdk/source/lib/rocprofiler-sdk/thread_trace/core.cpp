// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hsa.h>
#include <rocprofiler-sdk/intercept_table.h>

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

namespace rocprofiler
{
namespace thread_trace
{
constexpr size_t   QUEUE_SIZE      = 128;
constexpr uint64_t MIN_BUFFER_SIZE = 1 << 20;  // 1MB

struct cbdata_t
{
    rocprofiler_agent_id_t                          agent;
    rocprofiler_thread_trace_shader_data_callback_t cb_fn;
    const rocprofiler_user_data_t*                  userdata;
};

common::Synchronized<std::optional<int64_t>> client;

bool
thread_trace_parameter_pack::are_params_valid() const
{
    if(shader_cb_fn == nullptr)
    {
        ROCP_CI_LOG(WARNING) << "Callback cannot be null!";
        return false;
    }

    if(shader_engine_mask == 0) return false;

    if(buffer_size < MIN_BUFFER_SIZE)
    {
        ROCP_CI_LOG(WARNING) << "Invalid buffer size: " << buffer_size;
        return false;
    }

    if(target_cu > 0xF) return false;
    if(simd_select > 0xF) return false;  // Only 16 CUs and 4 SIMDs

    return true;
}

class Signal
{
public:
    Signal(hsa_ext_amd_aql_pm4_packet_t* packet)
    {
        auto& core = *hsa::get_core_table();
        auto& ext  = *hsa::get_amd_ext_table();
        ext.hsa_amd_signal_create_fn(0, 0, nullptr, 0, &signal);
        packet->completion_signal = signal;
        core.hsa_signal_store_screlease_fn(signal, 1);
    }
    ~Signal()
    {
        WaitOn();
        hsa::get_core_table()->hsa_signal_destroy_fn(signal);
    }
    Signal(Signal& other)       = delete;
    Signal(const Signal& other) = delete;
    Signal& operator=(Signal& other) = delete;
    Signal& operator=(const Signal& other) = delete;

    void WaitOn() const
    {
        auto wait_fn = hsa::get_core_table()->hsa_signal_wait_scacquire_fn;
        while(wait_fn(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0)
        {}
    }

    hsa_signal_t      signal;
    std::atomic<bool> released{false};
};

std::unique_ptr<Signal>
ThreadTracerQueue::Submit(hsa_ext_amd_aql_pm4_packet_t* packet, bool bWait) const
{
    auto* core = hsa::get_core_table();

    std::unique_ptr<Signal> signal{};
    const uint64_t          write_idx = core->hsa_queue_add_write_index_relaxed_fn(queue, 1);

    size_t index = (write_idx % queue->size) * sizeof(hsa_ext_amd_aql_pm4_packet_t);
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    auto* queue_slot = reinterpret_cast<uint32_t*>(size_t(queue->base_address) + index);

    const auto* slot_data = reinterpret_cast<const uint32_t*>(packet);

    memcpy(&queue_slot[1], &slot_data[1], sizeof(hsa_ext_amd_aql_pm4_packet_t) - sizeof(uint32_t));
    if(bWait)
        signal =
            std::make_unique<Signal>(reinterpret_cast<hsa_ext_amd_aql_pm4_packet_t*>(queue_slot));
    auto* header = reinterpret_cast<std::atomic<uint32_t>*>(queue_slot);

    header->store(slot_data[0], std::memory_order_release);
    core->hsa_signal_store_screlease_fn(queue->doorbell_signal, write_idx);

    return signal;
}

ThreadTracerQueue::ThreadTracerQueue(thread_trace_parameter_pack _params,
                                     rocprofiler_agent_id_t      cache)
: params(std::move(_params))
, agent_id(cache)
{
    ROCP_TRACE << "Constructing ATT instance for agent " << agent_id.handle;
    auto* core = hsa::get_core_table();
    auto* ext  = hsa::get_amd_ext_table();

    factory = std::make_unique<aql::ThreadTraceAQLPacketFactory>(
        *rocprofiler::agent::get_agent_cache(rocprofiler::agent::get_agent(agent_id)),
        this->params,
        *core,
        *ext);
    control_packet = factory->construct_control_packet();

    auto hsa_agent = rocprofiler::agent::get_hsa_agent(agent_id);
    CHECK(hsa_agent.has_value());

    auto status = core->hsa_queue_create_fn(*hsa_agent,
                                            QUEUE_SIZE,
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

    codeobj_reg = std::make_unique<code_object::CodeobjCallbackRegistry>(
        [this](rocprofiler_agent_id_t agent, uint64_t codeobj_id, uint64_t addr, uint64_t size) {
            if(agent == this->agent_id) this->load_codeobj(codeobj_id, addr, size);
        },
        [this](uint64_t codeobj_id) { this->unload_codeobj(codeobj_id); });

    codeobj_reg->IterateLoaded();
}

ThreadTracerQueue::~ThreadTracerQueue()
{
    ROCP_TRACE << "Destroying ATT Queue...";
    std::unique_lock<std::mutex> lk(trace_resources_mut);
    if(active_traces.load() < 1)
    {
        hsa::get_core_table()->hsa_queue_destroy_fn(this->queue);
        return;
    }

    ROCP_CI_LOG(WARNING) << "Thread tracer being destroyed with thread trace active";

    control_packet->clear();
    control_packet->populate_after();

    std::vector<std::unique_ptr<Signal>> wait_idx{};

    for(auto& after_packet : control_packet->after_krn_pkt)
        wait_idx.emplace_back(Submit(&after_packet, true));
}

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
std::unique_ptr<hsa::TraceControlAQLPacket>
ThreadTracerQueue::get_control(bool bStart)
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);

    auto active_resources = std::make_unique<hsa::TraceControlAQLPacket>(*control_packet);
    active_resources->clear();

    if(bStart) active_traces.fetch_add(1);

    return active_resources;
}

hsa_status_t
thread_trace_callback(uint32_t shader, void* buffer, uint64_t size, void* callback_data)
{
    auto& cb_data = *static_cast<cbdata_t*>(callback_data);

    cb_data.cb_fn(cb_data.agent, shader, buffer, size, *cb_data.userdata);
    return HSA_STATUS_SUCCESS;
}

void
ThreadTracerQueue::iterate_data(aqlprofile_handle_t handle, rocprofiler_user_data_t data)
{
    cbdata_t cb_dt{};

    cb_dt.agent    = agent_id;
    cb_dt.cb_fn    = params.shader_cb_fn;
    cb_dt.userdata = &data;

    auto status = aqlprofile_att_iterate_data(handle, thread_trace_callback, &cb_dt);
    if(status == HSA_STATUS_ERROR_OUT_OF_RESOURCES)
        ROCP_WARNING << "Thread trace buffer full!";
    else
        CHECK_HSA(status, "Failed to iterate ATT data");

    active_traces.fetch_sub(1);
}

void
ThreadTracerQueue::load_codeobj(code_object_id_t id, uint64_t addr, uint64_t size)
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);

    control_packet->add_codeobj(id, addr, size);

    if(!queue || active_traces.load() < 1) return;

    auto packet = factory->construct_load_marker_packet(id, addr, size);
    Submit(&packet->packet, true)->WaitOn();
}

void
ThreadTracerQueue::unload_codeobj(code_object_id_t id)
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);

    if(!control_packet->remove_codeobj(id)) return;
    if(!queue || active_traces.load() < 1) return;

    auto packet = factory->construct_unload_marker_packet(id);
    Submit(&packet->packet, true)->WaitOn();
}

void
DispatchThreadTracer::resource_init()
{
    auto rocp_agents = rocprofiler::agent::get_agents();

    auto lk = std::unique_lock{agents_map_mut};

    for(const auto* rocp_agent : rocp_agents)
    {
        auto it = params.find(rocp_agent->id);
        if(it == params.end()) continue;

        auto cache = rocprofiler::agent::get_hsa_agent(rocp_agent);
        if(!cache.has_value())
        {
            ROCP_CI_LOG(TRACE) << "Could not find HSA Agent for " << rocp_agent->id.handle;
            continue;
        }
        agents[*cache] = std::make_unique<ThreadTracerQueue>(it->second, rocp_agent->id);
    }
}

void
DispatchThreadTracer::resource_deinit()
{
    ROCP_TRACE << "Clearing agents";
    auto lk = std::unique_lock{agents_map_mut};
    agents.clear();
}

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
hsa::Queue::pkt_and_serialize_t
DispatchThreadTracer::pre_kernel_call(const hsa::Queue&              queue,
                                      rocprofiler_kernel_id_t        kernel_id,
                                      rocprofiler_dispatch_id_t      dispatch_id,
                                      rocprofiler_user_data_t*       user_data,
                                      const context::correlation_id* corr_id)
{
    rocprofiler_async_correlation_id_t rocprof_corr_id =
        rocprofiler_async_correlation_id_t{.internal = 0, .external = context::null_user_data};

    if(corr_id)
    {
        rocprof_corr_id.internal = corr_id->internal;
    }
    // TODO: Get external

    std::shared_lock<std::shared_mutex> lk(agents_map_mut);

    auto it = agents.find(queue.get_agent().get_hsa_agent());

    if(it == agents.end()) return {nullptr, false};

    auto&       agent      = *CHECK_NOTNULL(it->second);
    const auto& parameters = agent.params;

    auto control_flags = parameters.dispatch_cb_fn(queue.get_agent().get_rocp_agent()->id,
                                                   queue.get_id(),
                                                   rocprof_corr_id,
                                                   kernel_id,
                                                   dispatch_id,
                                                   parameters.callback_userdata.ptr,
                                                   user_data);

    if(control_flags == ROCPROFILER_THREAD_TRACE_CONTROL_NONE)
        return {nullptr, parameters.bSerialize};

    auto packet = agent.get_control(true);
    post_move_data.fetch_add(1);
    packet->populate_before();
    packet->populate_after();
    return {std::move(packet), true};
}

void
DispatchThreadTracer::post_kernel_call(DispatchThreadTracer::inst_pkt_t&       aql,
                                       const hsa::Queue::queue_info_session_t& session)
{
    if(post_move_data.load() < 1) return;

    for(auto& aql_pkt : aql)
    {
        auto* pkt = dynamic_cast<hsa::TraceControlAQLPacket*>(aql_pkt.first.get());
        if(!pkt) continue;

        std::shared_lock<std::shared_mutex> lk(agents_map_mut);
        post_move_data.fetch_sub(1);

        if(pkt->after_krn_pkt.empty()) continue;

        auto it = agents.find(pkt->GetAgent());
        if(it != agents.end() && it->second != nullptr)
            it->second->iterate_data(pkt->GetHandle(), session.user_data);
    }
}

void
DispatchThreadTracer::start_context()
{
    using corr_id_map_t = hsa::Queue::queue_info_session_t::external_corr_id_map_t;

    CHECK_NOTNULL(hsa::get_queue_controller())->enable_serialization();

    // Only one thread should be attempting to enable/disable this context
    client.wlock([&](auto& client_id) {
        if(client_id) return;

        client_id =
            CHECK_NOTNULL(hsa::get_queue_controller())
                ->add_callback(
                    std::nullopt,
                    [=](const hsa::Queue& q,
                        const hsa::rocprofiler_packet& /* kern_pkt */,
                        rocprofiler_kernel_id_t   kernel_id,
                        rocprofiler_dispatch_id_t dispatch_id,
                        rocprofiler_user_data_t*  user_data,
                        const corr_id_map_t& /* extern_corr_ids */,
                        const context::correlation_id* corr_id) {
                        return this->pre_kernel_call(q, kernel_id, dispatch_id, user_data, corr_id);
                    },
                    [=](const hsa::Queue& /* q */,
                        hsa::rocprofiler_packet /* kern_pkt */,
                        std::shared_ptr<hsa::Queue::queue_info_session_t>& session,
                        inst_pkt_t&                                        aql,
                        kernel_dispatch::profiling_time) {
                        this->post_kernel_call(aql, *session);
                    });
    });
}

void
DispatchThreadTracer::stop_context()  // NOLINT(readability-convert-member-functions-to-static)
{
    auto* controller = hsa::get_queue_controller();
    if(!controller) return;

    client.wlock([&](auto& client_id) {
        if(!client_id) return;

        // Remove our callbacks from HSA's queue controller
        controller->remove_callback(*client_id);
        client_id = std::nullopt;
    });

    controller->disable_serialization();
}

void
DeviceThreadTracer::resource_init()
{
    auto rocp_agents = rocprofiler::agent::get_agents();

    std::unique_lock<std::mutex> lk(agent_mut);

    for(const auto* rocp_agent : rocp_agents)
    {
        auto it = params.find(CHECK_NOTNULL(rocp_agent)->id);
        if(it == params.end()) continue;

        agents[it->first] = std::make_unique<ThreadTracerQueue>(it->second, rocp_agent->id);
    }
}

void
DeviceThreadTracer::resource_deinit()
{
    ROCP_TRACE << "Clearing agents";
    std::unique_lock<std::mutex> lk(agent_mut);
    agents.clear();
}

void
DeviceThreadTracer::start_context()
{
    ROCP_TRACE << "Start device context";
    std::unique_lock<std::mutex> lk(agent_mut);

    if(agents.empty())
    {
        ROCP_WARNING << "Thread trace context not present for agent!";
        return;
    }

    std::vector<std::unique_ptr<Signal>> wait_list{};

    for(auto& [_, tracer] : agents)
    {
        auto packet = tracer->get_control(true);
        packet->populate_before();

        auto sig = tracer->SubmitAndSignalLast(packet->before_krn_pkt);
        if(sig) wait_list.emplace_back(std::move(sig));
    }
}

void
DeviceThreadTracer::stop_context()
{
    using wait_t = std::tuple<ThreadTracerQueue*, aqlprofile_handle_t, std::unique_ptr<Signal>>;
    std::unique_lock<std::mutex> lk(agent_mut);

    if(agents.empty())
    {
        ROCP_WARNING << "Thread trace context not present for agent!";
        return;
    }

    std::vector<wait_t> wait_list{};

    for(auto& [_, tracer] : agents)
    {
        auto packet = tracer->get_control(false);
        packet->populate_after();

        auto signal = tracer->SubmitAndSignalLast(packet->after_krn_pkt);
        if(signal) wait_list.emplace_back(tracer.get(), packet->GetHandle(), std::move(signal));
    }

    for(auto& [tracer, handle, signal] : wait_list)
    {
        signal->WaitOn();
        tracer->iterate_data(handle, tracer->params.callback_userdata);
    }
}

void
initialize(HsaApiTable* table)
{
    ROCP_FATAL_IF(!table->core_ || !table->amd_ext_);

    for(auto& ctx : context::get_registered_contexts())
    {
        if(ctx->device_thread_trace) ctx->device_thread_trace->resource_init();
        if(ctx->dispatch_thread_trace) ctx->dispatch_thread_trace->resource_init();
    }
}

void
finalize()
{
    ROCP_TRACE << "Finalize called";
    for(auto& ctx : context::get_registered_contexts())
    {
        if(ctx->device_thread_trace) ctx->device_thread_trace->resource_deinit();
        if(ctx->dispatch_thread_trace) ctx->dispatch_thread_trace->resource_deinit();
    }

    code_object::finalize();
}

}  // namespace thread_trace

}  // namespace rocprofiler
