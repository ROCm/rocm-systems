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

namespace rocprofiler
{
namespace thread_trace
{
constexpr size_t   QUEUE_SIZE      = 128;
constexpr uint64_t MIN_BUFFER_SIZE = 1 << 18;                              // 2 pages per SE
constexpr uint64_t MAX_BUFFER_SIZE = std::numeric_limits<int32_t>::max();  // aqlprofile limit

struct cbdata_t
{
    rocprofiler_att_shader_data_callback_t cb_fn;
    const rocprofiler_user_data_t*         dispatch_userdata;
};

common::Synchronized<std::optional<int64_t>> client;

CoreApiTable&
get_core()
{
    static CoreApiTable api{};
    return api;
}

AmdExtTable&
get_ext()
{
    static AmdExtTable api{};
    return api;
}

bool
thread_trace_parameter_pack::are_params_valid() const
{
    if(shader_cb_fn == nullptr)
    {
        ROCP_WARNING << "Callback cannot be null!";
        return false;
    }

    if(shader_engine_mask == 0) return false;

    if(buffer_size > MAX_BUFFER_SIZE || buffer_size < MIN_BUFFER_SIZE)
    {
        ROCP_WARNING << "Invalid buffer size: " << buffer_size;
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
        get_ext().hsa_amd_signal_create_fn(0, 0, nullptr, 0, &signal);
        packet->completion_signal = signal;
        get_core().hsa_signal_store_screlease_fn(signal, 1);
    }
    ~Signal()
    {
        WaitOn();
        get_core().hsa_signal_destroy_fn(signal);
    }
    Signal(Signal& other)       = delete;
    Signal(const Signal& other) = delete;
    Signal& operator=(Signal& other) = delete;
    Signal& operator=(const Signal& other) = delete;

    void WaitOn() const
    {
        auto* wait_fn = get_core().hsa_signal_wait_scacquire_fn;
        while(wait_fn(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED) != 0)
        {}
    }

    hsa_signal_t      signal;
    std::atomic<bool> released{false};
};

std::unique_ptr<Signal>
ThreadTracerQueue::Submit(hsa_ext_amd_aql_pm4_packet_t* packet, bool bWait)
{
    std::unique_ptr<Signal> signal{};
    const uint64_t          write_idx = add_write_index_relaxed_fn(queue, 1);

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
    signal_store_screlease_fn(queue->doorbell_signal, write_idx);

    return signal;
}

ThreadTracerQueue::ThreadTracerQueue(thread_trace_parameter_pack _params,
                                     const hsa::AgentCache&      cache,
                                     const CoreApiTable&         coreapi,
                                     const AmdExtTable&          ext)
: params(std::move(_params))
, agent_id(cache.get_rocp_agent()->id)
{
    factory = std::make_unique<aql::ThreadTraceAQLPacketFactory>(cache, this->params, coreapi, ext);
    control_packet = factory->construct_control_packet();

    auto status = coreapi.hsa_queue_create_fn(cache.get_hsa_agent(),
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

    queue_destroy_fn           = coreapi.hsa_queue_destroy_fn;
    signal_store_screlease_fn  = coreapi.hsa_signal_store_screlease_fn;
    add_write_index_relaxed_fn = coreapi.hsa_queue_add_write_index_relaxed_fn;
    load_read_index_relaxed_fn = coreapi.hsa_queue_load_read_index_relaxed_fn;

    codeobj_reg = std::make_unique<code_object::CodeobjCallbackRegistry>(
        [this](rocprofiler_agent_id_t agent, uint64_t codeobj_id, uint64_t addr, uint64_t size) {
            if(agent == this->agent_id) this->load_codeobj(codeobj_id, addr, size);
        },
        [this](uint64_t codeobj_id) { this->unload_codeobj(codeobj_id); });

    codeobj_reg->IterateLoaded();
}

ThreadTracerQueue::~ThreadTracerQueue()
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);
    if(active_traces.load() < 1)
    {
        if(queue_destroy_fn) queue_destroy_fn(this->queue);
        return;
    }

    ROCP_WARNING << "Thread tracer being destroyed with thread trace active";

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

    cb_data.cb_fn(shader, buffer, size, *cb_data.dispatch_userdata);
    return HSA_STATUS_SUCCESS;
}

void
ThreadTracerQueue::iterate_data(aqlprofile_handle_t handle, rocprofiler_user_data_t data)
{
    cbdata_t cb_dt{};

    cb_dt.cb_fn             = params.shader_cb_fn;
    cb_dt.dispatch_userdata = &data;

    auto status = aqlprofile_att_iterate_data(handle, thread_trace_callback, &cb_dt);
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
DispatchThreadTracer::resource_init(const hsa::AgentCache& cache,
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

    auto new_tracer = std::make_unique<ThreadTracerQueue>(this->params, cache, coreapi, ext);
    agents.emplace(agent, std::move(new_tracer));
}

void
DispatchThreadTracer::resource_deinit(const hsa::AgentCache& cache)
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
DispatchThreadTracer::pre_kernel_call(const hsa::Queue&              queue,
                                      rocprofiler_kernel_id_t        kernel_id,
                                      rocprofiler_dispatch_id_t      dispatch_id,
                                      rocprofiler_user_data_t*       user_data,
                                      const context::correlation_id* corr_id)
{
    rocprofiler_correlation_id_t rocprof_corr_id =
        rocprofiler_correlation_id_t{.internal = 0, .external = context::null_user_data};

    if(corr_id) rocprof_corr_id.internal = corr_id->internal;
    // TODO: Get external

    // Maybe adds serialization packets to the AQLPacket (if serializer is enabled)
    // and maybe adds barrier packets if the state is transitioning from serialized <->
    // unserialized
    auto maybe_add_serialization = [&](auto& gen_pkt) {
        CHECK_NOTNULL(hsa::get_queue_controller())
            ->serializer(&queue)
            .rlock([&](const auto& serializer) {
                for(auto& s_pkt : serializer.kernel_dispatch(queue))
                    gen_pkt->before_krn_pkt.push_back(s_pkt.ext_amd_aql_pm4);
            });
    };

    auto control_flags = params.dispatch_cb_fn(queue.get_id(),
                                               queue.get_agent().get_rocp_agent(),
                                               rocprof_corr_id,
                                               kernel_id,
                                               dispatch_id,
                                               user_data,
                                               params.callback_userdata);

    if(control_flags == ROCPROFILER_ATT_CONTROL_NONE)
    {
        auto empty = std::make_unique<hsa::EmptyAQLPacket>();
        maybe_add_serialization(empty);
        return empty;
    }

    std::shared_lock<std::shared_mutex> lk(agents_map_mut);

    auto it = agents.find(queue.get_agent().get_hsa_agent());
    assert(it != agents.end() && it->second != nullptr);

    auto packet = it->second->get_control(true);

    post_move_data.fetch_add(1);
    maybe_add_serialization(packet);

    packet->populate_before();
    packet->populate_after();
    return packet;
}

class SignalSerializerExit
{
public:
    SignalSerializerExit(const hsa::Queue::queue_info_session_t& _session)
    : session(_session)
    {}
    ~SignalSerializerExit()
    {
        auto* controller = hsa::get_queue_controller();
        if(!controller) return;

        controller->serializer(&session.queue).wlock([&](auto& serializer) {
            serializer.kernel_completion_signal(session.queue);
        });
    }
    const hsa::Queue::queue_info_session_t& session;
};

void
DispatchThreadTracer::post_kernel_call(DispatchThreadTracer::inst_pkt_t&       aql,
                                       const hsa::Queue::queue_info_session_t& session)
{
    SignalSerializerExit signal(session);

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

        client_id = hsa::get_queue_controller()->add_callback(
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
                const hsa::Queue::queue_info_session_t& session,
                inst_pkt_t&                             aql,
                kernel_dispatch::profiling_time) { this->post_kernel_call(aql, session); });
    });
}

void
DispatchThreadTracer::stop_context()  // NOLINT(readability-convert-member-functions-to-static)
{
    client.wlock([&](auto& client_id) {
        if(!client_id) return;

        // Remove our callbacks from HSA's queue controller
        hsa::get_queue_controller()->remove_callback(*client_id);
        client_id = std::nullopt;
    });

    auto* controller = hsa::get_queue_controller();
    if(controller) controller->disable_serialization();
}

void
AgentThreadTracer::resource_init(const CoreApiTable& coreapi, const AmdExtTable& ext)
{
    auto rocp_agents = rocprofiler::agent::get_agents();

    std::unique_lock<std::mutex> lk(agent_mut);

    for(const auto* rocp_agent : rocp_agents)
    {
        auto        id    = rocp_agent->id;
        const auto* cache = rocprofiler::agent::get_agent_cache(rocp_agent);

        if(tracers.find(id) != tracers.end())
            ROCP_WARNING << "Agent configured twice: " << id.handle;
        else if(params.find(id) == params.end())
            ROCP_INFO << "Skipping agent " << id.handle;
        else if(cache == nullptr)
            ROCP_WARNING << "Invalid agent id: " << id.handle;
        else
            tracers[id] = std::make_unique<ThreadTracerQueue>(params.at(id), *cache, coreapi, ext);
    }
}

void
AgentThreadTracer::resource_deinit()
{
    std::unique_lock<std::mutex> lk(agent_mut);
    tracers.clear();
}

void
AgentThreadTracer::start_context()
{
    std::unique_lock<std::mutex> lk(agent_mut);

    if(tracers.empty())
    {
        ROCP_FATAL << "Thread trace context not present for agent!";
        return;
    }

    std::vector<std::unique_ptr<Signal>> wait_list{};

    for(auto& [_, tracer] : tracers)
    {
        auto packet = tracer->get_control(true);
        packet->populate_before();

        auto sig = tracer->SubmitAndSignalLast(packet->before_krn_pkt);
        if(sig) wait_list.emplace_back(std::move(sig));
    }
}

void
AgentThreadTracer::stop_context()
{
    std::unique_lock<std::mutex> lk(agent_mut);

    if(tracers.empty())
    {
        ROCP_FATAL << "Thread trace context not present for agent!";
        return;
    }

    std::vector<std::tuple<ThreadTracerQueue*, aqlprofile_handle_t, std::unique_ptr<Signal>>>
        wait_list{};

    for(auto& [id, tracer] : tracers)
    {
        auto packet = tracer->get_control(false);
        packet->populate_after();

        auto signal = tracer->SubmitAndSignalLast(packet->after_krn_pkt);
        if(signal) wait_list.emplace_back(tracer.get(), packet->GetHandle(), std::move(signal));
    }

    for(auto& [tracer, handle, signal] : wait_list)
    {
        signal->WaitOn();
        rocprofiler_user_data_t userdata{.ptr = tracer->params.callback_userdata};
        tracer->iterate_data(handle, userdata);
    }
}

void
initialize(HsaApiTable* table)
{
    assert(table->core_ && table->amd_ext_);
    get_core() = *table->core_;
    get_ext()  = *table->amd_ext_;

    code_object::initialize(table);

    for(auto& ctx : context::get_registered_contexts())
    {
        if(ctx->agent_thread_trace)
            ctx->agent_thread_trace->resource_init(*table->core_, *table->amd_ext_);
    }
}

void
finalize()
{
    for(auto& ctx : context::get_registered_contexts())
    {
        if(ctx->agent_thread_trace) ctx->agent_thread_trace->resource_deinit();
    }
}

}  // namespace thread_trace

}  // namespace rocprofiler
