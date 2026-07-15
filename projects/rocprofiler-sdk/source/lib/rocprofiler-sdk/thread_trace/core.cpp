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

// Implements the core coordination logic for thread trace start/stop, buffer
// iteration, and integration with the public API surface.
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"
#include "lib/rocprofiler-sdk/thread_trace/threading.hpp"

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
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
#include <thread>
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
constexpr uint64_t MIN_BUFFER_SIZE = 1 << 20;  // 1MB minimum to give the GPU room before copies

struct cbdata_t
{
    rocprofiler_agent_id_t                          agent    = {.handle = 0};
    rocprofiler_thread_trace_shader_data_callback_t cb_fn    = nullptr;
    const rocprofiler_user_data_t*                  userdata = nullptr;
};

// Keeps track of a single client registering for serialized thread trace
// operations so we can gate new traces while one is active.
common::Synchronized<std::optional<int64_t>> client;

// True once the HSA runtime is registered. Gates start_context() so pre-init
// start requests are deferred and replayed by initialize().
std::atomic<bool>&
hsa_inited()
{
    static std::atomic<bool> inited{false};
    return inited;
}

hsa_status_t
thread_trace_callback(uint32_t shader, void* buffer, uint64_t size, void* callback_data)
{
    auto& cb_data = *static_cast<cbdata_t*>(callback_data);

    cb_data.cb_fn(cb_data.agent,
                  shader,
                  buffer,
                  size,
                  ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END,
                  *cb_data.userdata);
    // The iterator guarantees the last chunk is tagged with END; here we just
    // ferry the data to the user callback.
    return HSA_STATUS_SUCCESS;
}

bool
thread_trace_parameter_pack::are_params_valid() const
{
    // Guard against the most common misconfigurations before touching HSA
    // state so we can fail early with a descriptive message.
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

ThreadTracerAgent::ThreadTracerAgent(thread_trace_parameter_pack _params,
                                     rocprofiler_agent_id_t      cache)
: params(std::move(_params))
, agent_id(cache)
{
    // Allocate and configure all heavy-weight objects up front: subsequent
    // start calls reuse the queue and packet factory without additional setup.
    ROCP_TRACE << "Constructing ATT instance for agent " << agent_id.handle;
    auto* core = CHECK_NOTNULL(hsa::get_core_table());
    auto* ext  = CHECK_NOTNULL(hsa::get_amd_ext_table());

    const auto* agent =
        CHECK_NOTNULL(rocprofiler::agent::get_agent_cache(rocprofiler::agent::get_agent(agent_id)));

    size_t triple_buffer_size = params.triple_buffering ? params.buffer_size : 0ul;
    queue                     = make_att_queue(*agent, triple_buffer_size);

    factory = std::make_unique<aql::ThreadTraceAQLPacketFactory>(*agent, this->params, *core, *ext);
    control_packet = factory->construct_control_packet();

    codeobj_reg = std::make_unique<code_object::CodeobjCallbackRegistry>(
        [this](rocprofiler_agent_id_t _agent, uint64_t codeobj_id, uint64_t addr, uint64_t size) {
            if(_agent == this->agent_id) this->load_codeobj(codeobj_id, addr, size);
        },
        [this](uint64_t codeobj_id) { this->unload_codeobj(codeobj_id); });

    codeobj_reg->IterateLoaded();
}

ThreadTracerAgent::~ThreadTracerAgent()
{
    ROCP_TRACE << "Destroying ATT Queue...";
    if(active_traces.load() < 1) return;

    // This is handled in triple buffer case
    if(worker_flag && params.triple_buffering)
        ROCP_INFO << "Thread tracer being destroyed with thread trace active";
    else
        ROCP_WARNING << "Thread tracer being destroyed with thread trace active";

    if(auto flag = worker_flag) flag->store(WORKER_FLAG_DESTRUCTOR);
    stop_thread_trace();
}

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
std::unique_ptr<hsa::TraceControlAQLPacket>
ThreadTracerAgent::get_control(bool bStart)
{
    auto active_resources = std::make_unique<hsa::TraceControlAQLPacket>(*control_packet);
    // Clone the control packet so callers can safely mutate state without
    // racing with concurrent dispatches.
    active_resources->clear();

    if(bStart) active_traces.fetch_add(1);

    return active_resources;
}

std::unique_ptr<hsa::TraceControlAQLPacket>
ThreadTracerAgent::get_start_packet()
{
    auto lock = std::unique_lock{trace_resources_mut};
    return get_control(true);
}

void
ThreadTracerAgent::iterate_data(aqlprofile_handle_t handle, rocprofiler_user_data_t data)
{
    if(active_traces.load() <= 0) return;

    cbdata_t cb_dt{};

    cb_dt.agent = agent_id;
    // Walk each buffer produced by the ATT runtime and forward it to the
    // registered shader callback.
    cb_dt.cb_fn    = params.shader_cb_fn;
    cb_dt.userdata = &data;

    auto status = aqlprofile_att_iterate_data(handle, thread_trace_callback, &cb_dt);
    if(status == HSA_STATUS_ERROR_OUT_OF_RESOURCES)
        ROCP_WARNING << "Thread trace buffer full!";
    else if(status != HSA_STATUS_SUCCESS)
        ROCP_CI_LOG(ERROR) << "Failed to iterate ATT data: " << status;

    active_traces.fetch_sub(1);
}

void
ThreadTracerAgent::iterate_data()
{
    // Already executed by producer thread, skip
    if(params.triple_buffering) return;

    auto lock = std::unique_lock{trace_resources_mut};
    iterate_data(control_packet->GetHandle(), params.callback_userdata);
}

void
ThreadTracerAgent::load_codeobj(code_object_id_t id, uint64_t addr, uint64_t size)
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);

    control_packet->add_codeobj(id, addr, size);
    // Keep shader metadata in sync while traces are live so symbol resolution
    // remains accurate in the emitted stream.

    if(!queue || active_traces.load() < 1) return;

    auto packet = factory->construct_load_marker_packet(id, addr, size);
    auto sig    = att_queue_submit(*queue, &packet->packet, true);
    if(sig) signal_wait(*sig);
}

void
ThreadTracerAgent::unload_codeobj(code_object_id_t id)
{
    std::unique_lock<std::mutex> lk(trace_resources_mut);

    if(!control_packet->remove_codeobj(id)) return;
    // Tear down metadata when code objects disappear to avoid dangling
    // references in the trace stream.
    if(!queue || active_traces.load() < 1) return;

    auto packet = factory->construct_unload_marker_packet(id);
    auto sig    = att_queue_submit(*queue, &packet->packet, true);
    if(sig) signal_wait(*sig);
}

std::shared_ptr<hsa_signal_t>
ThreadTracerAgent::start_thread_trace(std::shared_ptr<std::atomic<int>> _flag)
{
    ROCP_TRACE << "Starting thread trace for agent " << agent_id.handle;
    auto lock   = std::unique_lock{trace_resources_mut};
    worker_flag = std::move(_flag);

    auto control_packet_copy = get_control(true);
    control_packet_copy->clear();
    control_packet_copy->populate_before();
    control_packet_copy->populate_after();

    // Warmup the async copy so we dont wait too long for the flip.
    if(params.triple_buffering)
    {
        auto& buffer = queue->triple_buffer_memory;
        copy_data_sync(buffer.at(0),
                       buffer.at(1),
                       queue->near_cpu,
                       queue->hsa_agent,
                       MIN_BUFFER_SIZE,
                       nullptr);
    }

    // Submit the start packets without waiting: the producer thread (triple-buffer
    // path) and DeviceThreadTracer::start_context (single-buffer path) wait on the
    // returned signal so multiple agents can be launched in parallel.
    auto unique_signal = att_queue_submit_signal_last(*queue, control_packet_copy->before_krn_pkt);
    auto shared_signal = std::shared_ptr<hsa_signal_t>(std::move(unique_signal));

    if(params.triple_buffering)
    {
        // Find unique shader engine ID from mask
        uint64_t shader_engine_id = 0;
        for(uint64_t i = 0; (params.shader_engine_mask >> i) != 0; i++)
            if((params.shader_engine_mask >> i) % 2 == 1) shader_engine_id = i;

        auto buffer_packet = std::make_unique<rocprofiler::hsa::SQTTBufferingPackets>(
            control_packet_copy->GetHandle(), shader_engine_id);
        // Emit the optional buffer header first so consumers can prime state
        // before the main payload arrives.
        if(buffer_packet->header != 0)
        {
            params.shader_cb_fn(agent_id,
                                0,
                                &buffer_packet->header,
                                sizeof(buffer_packet->header),
                                ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_NONE,
                                params.callback_userdata);
        }

        auto worker_data   = std::make_shared<triple_buffer_shared_data_t>();
        worker_data->queue = queue.get();  // non-owning; ThreadTracerAgent owns queue

        // Initialize buffer memory pointers from the queue's triple buffer
        for(size_t i = 0; i < worker_data->buffers.size(); i++)
            worker_data->buffers.at(i).memory = worker_data->queue->triple_buffer_memory.at(i);

        auto producer_data             = triple_buffer_producer_data_t{};
        producer_data.producer_running = worker_flag;
        producer_data.start_pkt_signal = shared_signal;
        producer_data.control_packet   = std::move(control_packet_copy);
        producer_data.copy_data_fn     = copy_data_sync;
        producer_data.shared           = worker_data;
        producer_data.buffer_packet    = std::move(buffer_packet);

        auto consumer_data        = triple_buffer_consumer_data_t{};
        consumer_data.callback_fn = params.shader_cb_fn;
        consumer_data.userdata    = params.callback_userdata;
        consumer_data.shared      = worker_data;

        // Other call sites (kfd, internal_threading) wrap each std::thread
        // creation in its own pre/post pair, so match that convention.
        internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
        producer = std::thread{producer_loop, std::move(producer_data)};
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);

        internal_threading::notify_pre_internal_thread_create(ROCPROFILER_LIBRARY);
        consumer = std::thread{consumer_loop, std::move(consumer_data)};
        internal_threading::notify_post_internal_thread_create(ROCPROFILER_LIBRARY);
    }
    return shared_signal;
}

signal_ptr_t
ThreadTracerAgent::stop_thread_trace()
{
    ROCP_TRACE << "Stopping Thread trace for agent " << agent_id.handle;
    auto lock = std::unique_lock{trace_resources_mut};

    if(active_traces.load() == 0) return nullptr;

    if(params.triple_buffering)
    {
        int expected = WORKER_FLAG_RUNNING;
        worker_flag->compare_exchange_strong(expected, WORKER_FLAG_STOP);

        if(producer.joinable()) producer.join();
        if(consumer.joinable()) consumer.join();
        active_traces.fetch_sub(1);
        worker_flag = nullptr;
        return nullptr;
    }
    else
    {
        auto control_packet_copy = get_control(false);
        control_packet_copy->clear();
        // Join helpers and emit the final set of packets so the GPU drains.
        control_packet_copy->populate_after();
        // Submit without waiting; DeviceThreadTracer::stop_context fans out
        // submissions across agents and waits on every signal in parallel
        // before calling iterate_data.
        return att_queue_submit_signal_last(*queue, control_packet_copy->after_krn_pkt);
    }
}

// Single buffering: inject the stop packets directly and return.
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
            ROCP_CI_LOG_IF(TRACE, rocp_agent->runtime_visibility.hsa != 0)
                << fmt::format("Could not find HSA Agent for agent-{} (handle={}, name={})",
                               rocp_agent->node_id,
                               rocp_agent->id.handle,
                               rocp_agent->name);
            continue;
        }
        agents[*cache] = std::make_unique<ThreadTracerAgent>(it->second, rocp_agent->id);
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
hsa::write_packet_t
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

    auto packet = agent.get_start_packet();
    post_move_data.fetch_add(1);
    packet->populate_before();
    packet->populate_after();
    return {std::move(packet), true};
}

void
DispatchThreadTracer::post_kernel_call(DispatchThreadTracer::inst_pkt_t& aql,
                                       const hsa::queue_info_session_t& /*session*/,
                                       const hsa::packet_data_t& packet_data)
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
            it->second->iterate_data(pkt->GetHandle(), packet_data.user_data);
    }
}

void
DispatchThreadTracer::start_context()
{
    using corr_id_map_t = hsa::queue_info_session_t::external_corr_id_map_t;

    // Only installs queue-controller callbacks (cached and applied to queues as
    // they are created), so this is safe to call before hsa_init.
    CHECK_NOTNULL(hsa::get_queue_controller())->enable_serialization();

    // Only one thread should be attempting to enable/disable this context
    client.wlock([&](auto& client_id) {
        if(client_id) return;

        auto&& _callbacks = hsa::queue_callbacks_t{
            .batch_packets = []() { return false; },
            .write_interceptor =
                [=](const hsa::Queue& q,
                    const hsa::rocprofiler_packet& /* kern_pkt */,
                    rocprofiler_kernel_id_t   kernel_id,
                    rocprofiler_dispatch_id_t dispatch_id,
                    rocprofiler_user_data_t*  user_data,
                    const corr_id_map_t& /* extern_corr_ids */,
                    const context::correlation_id* corr_id) {
                    return this->pre_kernel_call(q, kernel_id, dispatch_id, user_data, corr_id);
                },
            .signal_completion =
                [=](const hsa::Queue& /* q */,
                    hsa::rocprofiler_packet /* kern_pkt */,
                    std::shared_ptr<hsa::queue_info_session_t>& session,
                    hsa::packet_data_t&                         packet_data,
                    inst_pkt_t&                                 aql,
                    kernel_dispatch::profiling_time) {
                    this->post_kernel_call(aql, *session, packet_data);
                }};

        client_id = CHECK_NOTNULL(hsa::get_queue_controller())
                        ->add_callback(std::nullopt, std::move(_callbacks));
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

DeviceThreadTracer::DeviceThreadTracer()
{
    worker_flag = std::make_shared<std::atomic<int>>(WORKER_FLAG_STOP);
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

        if(!rocprofiler::agent::get_hsa_agent(rocp_agent).has_value())
        {
            ROCP_TRACE << "Could not find HSA Agent for " << rocp_agent->id.handle
                       << ". This agent maybe isolated by ROCR_VISIBLE_DEVICES env variable";
            continue;
        }

        agents[it->first] = std::make_unique<ThreadTracerAgent>(it->second, rocp_agent->id);
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
    // Per-agent resources don't exist until HSA is registered; the request is
    // cached in the active-context array and replayed by initialize().
    if(!hsa_inited().load())
    {
        ROCP_INFO << "Device thread trace start requested before hsa_init; deferring";
        return;
    }

    ROCP_INFO << "Start device thread trace context";
    std::unique_lock<std::mutex> lk(agent_mut);

    if(agents.empty())
    {
        ROCP_WARNING << "Thread trace context not present for agent!";
        return;
    }

    int expected = WORKER_FLAG_STOP;
    CHECK_NOTNULL(worker_flag)->compare_exchange_strong(expected, WORKER_FLAG_RUNNING);
    auto wait_list = std::vector<std::shared_ptr<hsa_signal_t>>{};

    for(auto& [_, tracer] : agents)
        wait_list.emplace_back(tracer->start_thread_trace(worker_flag));

    for(auto& sig : wait_list)
        signal_wait(*CHECK_NOTNULL(sig));
}

void
DeviceThreadTracer::stop_context()
{
    auto lock = std::unique_lock{agent_mut};

    if(agents.empty()) return;

    ROCP_INFO << "Stopping device thread trace context";

    int expected = WORKER_FLAG_RUNNING;
    if(auto flag = worker_flag) flag->compare_exchange_strong(expected, WORKER_FLAG_STOP);

    auto wait_list = std::vector<signal_ptr_t>{};

    for(auto& [_, tracer] : agents)
        wait_list.emplace_back(tracer->stop_thread_trace());

    // Wait on every agent's after-packets explicitly so iterate_data only runs
    // once the GPU has drained the trace; mirrors start_context's parallel wait.
    for(auto& sig : wait_list)
        if(sig) signal_wait(*sig);

    for(auto& [_, tracer] : agents)
        tracer->iterate_data();
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

    // HSA resources now exist; allow start_context() to program the hardware.
    hsa_inited().store(true);

    // Replay device contexts started before hsa_init() (their start_context()
    // returned early above). Dispatch mode needs no replay.
    for(auto& ctx : context::get_active_contexts())
    {
        if(ctx->device_thread_trace) ctx->device_thread_trace->start_context();
    }
}

void
flush_and_stop()
{
    ROCP_TRACE << "flush_and_stop called";
    for(auto& ctx : context::get_registered_contexts())
    {
        if(ctx->device_thread_trace)
        {
            CHECK_NOTNULL(ctx->device_thread_trace->worker_flag)->store(WORKER_FLAG_DESTRUCTOR);
            ctx->device_thread_trace->stop_context();
        }
        if(ctx->dispatch_thread_trace) ctx->dispatch_thread_trace->stop_context();
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
