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

#pragma once

#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/thread_trace/code_object.hpp"
#include "lib/rocprofiler-sdk/thread_trace/hsa_util.hpp"

#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <atomic>
#include <cstdint>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
class AQLPacket;
};

namespace thread_trace
{
/// Collection of user-provided knobs that steer an ATT capture session.
/// The struct mirrors the public C API parameters and is validated before any
/// hardware programming happens.
struct thread_trace_parameter_pack
{
    rocprofiler_context_id_t                        context_id{0};
    rocprofiler_thread_trace_dispatch_callback_t    dispatch_cb_fn{nullptr};
    rocprofiler_thread_trace_shader_data_callback_t shader_cb_fn{nullptr};
    rocprofiler_user_data_t                         callback_userdata{};

    // Parameters
    uint8_t  target_cu          = 1;
    uint8_t  simd_select        = DEFAULT_SIMD;
    uint8_t  perfcounter_ctrl   = 0;
    uint64_t shader_engine_mask = DEFAULT_SE_MASK;
    uint64_t buffer_size        = DEFAULT_BUFFER_SIZE;
    uint64_t perf_exclude_mask  = 0;
    bool     no_detail_simd     = false;
    bool     triple_buffering   = false;

    bool bSerialize = false;

    // GFX9 Only
    std::vector<std::pair<uint32_t, uint32_t>> perfcounters;

    static constexpr size_t DEFAULT_SIMD        = 0xF;
    static constexpr size_t DEFAULT_SE_MASK     = 0x1;
    static constexpr size_t DEFAULT_BUFFER_SIZE = 0x8000000;

    /// Ensures the provided parameters match the hardware contract and public API.
    bool are_params_valid() const;
};

class ThreadTracerAgent
{
    using code_object_id_t = uint64_t;

public:
    /// Owns the lifetime of an ATT queue for a single GPU agent.
    ThreadTracerAgent(thread_trace_parameter_pack _params, rocprofiler_agent_id_t);
    virtual ~ThreadTracerAgent();

    /// Register the provided code object so trace markers have symbol context.
    void load_codeobj(code_object_id_t id, uint64_t addr, uint64_t size);
    /// Remove a previously registered code object.
    void unload_codeobj(code_object_id_t id);

    /// Acquire a copy of the control packet bundle for a dispatch boundary.
    std::unique_ptr<hsa::TraceControlAQLPacket> get_start_packet();
    /// Relay data buffers produced by the ATT to the user-mode callback.
    void iterate_data(aqlprofile_handle_t handle, rocprofiler_user_data_t data);
    /// Same as above but for device mode, using the internal data structures
    void iterate_data();

    thread_trace_parameter_pack  params;
    const rocprofiler_agent_id_t agent_id;

    std::unique_ptr<aql::ThreadTraceAQLPacketFactory> factory{nullptr};

    /// Start the trace and spawn helper threads when triple buffering is used.
    std::shared_ptr<hsa_signal_t> start_thread_trace(
        std::shared_ptr<std::atomic<int>> running_flag);
    /// Stop the trace and flush the outstanding hardware packets.
    signal_ptr_t stop_thread_trace();

private:
    /// Acquire a copy of the control packet, with optional increment to active_traces
    std::unique_ptr<hsa::TraceControlAQLPacket> get_control(bool bStart = false);

    att_queue_ptr_t queue{};

    std::atomic<int> active_traces{0};
    std::mutex       trace_resources_mut{};

    std::unique_ptr<hsa::TraceControlAQLPacket>           control_packet{nullptr};
    std::unique_ptr<code_object::CodeobjCallbackRegistry> codeobj_reg{nullptr};

    std::thread                       consumer{};
    std::thread                       producer{};
    std::shared_ptr<std::atomic<int>> worker_flag{nullptr};
};

class DispatchThreadTracer
{
    using code_object_id_t = uint64_t;
    using AQLPacketPtr     = std::unique_ptr<hsa::AQLPacket>;
    using inst_pkt_t       = common::container::small_vector<std::pair<AQLPacketPtr, int64_t>, 4>;

public:
    DispatchThreadTracer()  = default;
    ~DispatchThreadTracer() = default;

    /// Initializes shared resources needed by dispatch-based tracing.
    void start_context();
    void stop_context();
    void resource_init();
    void resource_deinit();

    void add_agent(rocprofiler_agent_id_t agent, thread_trace_parameter_pack pack)
    {
        auto lk       = std::unique_lock{agents_map_mut};
        params[agent] = std::move(pack);
    }

    hsa::write_packet_t pre_kernel_call(const hsa::Queue&              queue,
                                        uint64_t                       kernel_id,
                                        rocprofiler_dispatch_id_t      dispatch_id,
                                        rocprofiler_user_data_t*       user_data,
                                        const context::correlation_id* corr_id);

    void        post_kernel_call(inst_pkt_t&                      aql,
                                 const hsa::queue_info_session_t& session,
                                 const hsa::packet_data_t&        packet_data);
    const auto& get_agents() const { return agents; }

private:
    std::unordered_map<hsa_agent_t, std::unique_ptr<ThreadTracerAgent>>     agents{};
    std::unordered_map<rocprofiler_agent_id_t, thread_trace_parameter_pack> params{};

    std::shared_mutex agents_map_mut{};
    std::atomic<int>  post_move_data{0};
};

class DeviceThreadTracer
{
public:
    DeviceThreadTracer();
    ~DeviceThreadTracer() = default;

    /// Initializes shared resources needed by device-wide tracing.
    void start_context();
    void stop_context();
    void resource_init();
    void resource_deinit();

    void add_agent(rocprofiler_agent_id_t id, thread_trace_parameter_pack _params)
    {
        std::unique_lock<std::mutex> lk(agent_mut);
        params[id] = std::move(_params);
    }
    bool has_agent(rocprofiler_agent_id_t id)
    {
        std::unique_lock<std::mutex> lk(agent_mut);
        return params.find(id) != params.end();
    }

    const auto& get_agents() const { return agents; }

    friend void flush_and_stop();

private:
    std::map<rocprofiler_agent_id_t, std::unique_ptr<ThreadTracerAgent>> agents{};
    std::map<rocprofiler_agent_id_t, thread_trace_parameter_pack>        params{};

    std::mutex                        agent_mut;
    std::shared_ptr<std::atomic<int>> worker_flag{nullptr};
};

/// Install the thread trace service for newly created contexts.
void
initialize(HsaApiTable* table);

/// Tear down shared resources when the runtime shuts down.
void
finalize();

/// Stop and join all active producer/consumer threads, flushing any pending
/// data.  Safe to call before hsa_shut_down; prevents new traces from starting.
void
flush_and_stop();

}  // namespace thread_trace
}  // namespace rocprofiler
