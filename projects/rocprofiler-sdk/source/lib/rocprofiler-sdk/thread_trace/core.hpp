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

#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <atomic>
#include <cstdint>
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

    bool bSerialize = false;

    // GFX9 Only
    std::vector<std::pair<uint32_t, uint32_t>> perfcounters;

    static constexpr size_t DEFAULT_SIMD        = 0xF;
    static constexpr size_t DEFAULT_SE_MASK     = 0x1;
    static constexpr size_t DEFAULT_BUFFER_SIZE = 0x8000000;

    bool are_params_valid() const;
};

class ThreadTracerQueue
{
    using code_object_id_t = uint64_t;

public:
    ThreadTracerQueue(thread_trace_parameter_pack _params, rocprofiler_agent_id_t);
    virtual ~ThreadTracerQueue();

    void load_codeobj(code_object_id_t id, uint64_t addr, uint64_t size);
    void unload_codeobj(code_object_id_t id);

    std::unique_ptr<hsa::TraceControlAQLPacket> get_control(bool bStart);
    void iterate_data(aqlprofile_handle_t handle, rocprofiler_user_data_t data);

    thread_trace_parameter_pack  params;
    const rocprofiler_agent_id_t agent_id;

    [[nodiscard]] std::unique_ptr<class Signal> Submit(hsa_ext_amd_aql_pm4_packet_t* packet,
                                                       bool                          bWait) const;

    template <typename VecType>
    [[nodiscard]] std::unique_ptr<class Signal> SubmitAndSignalLast(VecType vec)
    {
        for(size_t i = 0; i < vec.size(); i++)
        {
            auto sig = Submit(&vec.at(i), i == vec.size() - 1);
            if(sig) return sig;
        }
        return nullptr;
    }
    std::unique_ptr<aql::ThreadTraceAQLPacketFactory> factory{nullptr};

private:
    hsa_queue_t*     queue{nullptr};
    std::atomic<int> active_traces{0};
    std::mutex       trace_resources_mut{};

    std::unique_ptr<hsa::TraceControlAQLPacket>           control_packet{nullptr};
    std::unique_ptr<code_object::CodeobjCallbackRegistry> codeobj_reg{nullptr};
};

class DispatchThreadTracer
{
    using code_object_id_t = uint64_t;
    using AQLPacketPtr     = std::unique_ptr<hsa::AQLPacket>;
    using inst_pkt_t       = common::container::small_vector<std::pair<AQLPacketPtr, int64_t>, 4>;

public:
    DispatchThreadTracer()  = default;
    ~DispatchThreadTracer() = default;

    void start_context();
    void stop_context();
    void resource_init();
    void resource_deinit();

    void add_agent(rocprofiler_agent_id_t agent, thread_trace_parameter_pack pack)
    {
        auto lk       = std::unique_lock{agents_map_mut};
        params[agent] = std::move(pack);
    }

    hsa::Queue::pkt_and_serialize_t pre_kernel_call(const hsa::Queue&              queue,
                                                    uint64_t                       kernel_id,
                                                    rocprofiler_dispatch_id_t      dispatch_id,
                                                    rocprofiler_user_data_t*       user_data,
                                                    const context::correlation_id* corr_id);

    void        post_kernel_call(inst_pkt_t& aql, const hsa::queue_info_session& session);
    const auto& get_agents() const { return agents; }

private:
    std::unordered_map<hsa_agent_t, std::unique_ptr<ThreadTracerQueue>>     agents{};
    std::unordered_map<rocprofiler_agent_id_t, thread_trace_parameter_pack> params{};

    std::shared_mutex agents_map_mut{};
    std::atomic<int>  post_move_data{0};
};

class DeviceThreadTracer
{
public:
    DeviceThreadTracer()  = default;
    ~DeviceThreadTracer() = default;

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

private:
    std::map<rocprofiler_agent_id_t, std::unique_ptr<ThreadTracerQueue>> agents{};
    std::map<rocprofiler_agent_id_t, thread_trace_parameter_pack>        params{};

    std::mutex agent_mut;
};

void
initialize(HsaApiTable* table);

void
finalize();

}  // namespace thread_trace
}  // namespace rocprofiler
