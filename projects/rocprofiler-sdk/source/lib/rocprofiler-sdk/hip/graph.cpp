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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hip/graph.hpp"

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/correlation_id.hpp"
#include "lib/rocprofiler-sdk/hip/hip.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/external_correlation.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>  // pulls in <hip/amd_detail/hip_api_trace.hpp>

#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace hip
{
namespace graph
{
// the structs below are used to improve backtrace readability for the template functions
// graph_{instantiate,destroy,launch}. They are declared this outside of the anonymous namespace for
// readability purposes too.
namespace api
{
struct hipGraphInstantiate;
struct hipGraphInstantiateWithFlags;
struct hipGraphInstantiateWithParams;
struct hipGraphExecDestroy;
struct hipGraphLaunch;
struct hipGraphLaunch_spt;
}  // namespace api

namespace
{
// Process-global hipGraphExec_t -> stable monotonic id map.
std::shared_mutex                              g_map_mutex;
std::unordered_map<::hipGraphExec_t, uint64_t> g_exec_to_id;
std::atomic<uint64_t> g_next_graph_exec_id{1};  // 0 reserved = "not from a graph"

// Get-or-create: returns the existing id on race, otherwise assigns a fresh one.
uint64_t
assign_graph_exec_id(::hipGraphExec_t exec, bool* out_was_new = nullptr)
{
    if(out_was_new) *out_was_new = false;
    if(exec == nullptr) return 0;
    std::unique_lock lock{g_map_mutex};
    if(auto it = g_exec_to_id.find(exec); it != g_exec_to_id.end())
    {
        return it->second;
    }
    auto id            = g_next_graph_exec_id.fetch_add(1, std::memory_order_relaxed);
    g_exec_to_id[exec] = id;
    if(out_was_new) *out_was_new = true;
    return id;
}

void
forget_graph_exec(::hipGraphExec_t exec)
{
    if(exec == nullptr) return;
    std::unique_lock lock{g_map_mutex};
    g_exec_to_id.erase(exec);
}

// Build the callback payload struct used by all HIP_GRAPH callback fires.
rocprofiler_callback_tracing_hip_graph_data_t
make_hip_graph_payload(uint64_t graph_exec_id, ::hipGraphExec_t exec)
{
    return common::init_public_api_struct(
        rocprofiler_callback_tracing_hip_graph_data_t{},
        rocprofiler_graph_exec_id_t{graph_exec_id},
        rocprofiler_address_t{.ptr = static_cast<const void*>(exec)});
}

// Fire a HIP_GRAPH phase-NONE callback (used for EXEC_CREATE and EXEC_DESTROY).
void
fire_hip_graph_none_callback(rocprofiler_hip_graph_operation_t op,
                             uint64_t                          graph_exec_id,
                             ::hipGraphExec_t                  exec)
{
    auto callback_contexts = tracing::callback_context_data_vec_t{};
    auto external_corr_ids = tracing::external_correlation_id_map_t{};

    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
                               static_cast<rocprofiler_tracing_operation_t>(op),
                               callback_contexts,
                               external_corr_ids);

    if(callback_contexts.empty()) return;

    auto thr_id = common::get_tid();

    tracing::update_external_correlation_ids(
        external_corr_ids, thr_id, ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API);

    auto* corr_id          = context::get_latest_correlation_id();
    auto  internal_corr_id = (corr_id) ? corr_id->internal : uint64_t{0};
    auto  ancestor_corr_id = (corr_id) ? corr_id->ancestor : uint64_t{0};
    auto  tracer_data      = make_hip_graph_payload(graph_exec_id, exec);

    tracing::execute_phase_none_callbacks(callback_contexts,
                                          thr_id,
                                          internal_corr_id,
                                          external_corr_ids,
                                          ancestor_corr_id,
                                          ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
                                          static_cast<rocprofiler_tracing_operation_t>(op),
                                          tracer_data);
}

// One static next_func slot per template instantiation (the 3 hipGraphInstantiate* signatures
// differ).
template <typename ApiId, typename RetT, typename... Args>
auto graph_instantiate(RetT (*next)(::hipGraphExec_t*, Args...))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t* out, Args... args) -> RetT {
        auto ret = next_func(out, std::forward<Args>(args)...);
        if(ret == hipSuccess && out != nullptr && *out != nullptr)
        {
            auto exec_id = assign_graph_exec_id(*out);
            fire_hip_graph_none_callback(
                ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_CREATE, exec_id, *out);
        }
        return ret;
    };
}

template <typename ApiId, typename RetT>
auto graph_destroy(RetT (*next)(::hipGraphExec_t))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t exec) -> RetT {
        // Capture id before next_func; only forget + fire on successful destroy.
        auto exec_id = (exec != nullptr) ? lookup_graph_exec_id(exec) : uint64_t{0};

        auto ret = next_func(exec);

        if(ret == hipSuccess && exec != nullptr)
        {
            fire_hip_graph_none_callback(
                ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_DESTROY, exec_id, exec);
            forget_graph_exec(exec);
        }
        return ret;
    };
}

// std::deque preserves references when nested host-callback launches push onto the stack.
thread_local std::deque<launch_state> g_launch_stack;

// GPU agent for the launch stream's device, or {0} on failure.
rocprofiler_agent_id_t
resolve_launch_stream_agent(::hipStream_t stream)
{
    auto& saved_table = ::rocprofiler::hip::get_table();
    auto* runtime     = saved_table.runtime;
    if(runtime == nullptr) return rocprofiler_agent_id_t{.handle = 0};

    int  device_id = -1;
    auto is_default_stream =
        (stream == nullptr || stream == hipStreamLegacy || stream == hipStreamPerThread);

    if(!is_default_stream && runtime->hipStreamGetDevice_fn != nullptr)
    {
        if(runtime->hipStreamGetDevice_fn(stream, &device_id) != hipSuccess) device_id = -1;
    }

    if(device_id < 0 && runtime->hipGetDevice_fn != nullptr)
    {
        if(runtime->hipGetDevice_fn(&device_id) != hipSuccess) device_id = -1;
    }

    if(device_id < 0) return rocprofiler_agent_id_t{.handle = 0};

    for(const auto* a : ::rocprofiler::agent::get_agents())
    {
        if(a != nullptr && a->type == ROCPROFILER_AGENT_TYPE_GPU &&
           a->logical_node_type_id == device_id)
        {
            return a->id;
        }
    }
    return rocprofiler_agent_id_t{.handle = 0};
}

void
emit_graph_launch_record(const launch_state& s, rocprofiler_timestamp_t end_ts)
{
    auto tracing_data_v = tracing::tracing_data{};
    tracing::populate_contexts(ROCPROFILER_BUFFER_TRACING_HIP_GRAPH,
                               ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_LAUNCH,
                               tracing_data_v.buffered_contexts,
                               tracing_data_v.external_correlation_ids);

    if(tracing_data_v.buffered_contexts.empty()) return;

    auto record = rocprofiler_buffer_tracing_hip_graph_record_t{
        sizeof(rocprofiler_buffer_tracing_hip_graph_record_t),
        ROCPROFILER_BUFFER_TRACING_HIP_GRAPH,
        ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_LAUNCH,
        rocprofiler_correlation_id_t{},
        s.thread_id,
        s.start_ts,
        end_ts,
        s.agent_id,
        s.queue_id,
        rocprofiler_graph_exec_id_t{s.graph_exec_id},
        s.dispatch_count};

    tracing::execute_buffer_record_emplace(tracing_data_v.buffered_contexts,
                                           s.thread_id,
                                           s.correlation_id.internal,
                                           tracing_data_v.external_correlation_ids,
                                           s.correlation_id.ancestor,
                                           ROCPROFILER_BUFFER_TRACING_HIP_GRAPH,
                                           ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_LAUNCH,
                                           record);
}

// Tag gives hipGraphLaunch and hipGraphLaunch_spt distinct next_func slots.
enum class LaunchApiTag
{
    hipGraphLaunch,
    hipGraphLaunch_spt
};

template <typename ApiId, typename RetT>
auto graph_launch(RetT (*next)(::hipGraphExec_t, ::hipStream_t))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t exec, ::hipStream_t stream) -> RetT {
        // Shared callback_contexts spans ENTER and EXIT per execute_phase_exit_callbacks contract.
        auto callback_contexts = tracing::callback_context_data_vec_t{};
        auto external_corr_ids = tracing::external_correlation_id_map_t{};
        tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
                                   static_cast<rocprofiler_tracing_operation_t>(
                                       ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_LAUNCH),
                                   callback_contexts,
                                   external_corr_ids);

        g_launch_stack.emplace_back();
        auto& s = g_launch_stack.back();

        // Attach-mid-process fallback: synthesize CREATE so subscribers see lifecycle in order.
        bool fallback_assigned = false;
        s.graph_exec_id        = lookup_graph_exec_id(exec);
        if(s.graph_exec_id == 0)
        {
            s.graph_exec_id = assign_graph_exec_id(exec, &fallback_assigned);
            if(fallback_assigned)
            {
                fire_hip_graph_none_callback(
                    ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_CREATE, s.graph_exec_id, exec);
            }
        }
        s.thread_id = common::get_tid();
        if(auto* cid = ::rocprofiler::context::get_latest_correlation_id())
            s.correlation_id = {cid->internal, rocprofiler_user_data_t{}, cid->ancestor};
        else
            s.correlation_id = {0, rocprofiler_user_data_t{}, 0};
        s.start_ts = rocprofiler_timestamp_t{common::timestamp_ns()};
        // queue_id stays zero (launches may span multiple HW queues for parallel branches).
        s.agent_id = resolve_launch_stream_agent(stream);

        if(!callback_contexts.empty())
        {
            auto  tracer_data            = make_hip_graph_payload(s.graph_exec_id, exec);
            auto* enter_cid              = ::rocprofiler::context::get_latest_correlation_id();
            auto  enter_internal_corr_id = (enter_cid) ? enter_cid->internal : uint64_t{0};
            auto  enter_ancestor_corr_id = (enter_cid) ? enter_cid->ancestor : uint64_t{0};

            tracing::update_external_correlation_ids(
                external_corr_ids,
                s.thread_id,
                ROCPROFILER_EXTERNAL_CORRELATION_REQUEST_HIP_RUNTIME_API);

            tracing::execute_phase_enter_callbacks(callback_contexts,
                                                   s.thread_id,
                                                   enter_internal_corr_id,
                                                   external_corr_ids,
                                                   enter_ancestor_corr_id,
                                                   ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
                                                   static_cast<rocprofiler_tracing_operation_t>(
                                                       ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_LAUNCH),
                                                   tracer_data);
        }

        auto ret = next_func(exec, stream);

        // EXIT must fire before pop_back so current_launch_state() still sees the active launch.
        if(!callback_contexts.empty())
        {
            auto tracer_data = make_hip_graph_payload(s.graph_exec_id, exec);
            tracing::execute_phase_exit_callbacks(callback_contexts,
                                                  external_corr_ids,
                                                  ROCPROFILER_CALLBACK_TRACING_HIP_GRAPH,
                                                  static_cast<rocprofiler_tracing_operation_t>(
                                                      ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_LAUNCH),
                                                  tracer_data);
        }

        auto end_ts = rocprofiler_timestamp_t{common::timestamp_ns()};
        if(ret == hipSuccess)
        {
            emit_graph_launch_record(s, end_ts);
        }
        else if(fallback_assigned)
        {
            // Failed launch on fallback-assigned id: undo the synthesized CREATE.
            fire_hip_graph_none_callback(
                ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_DESTROY, s.graph_exec_id, exec);
            forget_graph_exec(exec);
        }
        g_launch_stack.pop_back();
        return ret;
    };
}

// Map rocprofiler_hip_graph_operation_t to respective name
template <size_t OpIdx>
struct hip_graph_operation_name;

#define HIP_GRAPH_OPERATION_NAME(ENUM)                                                             \
    template <>                                                                                    \
    struct hip_graph_operation_name<ROCPROFILER_HIP_GRAPH_OPERATION_##ENUM>                        \
    {                                                                                              \
        static constexpr auto name          = "HIP_GRAPH_OPERATION_" #ENUM;                        \
        static constexpr auto operation_idx = ROCPROFILER_HIP_GRAPH_OPERATION_##ENUM;              \
    };

HIP_GRAPH_OPERATION_NAME(NONE)
HIP_GRAPH_OPERATION_NAME(EXEC_CREATE)
HIP_GRAPH_OPERATION_NAME(EXEC_DESTROY)
HIP_GRAPH_OPERATION_NAME(EXEC_LAUNCH)
#undef HIP_GRAPH_OPERATION_NAME

template <size_t OpIdx, size_t... OpIdxTail>
const char*
name_by_id(const uint32_t id, std::index_sequence<OpIdx, OpIdxTail...>)
{
    if(OpIdx == id) return hip_graph_operation_name<OpIdx>::name;

    if constexpr(sizeof...(OpIdxTail) > 0)
        return name_by_id(id, std::index_sequence<OpIdxTail...>{});
    else
        return nullptr;
}

template <size_t OpIdx, size_t... OpIdxTail>
void
get_ids(std::vector<uint32_t>& _id_list, std::index_sequence<OpIdx, OpIdxTail...>)
{
    auto _idx = hip_graph_operation_name<OpIdx>::operation_idx;
    if(_idx < ROCPROFILER_HIP_GRAPH_OPERATION_LAST) _id_list.emplace_back(_idx);

    if constexpr(sizeof...(OpIdxTail) > 0) get_ids(_id_list, std::index_sequence<OpIdxTail...>{});
}
}  // namespace

launch_state*
current_launch_state()
{
    return g_launch_stack.empty() ? nullptr : &g_launch_stack.back();
}

uint64_t
lookup_graph_exec_id(::hipGraphExec_t exec)
{
    if(exec == nullptr) return 0;
    std::shared_lock lock{g_map_mutex};
    auto             it = g_exec_to_id.find(exec);
    return it == g_exec_to_id.end() ? 0 : it->second;
}

const char*
name_by_id(uint32_t id)
{
    return name_by_id(id, std::make_index_sequence<ROCPROFILER_HIP_GRAPH_OPERATION_LAST>{});
}

std::vector<uint32_t>
get_ids()
{
    constexpr auto last_id = ROCPROFILER_HIP_GRAPH_OPERATION_LAST;
    auto           _data   = std::vector<uint32_t>{};
    _data.reserve(last_id);
    get_ids(_data, std::make_index_sequence<ROCPROFILER_HIP_GRAPH_OPERATION_LAST>{});
    return _data;
}

// Wrap the four graph lifecycle entry points on the HIP runtime dispatch table.
template <>
void
update_table(::HipDispatchTable* table)
{
    if(table == nullptr) return;
    if(table->hipGraphInstantiate_fn)
        table->hipGraphInstantiate_fn =
            graph_instantiate<api::hipGraphInstantiate>(table->hipGraphInstantiate_fn);
    if(table->hipGraphInstantiateWithFlags_fn)
        table->hipGraphInstantiateWithFlags_fn =
            graph_instantiate<api::hipGraphInstantiateWithFlags>(
                table->hipGraphInstantiateWithFlags_fn);
    if(table->hipGraphInstantiateWithParams_fn)
        table->hipGraphInstantiateWithParams_fn =
            graph_instantiate<api::hipGraphInstantiateWithParams>(
                table->hipGraphInstantiateWithParams_fn);
    if(table->hipGraphExecDestroy_fn)
        table->hipGraphExecDestroy_fn =
            graph_destroy<api::hipGraphExecDestroy>(table->hipGraphExecDestroy_fn);
    if(table->hipGraphLaunch_fn)
        table->hipGraphLaunch_fn = graph_launch<api::hipGraphLaunch>(table->hipGraphLaunch_fn);
    if(table->hipGraphLaunch_spt_fn)
        table->hipGraphLaunch_spt_fn =
            graph_launch<api::hipGraphLaunch_spt>(table->hipGraphLaunch_spt_fn);
}

}  // namespace graph
}  // namespace hip
}  // namespace rocprofiler
