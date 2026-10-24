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
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
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
struct hipStreamBeginCapture;
struct hipStreamBeginCapture_spt;
struct hipStreamBeginCaptureToGraph;
struct hipStreamEndCapture;
struct hipStreamEndCapture_spt;
struct hipGraphDestroy;
}  // namespace api

struct capture_plan
{
    struct kernel_node
    {
        std::string name           = {};  // device function name, as hipKernelNameRefByPtr gives it
        uint64_t    correlation_id = 0;   // the HIP call whose capture created the node
        mutable std::atomic<uint64_t> kernel_id = {0};  // the kernel id that last matched the name
    };

    std::deque<kernel_node> kernels = {};  // the graph's captured kernel nodes, in capture order
};

namespace
{
// Process-global hipGraphExec_t -> stable monotonic id map.
std::shared_mutex                              g_map_mutex;
std::unordered_map<::hipGraphExec_t, uint64_t> g_exec_to_id;
std::atomic<uint64_t> g_next_graph_exec_id{1};  // 0 reserved = "not from a graph"
// hipGraphExec_t -> its captured kernel nodes, for an exec instantiated from a captured graph.
std::unordered_map<::hipGraphExec_t, std::shared_ptr<const capture_plan>> g_exec_capture;

// HIP runtime functions the capture tracking calls itself, taken from the dispatch table before
// the API tracing wrappers are installed: its queries are neither traced nor seen by its own hook.
struct capture_api_t
{
    decltype(::HipDispatchTable::hipStreamGetCaptureInfo_v2_fn)  get_capture_info   = nullptr;
    decltype(::HipDispatchTable::hipGraphGetNodes_fn)            get_nodes          = nullptr;
    decltype(::HipDispatchTable::hipGraphNodeGetType_fn)         node_get_type      = nullptr;
    decltype(::HipDispatchTable::hipGraphKernelNodeGetParams_fn) kernel_node_params = nullptr;
    decltype(::HipDispatchTable::hipKernelNameRefByPtr_fn)       kernel_name        = nullptr;
};

capture_api_t g_capture_api = {};

struct captured_node
{
    ::hipGraphNode_t node           = nullptr;
    uint64_t         correlation_id = 0;
};

// A stream capture begun on this thread.
struct capture_session
{
    ::hipStream_t              stream = nullptr;  // the stream that began it
    ::hipGraph_t               graph  = nullptr;  // the graph it builds
    size_t                     known  = 0;        // the graph's nodes already seen
    std::vector<captured_node> nodes  = {};       // nodes attributed to a traced call, in order
};

// The captures begun on this thread. A traced HIP call on this thread attributes the nodes its
// captures gained during the call, so work launched into them from another thread is not attributed
// correctly.
thread_local std::vector<capture_session> t_captures;

// hipGraph_t -> its attributed nodes, from the end of its capture until it is destroyed.
std::mutex                                                   g_capture_mutex;
std::unordered_map<::hipGraph_t, std::vector<captured_node>> g_captured_graphs;

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
    g_exec_capture.erase(exec);
}

std::shared_ptr<const capture_plan>
lookup_capture_plan(::hipGraphExec_t exec)
{
    std::shared_lock lock{g_map_mutex};
    auto             it = g_exec_capture.find(exec);
    return it == g_exec_capture.end() ? nullptr : it->second;
}

// Whether a capture is still active, updating its graph; false once it has ended on another thread
// or been invalidated.
bool
refresh_capture(capture_session& session)
{
    if(!g_capture_api.get_capture_info) return false;

    auto                  status = hipStreamCaptureStatusNone;
    unsigned long long    id     = 0;
    ::hipGraph_t          graph  = nullptr;
    const hipGraphNode_t* deps   = nullptr;
    size_t                ndeps  = 0;
    if(g_capture_api.get_capture_info(session.stream, &status, &id, &graph, &deps, &ndeps) !=
           hipSuccess ||
       status != hipStreamCaptureStatusActive)
        return false;
    if(graph != nullptr) session.graph = graph;
    return true;
}

size_t
graph_node_count(::hipGraph_t graph)
{
    size_t count = 0;
    if(graph == nullptr || !g_capture_api.get_nodes ||
       g_capture_api.get_nodes(graph, nullptr, &count) != hipSuccess)
        return 0;
    return count;
}

// Record that the nodes a capture's graph gained since the last look were added by correlation_id.
void
attribute_new_nodes(capture_session& session, uint64_t correlation_id)
{
    auto count = graph_node_count(session.graph);
    if(count <= session.known) return;

    // hipGraphGetNodes lists a graph's nodes in the order they were added
    auto nodes = std::vector<::hipGraphNode_t>(count);
    if(g_capture_api.get_nodes(session.graph, nodes.data(), &count) != hipSuccess) return;
    for(size_t i = session.known; i < count && i < nodes.size(); ++i)
        session.nodes.emplace_back(captured_node{nodes.at(i), correlation_id});
    session.known = count;
}

void
begin_capture(::hipStream_t stream)
{
    auto session = capture_session{.stream = stream};
    if(!refresh_capture(session)) return;
    // nodes already in the graph (hipStreamBeginCaptureToGraph) were not captured
    session.known = graph_node_count(session.graph);
    t_captures.emplace_back(std::move(session));
}

void
end_capture(::hipStream_t stream, ::hipGraph_t graph)
{
    auto itr = std::find_if(t_captures.begin(), t_captures.end(), [stream](const auto& session) {
        return session.stream == stream;
    });
    if(itr == t_captures.end()) return;

    if(graph != nullptr && !itr->nodes.empty())
    {
        std::unique_lock lock{g_capture_mutex};
        g_captured_graphs[graph] = std::move(itr->nodes);
    }
    t_captures.erase(itr);
}

void
forget_captured_graph(::hipGraph_t graph)
{
    std::unique_lock lock{g_capture_mutex};
    g_captured_graphs.erase(graph);
}

// At instantiation: the captured kernel nodes of the source graph, in capture order, with the
// names a replayed dispatch's kernel must match.
void
plan_capture(::hipGraphExec_t exec, ::hipGraph_t graph)
{
    auto nodes = std::vector<captured_node>{};
    {
        std::unique_lock lock{g_capture_mutex};
        auto             itr = g_captured_graphs.find(graph);
        if(itr == g_captured_graphs.end()) return;
        nodes = itr->second;
    }
    if(!g_capture_api.node_get_type || !g_capture_api.kernel_node_params ||
       !g_capture_api.kernel_name)
        return;

    auto plan = std::make_shared<capture_plan>();
    for(const auto& itr : nodes)
    {
        auto type = hipGraphNodeTypeEmpty;
        if(g_capture_api.node_get_type(itr.node, &type) != hipSuccess ||
           type != hipGraphNodeTypeKernel)
            continue;

        auto params = hipKernelNodeParams{};
        if(g_capture_api.kernel_node_params(itr.node, &params) != hipSuccess) continue;

        // a node without a name could never match: leaving it out keeps the rest matchable
        const auto* name = g_capture_api.kernel_name(params.func, nullptr);
        if(name == nullptr || *name == '\0') continue;

        auto& kernel          = plan->kernels.emplace_back();
        kernel.name           = name;
        kernel.correlation_id = itr.correlation_id;
    }
    if(plan->kernels.empty()) return;

    std::unique_lock lock{g_map_mutex};
    g_exec_capture[exec] = std::move(plan);
}

// The capture correlation id of a launch's next captured kernel node, if the dispatched kernel is
// that node's (the kernel id that matched before, or else the same name); the launch then moves to
// the following node. Otherwise 0 and no move: a kernel the capture did not record (a runtime fill
// or copy kernel) takes no node, and a changed order leaves the rest of the launch unattributed
// rather than mislabeled.
uint64_t
match_captured_kernel(launch_state& state, uint64_t kernel_id)
{
    if(!state.capture || state.capture_cursor >= state.capture->kernels.size()) return 0;

    const auto& node = state.capture->kernels.at(state.capture_cursor);
    if(node.kernel_id.load(std::memory_order_relaxed) != kernel_id)
    {
        const auto* symbol = code_object::get_kernel_name(kernel_id);
        if(symbol == nullptr) return 0;

        // the code object's symbol name may end in the kernel descriptor's ".kd"
        auto name = std::string_view{*symbol};
        if(constexpr auto kd = std::string_view{".kd"};
           name.size() > kd.size() && name.substr(name.size() - kd.size()) == kd)
            name.remove_suffix(kd.size());
        if(name != node.name) return 0;

        node.kernel_id.store(kernel_id, std::memory_order_relaxed);
    }

    ++state.capture_cursor;
    return node.correlation_id;
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

// The source graph: the first argument after the exec in every hipGraphInstantiate* signature.
template <typename... Args>
::hipGraph_t
source_graph(::hipGraph_t graph, const Args&...)
{
    return graph;
}

// One static next_func slot per template instantiation (the 3 hipGraphInstantiate* signatures
// differ).
template <typename ApiId, typename RetT, typename... Args>
auto graph_instantiate(RetT (*next)(::hipGraphExec_t*, Args...))
{
    static auto next_func = next;
    return +[](::hipGraphExec_t* out, Args... args) -> RetT {
        auto graph = source_graph(args...);
        auto ret   = next_func(out, std::forward<Args>(args)...);
        if(ret == hipSuccess && out != nullptr && *out != nullptr)
        {
            auto exec_id = assign_graph_exec_id(*out);
            plan_capture(*out, graph);
            fire_hip_graph_none_callback(
                ROCPROFILER_HIP_GRAPH_OPERATION_EXEC_CREATE, exec_id, *out);
        }
        return ret;
    };
}

// hipStreamBeginCapture{,_spt} and hipStreamBeginCaptureToGraph: the stream's capture starts on
// this thread.
template <typename ApiId, typename RetT, typename... Args>
auto stream_begin_capture(RetT (*next)(::hipStream_t, Args...))
{
    static auto next_func = next;
    return +[](::hipStream_t stream, Args... args) -> RetT {
        auto ret = next_func(stream, std::forward<Args>(args)...);
        if(ret == hipSuccess) begin_capture(stream);
        return ret;
    };
}

// hipStreamEndCapture{,_spt}: the capture's attributed nodes move to its graph.
template <typename ApiId, typename RetT>
auto stream_end_capture(RetT (*next)(::hipStream_t, ::hipGraph_t*))
{
    static auto next_func = next;
    return +[](::hipStream_t stream, ::hipGraph_t* graph_out) -> RetT {
        auto ret = next_func(stream, graph_out);
        end_capture(stream, (ret == hipSuccess && graph_out != nullptr) ? *graph_out : nullptr);
        return ret;
    };
}

// hipGraphDestroy: a destroyed graph's handle can be reused.
template <typename ApiId, typename RetT>
auto graph_free(RetT (*next)(::hipGraph_t))
{
    static auto next_func = next;
    return +[](::hipGraph_t graph) -> RetT {
        auto ret = next_func(graph);
        if(ret == hipSuccess && graph != nullptr) forget_captured_graph(graph);
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
        s.capture   = lookup_capture_plan(exec);
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

enqueue_identity
next_enqueue_identity(uint64_t kernel_id)
{
    auto* state = current_launch_state();
    if(state == nullptr) return enqueue_identity{};

    return enqueue_identity{
        .graph_exec_id          = {.handle = state->graph_exec_id},
        .graph_node_id          = {.handle = state->node_counter++},
        .capture_correlation_id = (kernel_id != 0) ? match_captured_kernel(*state, kernel_id) : 0};
}

bool
capture_active()
{
    return !t_captures.empty();
}

void
on_capture_api_exit(uint64_t internal_correlation_id)
{
    for(auto itr = t_captures.begin(); itr != t_captures.end();)
    {
        if(!refresh_capture(*itr))
        {
            // ended without hipStreamEndCapture on this thread, or invalidated
            itr = t_captures.erase(itr);
            continue;
        }
        attribute_new_nodes(*itr, internal_correlation_id);
        ++itr;
    }
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

// Wrap the graph lifecycle and stream capture entry points on the HIP runtime dispatch table.
template <>
void
update_table(::HipDispatchTable* table)
{
    if(table == nullptr) return;

    // the implementations, before the API tracing wrappers are installed over them
    g_capture_api = capture_api_t{.get_capture_info   = table->hipStreamGetCaptureInfo_v2_fn,
                                  .get_nodes          = table->hipGraphGetNodes_fn,
                                  .node_get_type      = table->hipGraphNodeGetType_fn,
                                  .kernel_node_params = table->hipGraphKernelNodeGetParams_fn,
                                  .kernel_name        = table->hipKernelNameRefByPtr_fn};

    if(table->hipStreamBeginCapture_fn)
        table->hipStreamBeginCapture_fn =
            stream_begin_capture<api::hipStreamBeginCapture>(table->hipStreamBeginCapture_fn);
    if(table->hipStreamBeginCapture_spt_fn)
        table->hipStreamBeginCapture_spt_fn = stream_begin_capture<api::hipStreamBeginCapture_spt>(
            table->hipStreamBeginCapture_spt_fn);
    if(table->hipStreamBeginCaptureToGraph_fn)
        table->hipStreamBeginCaptureToGraph_fn =
            stream_begin_capture<api::hipStreamBeginCaptureToGraph>(
                table->hipStreamBeginCaptureToGraph_fn);
    if(table->hipStreamEndCapture_fn)
        table->hipStreamEndCapture_fn =
            stream_end_capture<api::hipStreamEndCapture>(table->hipStreamEndCapture_fn);
    if(table->hipStreamEndCapture_spt_fn)
        table->hipStreamEndCapture_spt_fn =
            stream_end_capture<api::hipStreamEndCapture_spt>(table->hipStreamEndCapture_spt_fn);
    if(table->hipGraphDestroy_fn)
        table->hipGraphDestroy_fn = graph_free<api::hipGraphDestroy>(table->hipGraphDestroy_fn);
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
