// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Triple-buffered SQTT capture that runs rocprof-trace-decoder's gfx9
// quick-scan inline on each shader-data chunk as it arrives. Demonstrates the
// streaming use case for the quick-scan path: per-event-type counts and
// PGM_LO/HI program-start histograms are aggregated across the run.
//
// First implementation note: we deliberately scan inline on the rocprofiler-sdk
// consumer (shader-data) callback thread rather than handing off to a worker.
// Two reasons:
//   1. The quick scanner runs comfortably faster than std::vector::assign could
//      *copy* the chunk, so deferring would actually be slower than just doing
//      the work here.
//   2. The SDK owns the chunk memory across its triple buffer; if we returned
//      from the callback while still holding the pointer, the next GPU drain
//      could overwrite it. Inline scanning keeps the lifetime obviously safe.
// A future revision that needs to overlap scan time with decode work can fan
// the chunks out to a worker (with a copy or a pinned buffer pool).

#ifdef NDEBUG
#    undef NDEBUG
#endif

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <rocprof_trace_decoder/rocprof_trace_decoder.h>
#include <rocprof_trace_decoder/trace_decoder_types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    if(auto ec = (result); ec != ROCPROFILER_STATUS_SUCCESS)                                       \
    {                                                                                              \
        std::cerr << "rocprofiler-sdk error at " << __FILE__ << ":" << __LINE__                    \
                  << " :: " << #result << " :: " << msg << " :: "                                  \
                  << rocprofiler_get_status_string(ec) << std::endl;                               \
        abort();                                                                                   \
    }

namespace ScanState
{
// Triple-buffer mode delivers the 8-byte gfx9 trace header at chunk_index 0
// before any payload chunk; payload chunks arrive at chunk_index 1, 2, ...
// The handle-based quick_scan API consumes the header on the chunk_index 0
// call and threads register-tracking state through the handle to subsequent
// chunks, so we no longer need to cache the header here.

// Hot counters use atomics; the data histograms are mutex-protected because
// even though rocprofiler-sdk's triple-buffer consumer thread is single-
// threaded today, that's an implementation detail.
std::atomic<uint64_t> chunks_processed{0};
std::atomic<uint64_t> bytes_processed{0};
std::atomic<uint64_t> events_seen{0};
std::atomic<uint64_t> dispatches_seen{0};

// Per event-type histogram (sized to ROCPROF_TRACE_DECODER_EVENT_LAST).
std::array<std::atomic<uint64_t>, ROCPROF_TRACE_DECODER_EVENT_LAST> event_counts{};

// Trace-interrupt counters. END is benign (signals last record per SE);
// the BUFFER_FULL flags mean we lost data — GPU_FULL = drainer too slow,
// CPU_FULL = our callback too slow.
std::atomic<uint64_t> flag_end{0};
std::atomic<uint64_t> flag_gpu_full{0};
std::atomic<uint64_t> flag_cpu_full{0};

// Cumulative timing, in nanoseconds.
//   scanner_ns_total: only the rocprof_trace_decoder_quick_scan call
//                     (including the in-API event callback we register).
//   post_ns_total:    everything our callback does AFTER quick_scan returns
//                     (currently just the mutex merge of any per-call locals).
//   callback_ns_total: scanner_ns + post_ns; the budget for not falling
//                      behind the GPU's data rate.
std::atomic<uint64_t> scanner_ns_total{0};
std::atomic<uint64_t> post_ns_total{0};
std::atomic<uint64_t> callback_ns_total{0};

// Time the trace context was actively collecting (ns). Accumulated across
// all roctxProfilerResume → roctxProfilerPause intervals. bytes_processed
// divided by this is the actual GPU trace bandwidth — i.e. the rate at
// which the SQTT path is producing data while it's enabled.
std::atomic<uint64_t> trace_active_ns{0};
std::atomic<uint64_t> resume_tsc_ns{0};  // 0 == not currently active

std::mutex hist_mu;
// Heap-allocated and intentionally leaked: rocprofiler-sdk invokes tool_fini
// during its own shutdown, which on this platform happens AFTER the
// namespace-scope std::unordered_map's destructor would otherwise run.
// Atomics above are trivially destructible so they survive; the maps are not.
//
// Dispatch entry-point histogram. Quick-scan emits a DISPATCH record per
// COMPUTE_DISPATCH_INITIATOR launch with entry_point.address holding the
// kernel program-start PC reconstructed from the COMPUTE_PGM_LO/HI register
// writes that preceded it. Counting distinct addresses tells us how many
// distinct kernels SQTT saw (vs how often each was launched).
inline std::unordered_map<uint64_t, uint64_t>&
dispatch_pc_hist()
{
    static auto* h = new std::unordered_map<uint64_t, uint64_t>();
    return *h;
}

// Per-call locals to keep the critical section tiny. thread_local so the
// maps' buckets are constructed once per consumer thread and reused.
struct CallLocal
{
    std::unordered_map<uint64_t, uint64_t> dispatch_pc;
    uint64_t                               dispatch_n = 0;

    // Cut tracking. Begin = target dispatch byte_offset. End = byte_offset
    // of the 2nd CS_PARTIAL_FLUSH after the dispatch on the same me/pipe.
    bool     cut_target_in_call    = false;
    uint64_t cut_offset_begin      = 0;
    uint8_t  target_me             = 0;
    uint8_t  target_pipe           = 0;
    uint64_t pf_count_after_target = 0;
    uint64_t cut_offset_end        = 0;
    bool     cut_ready             = false;
};

// Cut the standalone trace at the dispatch whose 1-based count equals this.
constexpr uint64_t TARGET_DISPATCH_ID = 20000;

std::atomic<uint64_t> dispatch_counter{0};
std::atomic<bool>     cut_captured{false};

struct CutTrace
{
    rocprof_trace_decoder_handle_t decoder      = {};
    uint64_t                       agent_handle = 0;
    uint64_t                       chunk_index  = 0;
    uint64_t                       begin        = 0;
    uint64_t                       end          = 0;
    std::vector<uint8_t>           bytes;
};
inline CutTrace&
cut_trace()
{
    static auto* p = new CutTrace{};
    return *p;
}
std::mutex cut_mu;

rocprofiler_thread_trace_decoder_status_t
event_callback(rocprofiler_thread_trace_decoder_record_type_t type,
               void*                                          records,
               uint64_t                                       n,
               void*                                          userdata)
{
    auto* cl = static_cast<CallLocal*>(userdata);

    if(type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH)
    {
        const auto* dispatches =
            static_cast<const rocprofiler_thread_trace_decoder_dispatch_t*>(records);
        dispatches_seen.fetch_add(n, std::memory_order_relaxed);
        cl->dispatch_n += n;
        for(uint64_t i = 0; i < n; ++i)
        {
            ++cl->dispatch_pc[dispatches[i].entry_point.address];
            uint64_t my_id =
                dispatch_counter.fetch_add(1, std::memory_order_relaxed) + 1;
            if(my_id >= TARGET_DISPATCH_ID &&
               !cut_captured.load(std::memory_order_acquire) &&
               !cl->cut_target_in_call)
            {
                cl->cut_target_in_call = true;
                cl->cut_offset_begin   = dispatches[i].byte_offset;
                cl->target_me          = dispatches[i].me_id;
                cl->target_pipe        = dispatches[i].pipe_id;
            }
        }
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
    }

    if(type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT)
    {
        const auto* events =
            static_cast<const rocprofiler_thread_trace_decoder_event_t*>(records);
        events_seen.fetch_add(n, std::memory_order_relaxed);
        for(uint64_t i = 0; i < n; ++i)
        {
            const auto& ev = events[i];
            if(ev.type < event_counts.size())
                event_counts[ev.type].fetch_add(1, std::memory_order_relaxed);

            if(cl->cut_target_in_call && !cl->cut_ready &&
               ev.type == ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH &&
               ev.me_id == cl->target_me && ev.pipe_id == cl->target_pipe &&
               ev.byte_offset > cl->cut_offset_begin)
            {
                if(++cl->pf_count_after_target == 2)
                {
                    cl->cut_offset_end = ev.byte_offset;
                    cl->cut_ready      = true;
                }
            }
        }
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

void
merge_dispatch_pc(const CallLocal& cl)
{
    if(cl.dispatch_pc.empty()) return;
    std::lock_guard<std::mutex> lock(hist_mu);
    auto&                       g = dispatch_pc_hist();
    for(const auto& [pc, c] : cl.dispatch_pc) g[pc] += c;
}

// Two-call build: the second call gets the size the decoder asked for if
// the first failed with OUT_OF_RESOURCES.
rocprofiler_thread_trace_decoder_status_t
build_standalone(rocprof_trace_decoder_handle_t handle,
                 uint64_t                       chunk_index,
                 const uint8_t*                 buf,
                 size_t                         size,
                 uint64_t                       offset_begin,
                 uint64_t                       offset_end,
                 std::vector<uint8_t>&          out)
{
    out.assign((offset_end - offset_begin) + 4096, 0);
    uint64_t out_size = out.size();
    auto     st       = rocprof_trace_decoder_build_standalone(
        handle, chunk_index, buf, size, offset_begin, offset_end, out.data(), &out_size);
    if(st == ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES)
    {
        out.resize(out_size);
        out_size = out.size();
        st       = rocprof_trace_decoder_build_standalone(
            handle, chunk_index, buf, size, offset_begin, offset_end, out.data(), &out_size);
    }
    if(st == ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) out.resize(out_size);
    return st;
}

void
try_capture_cut(rocprof_trace_decoder_handle_t handle,
                uint64_t                       agent_handle,
                uint64_t                       chunk_index,
                const uint8_t*                 buf,
                size_t                         size,
                const CallLocal&               cl)
{
    if(!cl.cut_target_in_call) return;

    if(!cl.cut_ready)
    {
        std::fprintf(
            stderr,
            "[scan] skip cut: chunk=%lu begin=%lu, only %lu CS_PARTIAL_FLUSH(me=%u,pipe=%u) seen, need 2\n",
            (unsigned long) chunk_index,
            (unsigned long) cl.cut_offset_begin,
            (unsigned long) cl.pf_count_after_target,
            (unsigned) cl.target_me,
            (unsigned) cl.target_pipe);
        return;
    }

    bool expected = false;
    if(!cut_captured.compare_exchange_strong(
           expected, true, std::memory_order_acq_rel, std::memory_order_acquire))
        return;

    std::vector<uint8_t> out;
    auto                 st = build_standalone(
        handle, chunk_index, buf, size, cl.cut_offset_begin, cl.cut_offset_end, out);

    if(st != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
    {
        std::fprintf(stderr,
                     "[scan] build_standalone failed: chunk=%lu begin=%lu end=%lu : %s\n",
                     (unsigned long) chunk_index,
                     (unsigned long) cl.cut_offset_begin,
                     (unsigned long) cl.cut_offset_end,
                     rocprof_trace_decoder_get_status_string(st));
        cut_captured.store(false, std::memory_order_release);
        return;
    }

    size_t bytes = out.size();
    {
        std::lock_guard<std::mutex> lk(cut_mu);
        auto&                       ct = cut_trace();
        ct.decoder                     = handle;
        ct.agent_handle                = agent_handle;
        ct.chunk_index                 = chunk_index;
        ct.begin                       = cl.cut_offset_begin;
        ct.end                         = cl.cut_offset_end;
        ct.bytes                       = std::move(out);
    }
    std::fprintf(stderr,
                 "[scan] cut dispatch >= #%lu: chunk=%lu bytes %lu..%lu (me=%u,pipe=%u) -> %lu B\n",
                 (unsigned long) TARGET_DISPATCH_ID,
                 (unsigned long) chunk_index,
                 (unsigned long) cl.cut_offset_begin,
                 (unsigned long) cl.cut_offset_end,
                 (unsigned) cl.target_me,
                 (unsigned) cl.target_pipe,
                 (unsigned long) bytes);
}

void
scan_inline(rocprof_trace_decoder_handle_t handle,
            uint64_t                       agent_handle,
            uint64_t                       chunk_index,
            const uint8_t*                 buf,
            size_t                         size)
{
    thread_local CallLocal cl;
    cl = {};

    auto t0 = std::chrono::steady_clock::now();
    rocprof_trace_decoder_quick_scan(handle, chunk_index, buf, size, &event_callback, &cl);
    auto t1 = std::chrono::steady_clock::now();

    chunks_processed.fetch_add(1, std::memory_order_relaxed);
    bytes_processed.fetch_add(size, std::memory_order_relaxed);

    merge_dispatch_pc(cl);
    try_capture_cut(handle, agent_handle, chunk_index, buf, size, cl);

    auto t2 = std::chrono::steady_clock::now();
    auto ns = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };
    scanner_ns_total.fetch_add(ns(t0, t1), std::memory_order_relaxed);
    post_ns_total.fetch_add(ns(t1, t2), std::memory_order_relaxed);
    callback_ns_total.fetch_add(ns(t0, t2), std::memory_order_relaxed);
}
}  // namespace ScanState

namespace ATTClient
{
// One decoder handle per GPU agent, keyed by rocprofiler_agent_id_t::handle.
// quick_scan, codeobj_load and parse must all be issued on the handle that
// belongs to the agent the chunk/code object came from — decoder state and
// loaded code objects are per-agent. The map is populated in
// query_available_agents at startup, before any thread trace or code-object
// callback can fire, so callbacks read it lock-free.
inline std::unordered_map<uint64_t, rocprof_trace_decoder_handle_t>&
decoders()
{
    static auto* m = new std::unordered_map<uint64_t, rocprof_trace_decoder_handle_t>();
    return *m;
}
rocprofiler_context_id_t agent_ctx{};
rocprofiler_context_id_t tracing_ctx{};

constexpr uint64_t TARGET_CU                 = 1;
constexpr uint64_t SHADER_MASK               = 0x1;
constexpr uint64_t GPU_BUFFER_SIZE_DEFAULT   = 64ul << 20;
constexpr size_t   NUM_BUFFERS               = 4;

// Allow override at startup for sweeping.  GPU_BUFFER_SIZE_MB=N picks N MB.
inline uint64_t
gpu_buffer_size()
{
    if(const char* env = std::getenv("GPU_BUFFER_SIZE_MB"))
    {
        char*    end = nullptr;
        uint64_t v   = std::strtoull(env, &end, 10);
        if(end != env && v > 0) return v << 20;
    }
    return GPU_BUFFER_SIZE_DEFAULT;
}

void
shader_data_callback(rocprofiler_agent_id_t                       agent,
                     int64_t /*se_id*/,
                     uint64_t                                     chunk_index,
                     void*                                        se_data,
                     size_t                                       data_size,
                     rocprofiler_thread_trace_shader_data_flags_t flags,
                     rocprofiler_user_data_t                      userdata)
{
    if(flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END)
        ScanState::flag_end.fetch_add(1, std::memory_order_relaxed);
    if(flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL)
    {
        std::cout << "GPU full for chunk: " << chunk_index << std::endl;
        ScanState::flag_gpu_full.fetch_add(1, std::memory_order_relaxed);
    }
    if(flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL)
    {
        std::cout << "CPU full for chunk: " << chunk_index << std::endl;
        ScanState::flag_cpu_full.fetch_add(1, std::memory_order_relaxed);
    }

    if(data_size == 0 || se_data == nullptr) return;

    // chunk_index 0 is the gfx9 trace header; payload chunks follow at 1, 2, ...
    // The decoder handle for this agent is bound into userdata at service-
    // configure time in query_available_agents — reconstruct it directly to
    // avoid a per-callback map lookup on the hot path.
    rocprof_trace_decoder_handle_t handle{userdata.value};
    ScanState::scan_inline(handle,
                           agent.handle,
                           chunk_index,
                           static_cast<const uint8_t*>(se_data),
                           data_size);
}

rocprofiler_status_t
query_available_agents(rocprofiler_agent_version_t /*version*/,
                       const void** agents,
                       size_t       num_agents,
                       void* /*user_data*/)
{
    auto parameters = std::vector<rocprofiler_thread_trace_parameter_t>{};
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, {TARGET_CU}});
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, {SHADER_MASK}});
    uint64_t buf_size = gpu_buffer_size();
    std::fprintf(stderr, "[scan] GPU_BUFFER_SIZE = %lu bytes (%lu MB)\n",
                 (unsigned long) buf_size, (unsigned long) (buf_size >> 20));
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {buf_size}});
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_NUM_BUFFERS, {NUM_BUFFERS}});

    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
        if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        // One decoder handle per GPU agent. quick_scan keeps register-
        // tracking state on the handle; loading code objects also targets
        // a specific handle. Created here, before any callback can run.
        // The handle is also kept in decoders() so codeobj_tracing_callback
        // can find it by agent_id (code-object loads are rare; this map is
        // not on the hot path).
        rocprof_trace_decoder_handle_t h{};
        rocprof_trace_decoder_create_handle(&h);
        decoders()[agent->id.handle] = h;

        // Pass the handle to shader_data_callback via userdata so the hot
        // path doesn't have to do a per-chunk map lookup.
        rocprofiler_user_data_t user{};
        user.value = h.handle;
        ROCPROFILER_CALL(
            rocprofiler_configure_device_thread_trace_service(agent_ctx,
                                                              agent->id,
                                                              parameters.data(),
                                                              parameters.size(),
                                                              shader_data_callback,
                                                              user),
            "thread trace service configure");
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

void
codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                         rocprofiler_user_data_t* /*user_data*/,
                         void* /*cb_data*/)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT) return;
    if(record.operation != ROCPROFILER_CODE_OBJECT_LOAD) return;
    if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

    auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);
    if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) return;

    const auto* memorybase = reinterpret_cast<const void*>(data->memory_base);
    if(memorybase == nullptr) return;

    // Code objects are scoped to a specific agent — load only onto that
    // agent's decoder handle. Other agents' handles must not see this load.
    auto it = decoders().find(data->agent_id.handle);
    if(it == decoders().end())
    {
        std::fprintf(stderr,
                     "[scan] codeobj_load(id=%lu): no decoder for agent.handle=%lu\n",
                     (unsigned long) data->code_object_id,
                     (unsigned long) data->agent_id.handle);
        return;
    }
    auto st = rocprof_trace_decoder_codeobj_load(it->second,
                                                 data->code_object_id,
                                                 data->load_delta,
                                                 data->load_size,
                                                 memorybase,
                                                 data->memory_size);
    if(st != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
        std::fprintf(stderr,
                     "[scan] codeobj_load(id=%lu) returned %s\n",
                     (unsigned long) data->code_object_id,
                     rocprof_trace_decoder_get_status_string(st));
}

void
cntrl_tracing_callback(rocprofiler_callback_tracing_record_t record,
                       rocprofiler_user_data_t* /*user_data*/,
                       void* /*cb_data*/)
{
    if(record.kind != ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API) return;

    if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER &&
       record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerPause)
    {
        // Stamp the close of the active interval BEFORE issuing stop_context,
        // so stop overhead isn't counted as trace-active time.
        uint64_t resumed = ScanState::resume_tsc_ns.exchange(0, std::memory_order_acq_rel);
        if(resumed != 0)
        {
            uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now().time_since_epoch())
                               .count();
            ScanState::trace_active_ns.fetch_add(now - resumed, std::memory_order_relaxed);
        }
        ROCPROFILER_CALL(rocprofiler_stop_context(agent_ctx), "stopping context");
    }
    else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT &&
            record.operation == ROCPROFILER_MARKER_CONTROL_API_ID_roctxProfilerResume)
    {
        // Open the active interval AFTER start_context returns, so start
        // overhead isn't counted as trace-active time.
        ROCPROFILER_CALL(rocprofiler_start_context(agent_ctx), "starting context");
        uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
        ScanState::resume_tsc_ns.store(now, std::memory_order_release);
    }
}

int
tool_init(rocprofiler_client_finalize_t /*fini_func*/, void* /*tool_data*/)
{
    // Per-agent decoder handles are created lazily inside
    // query_available_agents (one per GPU agent we configure thread-trace
    // on). The rocprofiler-sdk thread-trace path requires a valid handle
    // for triple-buffer mode; code objects loaded via
    // codeobj_tracing_callback are stashed on the matching agent's handle
    // so the post-mortem parse in tool_fini can disassemble instructions
    // for hotspots.
    ROCPROFILER_CALL(rocprofiler_create_context(&agent_ctx), "context creation");
    ROCPROFILER_CALL(rocprofiler_create_context(&tracing_ctx), "context creation");

    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         tracing_ctx,
                         ROCPROFILER_CALLBACK_TRACING_MARKER_CONTROL_API,
                         nullptr,
                         0,
                         cntrl_tracing_callback,
                         nullptr),
                     "marker tracing callback service configure");

    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(
                         tracing_ctx,
                         ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                         nullptr,
                         0,
                         codeobj_tracing_callback,
                         nullptr),
                     "code object tracing callback service configure");

    ROCPROFILER_CALL(rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                                        &query_available_agents,
                                                        sizeof(rocprofiler_agent_t),
                                                        nullptr),
                     "Failed to find GPU agents");

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(agent_ctx, &valid_ctx), "validity check");
    if(valid_ctx == 0) throw std::runtime_error("agent_ctx is not valid!");
    ROCPROFILER_CALL(rocprofiler_context_is_valid(tracing_ctx, &valid_ctx), "validity check");
    if(valid_ctx == 0) throw std::runtime_error("tracing_ctx is not valid!");

    ROCPROFILER_CALL(rocprofiler_start_context(tracing_ctx), "context start");

    return 0;
}

// Aggregated stats from the post-mortem rocprof_trace_decoder_parse call
// over the standalone cut trace. Filled from inside the parse callback;
// printed by tool_fini.
struct CutStats
{
    uint64_t                                                                  waves_started = 0;
    uint64_t                                                                  waves_ended   = 0;
    uint64_t                                                                  insts         = 0;
    std::unordered_map<uint64_t, std::pair<uint64_t /*hits*/, uint64_t /*lat*/>> pc_hist;
};

rocprofiler_thread_trace_decoder_status_t
parse_callback(rocprofiler_thread_trace_decoder_record_type_t type,
               void*                                          records,
               uint64_t                                       n,
               void*                                          userdata)
{
    auto* st = static_cast<CutStats*>(userdata);
    if(type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_OCCUPANCY)
    {
        const auto* occ =
            static_cast<const rocprofiler_thread_trace_decoder_occupancy_t*>(records);
        for(uint64_t i = 0; i < n; ++i)
        {
            if(occ[i].start)
                ++st->waves_started;
            else
                ++st->waves_ended;
        }
    }
    else if(type == ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE)
    {
        const auto* waves = static_cast<const rocprofiler_thread_trace_decoder_wave_t*>(records);
        for(uint64_t w = 0; w < n; ++w)
        {
            const auto& wv = waves[w];
            for(uint64_t i = 0; i < wv.instructions_size; ++i)
            {
                const auto& inst = wv.instructions_array[i];
                ++st->insts;
                auto& slot = st->pc_hist[inst.pc.address];
                slot.first  += 1;
                slot.second += static_cast<uint64_t>(inst.duration < 0 ? 0 : inst.duration);
            }
        }
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

void
tool_fini(void* /*tool_data*/)
{
    size_t events      = ScanState::events_seen.load();
    size_t dispatches  = ScanState::dispatches_seen.load();
    size_t bytes       = ScanState::bytes_processed.load();
    size_t chunks      = ScanState::chunks_processed.load();
    size_t scanner_ns  = ScanState::scanner_ns_total.load();
    size_t post_ns     = ScanState::post_ns_total.load();
    size_t callback_ns = ScanState::callback_ns_total.load();

    // If the process exited while still resumed (no closing roctxProfilerPause),
    // close the interval here so trace_active_ns reflects everything.
    uint64_t resumed = ScanState::resume_tsc_ns.exchange(0, std::memory_order_acq_rel);
    if(resumed != 0)
    {
        uint64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::steady_clock::now().time_since_epoch())
                           .count();
        ScanState::trace_active_ns.fetch_add(now - resumed, std::memory_order_relaxed);
    }
    size_t trace_active_ns = ScanState::trace_active_ns.load();

    double bytes_gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "\n[scan] chunks=" << chunks
              << " bytes=" << bytes
              << " (" << bytes_gb << " GB)"
              << " events=" << events
              << " dispatches=" << dispatches << "\n";

    // Throughput, broken into:
    //   scanner: rocprof_trace_decoder_quick_scan (the AVX-512 path + our cb)
    //   post:    our merge of per-call dispatch histogram into the global one
    //   total:   sum, the full callback budget
    auto report = [bytes](const char* label, size_t ns) {
        if(ns == 0 || bytes == 0) return;
        double seconds  = ns / 1e9;
        double gb_per_s = (double) bytes / seconds / (1024.0 * 1024.0 * 1024.0);
        std::printf(
            "[scan] %-8s  wall=%.3fs  throughput=%.2f GB/s\n",
            label, seconds, gb_per_s);
    };
    report("scanner", scanner_ns);
    report("post",    post_ns);
    report("total",   callback_ns);

    // GPU trace bandwidth: bytes / wall time the SQTT context was active.
    if(trace_active_ns > 0 && bytes > 0)
    {
        double seconds  = trace_active_ns / 1e9;
        double gb_per_s = (double) bytes / seconds / (1024.0 * 1024.0 * 1024.0);
        std::printf("[scan] trace     active=%.3fs  bandwidth=%.2f GB/s"
                    "  (GPU-side SQTT data rate while resumed)\n",
                    seconds, gb_per_s);
    }

    // Trace-interrupt flags. END is just per-SE end markers; the BUFFER_FULL
    // flags mean we lost data and are worth surfacing prominently.
    uint64_t end_n = ScanState::flag_end.load();
    uint64_t gpu_n = ScanState::flag_gpu_full.load();
    uint64_t cpu_n = ScanState::flag_cpu_full.load();
    std::cout << "[scan] flags: END=" << end_n << " GPU_BUFFER_FULL=" << gpu_n
              << " CPU_BUFFER_FULL=" << cpu_n;
    if(gpu_n != 0 || cpu_n != 0)
        std::cout << "  *** TRACE INTERRUPTED — data was lost ***";
    std::cout << "\n";

    std::cout << "[scan] event counts:";
    for(size_t i = 0; i < ScanState::event_counts.size(); i++)
    {
        uint64_t c = ScanState::event_counts[i].load();
        if(c != 0) std::cout << " e" << i << "=" << c;
    }
    std::cout << "\n";

    // Dispatch entry-point histogram. quick_scan emits one DISPATCH record
    // per launch with entry_point.address = the kernel program-start PC the
    // decoder reconstructed from COMPUTE_PGM_LO/HI; counting distinct PCs
    // tells us how many distinct kernels were launched (vs the per-PC counts
    // for launch frequency).
    {
        const auto& h = ScanState::dispatch_pc_hist();
        std::vector<std::pair<uint64_t, uint64_t>> v(h.begin(), h.end());
        std::sort(v.begin(), v.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::cout << "[scan] dispatch entry points (" << v.size() << " distinct):\n";
        for(const auto& [pc, count] : v)
            std::printf("  pc=0x%012lx  count=%lu\n",
                        static_cast<unsigned long>(pc),
                        static_cast<unsigned long>(count));
    }

    // Post-mortem parse of the standalone trace cut at TARGET_DISPATCH_ID.
    // Mirrors what samples/thread_trace does on every chunk, but here only
    // for the single ~one-dispatch slice we extracted via build_standalone.
    {
        auto& ct = ScanState::cut_trace();
        if(ct.bytes.empty())
        {
            std::printf("[scan] cut trace: NOT CAPTURED (TARGET_DISPATCH_ID=%lu, %lu dispatches seen)\n",
                        (unsigned long) ScanState::TARGET_DISPATCH_ID,
                        (unsigned long) ScanState::dispatch_counter.load());
        }
        else
        {
            char path[64];
            std::snprintf(path,
                          sizeof(path),
                          "cut_dispatch_%lu.att",
                          (unsigned long) ScanState::TARGET_DISPATCH_ID);
            std::ofstream ofs(path, std::ios::binary);
            if(ofs)
            {
                ofs.write(reinterpret_cast<const char*>(ct.bytes.data()),
                          static_cast<std::streamsize>(ct.bytes.size()));
                std::printf("[scan] wrote %lu B to %s\n",
                            (unsigned long) ct.bytes.size(),
                            path);
            }
            else
            {
                std::fprintf(stderr, "[scan] failed to open %s for writing\n", path);
            }

            // Parse must run on the same agent's handle that produced the
            // standalone bytes — code objects are loaded per-handle, so a
            // different handle would fail to disassemble. We stored the
            // exact decoder used at cut time on the CutTrace itself.
            CutStats stats{};
            auto pst = rocprof_trace_decoder_parse(ct.decoder,
                                                   ct.bytes.data(),
                                                   ct.bytes.size(),
                                                   &parse_callback,
                                                   &stats);
            std::printf("[scan] cut trace: agent=%lu dispatch #%lu chunk=%lu range=%lu..%lu (%lu B standalone) parse=%s\n",
                        (unsigned long) ct.agent_handle,
                        (unsigned long) ScanState::TARGET_DISPATCH_ID,
                        (unsigned long) ct.chunk_index,
                        (unsigned long) ct.begin,
                        (unsigned long) ct.end,
                        (unsigned long) ct.bytes.size(),
                        rocprof_trace_decoder_get_status_string(pst));

            if(pst == ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
            {
                std::printf("[scan] cut stats: waves_started=%lu waves_ended=%lu instructions=%lu unique_pcs=%lu\n",
                            (unsigned long) stats.waves_started,
                            (unsigned long) stats.waves_ended,
                            (unsigned long) stats.insts,
                            (unsigned long) stats.pc_hist.size());

                using Entry = std::pair<uint64_t, std::pair<uint64_t, uint64_t>>;
                std::vector<Entry> hot(stats.pc_hist.begin(), stats.pc_hist.end());
                std::sort(hot.begin(), hot.end(),
                          [](const Entry& a, const Entry& b) {
                              if(a.second.second != b.second.second)
                                  return a.second.second > b.second.second;
                              return a.second.first > b.second.first;
                          });
                size_t topn = hot.size() < 10 ? hot.size() : 10;
                std::printf("[scan] cut top %lu hotspots (by latency cycles):\n",
                            (unsigned long) topn);
                for(size_t i = 0; i < topn; ++i)
                    std::printf("  pc=0x%012lx  hits=%lu  latency=%lu\n",
                                (unsigned long) hot[i].first,
                                (unsigned long) hot[i].second.first,
                                (unsigned long) hot[i].second.second);
            }
        }
    }

    for(auto& [agent_handle, h] : decoders())
    {
        (void) agent_handle;
        rocprof_trace_decoder_destroy_handle(h);
    }
}
}  // namespace ATTClient

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/,
                      const char* /*runtime_version*/,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name = "Thread Trace Quick-Scan Sample";
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ATTClient::tool_init,
                                            &ATTClient::tool_fini,
                                            nullptr};
    return &cfg;
}
