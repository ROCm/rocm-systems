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
// streaming use case for the quick-scan path: bandwidth/interrupt counters
// are aggregated across the run, and one dispatch slice is saved for the
// shared hotspot/disassembly report.
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

#include "thread_trace_hotspots.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <rocprof_trace_decoder/rocprof_trace_decoder.h>
#include <rocprof_trace_decoder/trace_decoder_types.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include <vector>

#define ROCPROFILER_CALL(result, msg)                                                              \
    if(auto ec = (result); ec != ROCPROFILER_STATUS_SUCCESS)                                       \
    {                                                                                              \
        std::cerr << "rocprofiler-sdk error at " << __FILE__ << ":" << __LINE__                    \
                  << " :: " << #result << " :: " << msg                                            \
                  << " :: " << rocprofiler_get_status_string(ec) << std::endl;                     \
        abort();                                                                                   \
    }

namespace ScanState
{
// Triple-buffer mode delivers the 8-byte gfx9 trace header at chunk_index 0
// before any payload chunk; payload chunks arrive at chunk_index 1, 2, ...
// The handle-based quick_scan API consumes the header on the chunk_index 0
// call and threads register-tracking state through the handle to subsequent
// chunks, so we no longer need to cache the header here.

// Hot counters use atomics because shader-data callbacks can be delivered
// asynchronously by rocprofiler-sdk.
std::atomic<uint64_t> chunks_processed{0};
std::atomic<uint64_t> bytes_processed{0};
std::atomic<uint64_t> events_seen{0};

// Trace-interrupt counters. END is benign (signals last record per SE);
// the BUFFER_FULL flags mean we lost data — GPU_FULL = drainer too slow,
// CPU_FULL = our callback too slow.
std::atomic<uint64_t> flag_end{0};
std::atomic<uint64_t> flag_gpu_full{0};
std::atomic<uint64_t> flag_cpu_full{0};

std::atomic<uint64_t> callback_ns_total{0};

struct CallLocal
{
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
    rocprof_trace_decoder_handle_t decoder     = {};
    uint64_t                       chunk_index = 0;
    uint64_t                       begin       = 0;
    uint64_t                       end         = 0;
    std::vector<uint8_t>           bytes;
};
inline CutTrace&
cut_trace()
{
    static auto* p = new CutTrace{};
    return *p;
}

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
        for(uint64_t i = 0; i < n; ++i)
        {
            uint64_t my_id = dispatch_counter.fetch_add(1, std::memory_order_relaxed) + 1;
            if(my_id >= TARGET_DISPATCH_ID && !cut_captured.load(std::memory_order_acquire) &&
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
        const auto* events = static_cast<const rocprofiler_thread_trace_decoder_event_t*>(records);
        events_seen.fetch_add(n, std::memory_order_relaxed);
        for(uint64_t i = 0; i < n; ++i)
        {
            const auto& ev = events[i];
            if(cl->cut_target_in_call && !cl->cut_ready && ev.me_id == cl->target_me &&
               ev.pipe_id == cl->target_pipe && ev.byte_offset > cl->cut_offset_begin)
            {
                if(ev.type == ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH ||
                   cl->pf_count_after_target != 0)
                {
                    if(++cl->pf_count_after_target == 2)
                    {
                        cl->cut_offset_end = ev.byte_offset;
                        cl->cut_ready      = true;
                    }
                }
            }
        }
    }
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
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
                uint64_t                       chunk_index,
                const uint8_t*                 buf,
                size_t                         size,
                const CallLocal&               cl)
{
    if(!cl.cut_target_in_call) return;

    if(!cl.cut_ready)
    {
        std::fprintf(stderr,
                     "[scan] skip cut: chunk=%lu begin=%lu, only %lu "
                     "CS_PARTIAL_FLUSH(me=%u,pipe=%u) seen, need 2\n",
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

    size_t bytes   = out.size();
    auto&  ct      = cut_trace();
    ct.decoder     = handle;
    ct.chunk_index = chunk_index;
    ct.begin       = cl.cut_offset_begin;
    ct.end         = cl.cut_offset_end;
    ct.bytes       = std::move(out);
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
            uint64_t                       chunk_index,
            const uint8_t*                 buf,
            size_t                         size)
{
    thread_local CallLocal cl;
    cl = {};

    auto t0 = std::chrono::steady_clock::now();
    rocprof_trace_decoder_quick_scan(handle, chunk_index, buf, size, &event_callback, &cl);

    chunks_processed.fetch_add(1, std::memory_order_relaxed);
    bytes_processed.fetch_add(size, std::memory_order_relaxed);

    try_capture_cut(handle, chunk_index, buf, size, cl);

    auto t2 = std::chrono::steady_clock::now();
    auto ns = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
    };
    callback_ns_total.fetch_add(ns(t0, t2), std::memory_order_relaxed);
}
}  // namespace ScanState

namespace ATTClient
{
struct AgentTraceState
{
    rocprofiler_context_id_t       context{};
    rocprof_trace_decoder_handle_t decoder{};
    const char*                    agent_name = "unknown";
    bool                           started    = false;
    uint64_t                       start_ns   = 0;
    uint64_t                       stop_ns    = 0;
};

// Leaked because tool_fini runs after static destructors.
inline std::unordered_map<uint64_t, AgentTraceState>&
agent_states()
{
    static auto* m = new std::unordered_map<uint64_t, AgentTraceState>();
    return *m;
}
inline std::mutex&
agent_state_lock()
{
    static auto* m = new std::mutex{};
    return *m;
}

rocprofiler_context_id_t tracing_ctx{};

uint64_t
now_ns()
{
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                     std::chrono::steady_clock::now().time_since_epoch())
                                     .count());
}

AgentTraceState*
start_agent_context(uint64_t agent_handle)
{
    std::lock_guard<std::mutex> lk(agent_state_lock());

    auto it = agent_states().find(agent_handle);
    if(it == agent_states().end())
    {
        std::fprintf(stderr,
                     "[scan] codeobj_load: no ATT context for agent.handle=%lu\n",
                     (unsigned long) agent_handle);
        return nullptr;
    }

    auto& state = it->second;
    if(!state.started)
    {
        ROCPROFILER_CALL(rocprofiler_start_context(state.context), "starting context");
        state.started  = true;
        state.start_ns = now_ns();
    }

    return &state;
}

void
stop_started_agent_contexts()
{
    std::lock_guard<std::mutex> lk(agent_state_lock());

    for(auto& [agent_handle, state] : agent_states())
    {
        (void) agent_handle;
        if(!state.started) continue;

        auto status = rocprofiler_stop_context(state.context);
        if(status != ROCPROFILER_STATUS_SUCCESS &&
           status != ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND)
            ROCPROFILER_CALL(status, "stopping context");
        state.started = false;
        state.stop_ns = now_ns();
    }
}

uint64_t
trace_active_span_ns()
{
    std::lock_guard<std::mutex> lk(agent_state_lock());

    uint64_t begin = 0;
    uint64_t end   = 0;
    for(auto& [agent_handle, state] : agent_states())
    {
        (void) agent_handle;
        if(state.start_ns == 0 || state.stop_ns == 0) continue;
        if(begin == 0 || state.start_ns < begin) begin = state.start_ns;
        if(state.stop_ns > end) end = state.stop_ns;
    }

    return (begin != 0 && end > begin) ? end - begin : 0;
}

constexpr uint64_t TARGET_CU               = 1;
constexpr uint64_t SHADER_MASK             = 0x1;
constexpr uint64_t GPU_BUFFER_SIZE_DEFAULT = 64ul << 20;
constexpr size_t   NUM_BUFFERS             = 6;

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
shader_data_callback(rocprofiler_agent_id_t /*agent*/,
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
        ScanState::flag_gpu_full.fetch_add(1, std::memory_order_relaxed);
    if(flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL)
        ScanState::flag_cpu_full.fetch_add(1, std::memory_order_relaxed);

    if(data_size == 0 || se_data == nullptr) return;

    // chunk_index 0 is the gfx9 trace header; payload chunks follow at 1, 2, ...
    // The per-agent state is bound into userdata at service-configure time
    // so the hot path avoids a map lookup.
    auto* state = static_cast<AgentTraceState*>(userdata.ptr);
    if(state == nullptr) return;

    ScanState::scan_inline(
        state->decoder, chunk_index, static_cast<const uint8_t*>(se_data), data_size);
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
    std::fprintf(stderr,
                 "[scan] GPU_BUFFER_SIZE = %lu bytes (%lu MB)\n",
                 (unsigned long) buf_size,
                 (unsigned long) (buf_size >> 20));
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, {buf_size}});
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_NUM_BUFFERS, {NUM_BUFFERS}});

    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
        if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        auto& state      = agent_states()[agent->id.handle];
        state.agent_name = (agent->name != nullptr) ? agent->name : "unknown";

        ROCPROFILER_CALL(rocprofiler_create_context(&state.context), "context creation");

        auto dst = rocprof_trace_decoder_create_handle(&state.decoder);
        if(dst != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
        {
            std::cerr << "Decoder error at " << __FILE__ << ":" << __LINE__
                      << " :: " << rocprof_trace_decoder_get_status_string(dst) << std::endl;
            abort();
        }

        // Pass the per-agent state to shader_data_callback via userdata so
        // the hot path doesn't have to do a per-chunk map lookup.
        rocprofiler_user_data_t user{};
        user.ptr = &state;
        ROCPROFILER_CALL(rocprofiler_configure_device_thread_trace_service(state.context,
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

    auto* state = start_agent_context(data->agent_id.handle);

    // Register with the shared disassembler-side AddressTable for both
    // FILE- and MEMORY-backed code objects so write_top_hotspots can
    // disassemble instructions at PC.
    hotspots::register_codeobj_disasm(*data);

    if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY &&
       data->memory_base != 0 && data->memory_size != 0)
    {
        char path[256];
        std::snprintf(path,
                      sizeof(path),
                      "%s_code_object_id_%lu.out",
                      (state != nullptr) ? state->agent_name : "unknown",
                      (unsigned long) data->code_object_id);
        std::ofstream ofs(path, std::ios::binary);
        if(ofs)
            ofs.write(reinterpret_cast<const char*>(data->memory_base),
                      static_cast<std::streamsize>(data->memory_size));
    }

    if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) return;

    const auto* memorybase = reinterpret_cast<const void*>(data->memory_base);
    if(memorybase == nullptr) return;

    // Code objects are scoped to a specific agent — load only onto that
    // agent's decoder handle. Other agents' handles must not see this load.
    if(state == nullptr) return;
    auto st = rocprof_trace_decoder_codeobj_load(state->decoder,
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

int
tool_init(rocprofiler_client_finalize_t /*fini_func*/, void* /*tool_data*/)
{
    // Per-agent ATT contexts and decoder handles are created inside
    // query_available_agents. codeobj_tracing_callback starts the matching
    // agent context on first code-object load.
    ROCPROFILER_CALL(rocprofiler_create_context(&tracing_ctx), "context creation");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(tracing_ctx,
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
    for(auto& [agent_handle, state] : agent_states())
    {
        (void) agent_handle;
        ROCPROFILER_CALL(rocprofiler_context_is_valid(state.context, &valid_ctx), "validity check");
        if(valid_ctx == 0) throw std::runtime_error("agent context is not valid!");
    }
    ROCPROFILER_CALL(rocprofiler_context_is_valid(tracing_ctx, &valid_ctx), "validity check");
    if(valid_ctx == 0) throw std::runtime_error("tracing_ctx is not valid!");

    ROCPROFILER_CALL(rocprofiler_start_context(tracing_ctx), "context start");

    return 0;
}

void
tool_fini(void* /*tool_data*/)
{
    stop_started_agent_contexts();

    size_t events      = ScanState::events_seen.load();
    size_t dispatches  = ScanState::dispatch_counter.load();
    size_t bytes       = ScanState::bytes_processed.load();
    size_t chunks      = ScanState::chunks_processed.load();
    size_t callback_ns = ScanState::callback_ns_total.load();
    size_t active_ns   = trace_active_span_ns();

    double bytes_gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "\n[scan] chunks=" << chunks << " bytes=" << bytes << " (" << bytes_gb << " GB)"
              << " events=" << events << " dispatches=" << dispatches << "\n";

    auto report = [bytes](const char* label, size_t ns) {
        if(ns == 0 || bytes == 0) return;
        double seconds  = ns / 1e9;
        double gb_per_s = (double) bytes / seconds / (1024.0 * 1024.0 * 1024.0);
        std::printf("[scan] %-8s  wall=%.3fs  throughput=%.2f GB/s\n", label, seconds, gb_per_s);
    };
    report("scan", callback_ns);
    report("trace", active_ns);

    // Trace-interrupt flags. END is just per-SE end markers; the BUFFER_FULL
    // flags mean we lost data and are worth surfacing prominently.
    uint64_t end_n = ScanState::flag_end.load();
    uint64_t gpu_n = ScanState::flag_gpu_full.load();
    uint64_t cpu_n = ScanState::flag_cpu_full.load();
    std::cout << "[scan] flags: END=" << end_n << " GPU_BUFFER_FULL=" << gpu_n
              << " CPU_BUFFER_FULL=" << cpu_n;
    if(gpu_n != 0 || cpu_n != 0) std::cout << "  *** TRACE INTERRUPTED — data was lost ***";
    std::cout << "\n";

    // Post-mortem parse of the standalone trace cut at TARGET_DISPATCH_ID.
    // Mirrors what samples/thread_trace does on every chunk, but here only
    // for the single ~one-dispatch slice we extracted via build_standalone.
    {
        auto& ct = ScanState::cut_trace();
        if(ct.bytes.empty())
        {
            std::printf(
                "[scan] cut trace: NOT CAPTURED (TARGET_DISPATCH_ID=%lu, %lu dispatches seen)\n",
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
                std::printf("[scan] wrote %lu B to %s\n", (unsigned long) ct.bytes.size(), path);
            }
            else
            {
                std::fprintf(stderr, "[scan] failed to open %s for writing\n", path);
            }

            // Parse must run on the same agent's handle that produced the
            // standalone bytes — code objects are loaded per-handle, so a
            // different handle would fail to disassemble. We stored the
            // exact decoder used at cut time on the CutTrace itself.
            // Feed the per-instruction records into the shared latency
            // table so write_top_hotspots below produces a disassembled
            // top-N report identical in format to the simple agent.
            auto pst = rocprof_trace_decoder_parse(
                ct.decoder, ct.bytes.data(), ct.bytes.size(), &hotspots::accumulate, nullptr);
            std::printf("[scan] cut trace: dispatch #%lu chunk=%lu range=%lu..%lu "
                        "(%lu B standalone) parse=%s\n",
                        (unsigned long) ScanState::TARGET_DISPATCH_ID,
                        (unsigned long) ct.chunk_index,
                        (unsigned long) ct.begin,
                        (unsigned long) ct.end,
                        (unsigned long) ct.bytes.size(),
                        rocprof_trace_decoder_get_status_string(pst));
        }
    }

    // Disassembled top-N hotspot report, shared with the simple sample.
    hotspots::write_top_hotspots("thread_trace_buffered.log");

    for(auto& [agent_handle, state] : agent_states())
    {
        (void) agent_handle;
        rocprof_trace_decoder_destroy_handle(state.decoder);
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
