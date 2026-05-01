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

// Triple-buffered SQTT capture that runs rocprof-trace-decoder's gfx9 rare-token
// "quick scan" inline on each shader-data chunk as it arrives. Demonstrates the
// streaming use case for the rare-scan path: per-register-address counts are
// aggregated across the run.
//
// First implementation note: we deliberately scan inline on the rocprofiler-sdk
// consumer (shader-data) callback thread rather than handing off to a worker.
// Two reasons:
//   1. The AVX-512 rare scanner runs comfortably faster than std::vector::assign
//      could *copy* the chunk, so deferring would actually be slower than just
//      doing the work here.
//   2. The SDK owns the chunk memory across its triple buffer; if we returned
//      from the callback while still holding the pointer, the next GPU drain
//      could overwrite it. Inline scanning keeps the lifetime obviously safe.
// A future revision that needs to overlap scan time with decode work can fan
// the chunks out to a worker (with a copy or a pinned buffer pool).
//
// The rare-scan API is consumed via a directly-linked extern "C" symbol from
// librocprof-trace-decoder.so. The RareToken layout is duplicated here on
// purpose; once the API is finalized in the public decoder header, this sample
// will switch to including that header instead.

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

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// ---------------------------------------------------------------------------
// Duplicated definitions from rocprof-trace-decoder/source/gfx9/rare_scan.h.
// Treated as a private contract with the .so we link against; replaced by an
// include when the public API is added.
// ---------------------------------------------------------------------------

namespace rare_scan
{
struct RareToken
{
    uint64_t contents;
    uint32_t type;
};

// gfx9 sqtt_token_type_t ids for the captured rare cluster (see gfx9token.h).
enum TokenType : uint32_t
{
    TOKEN_REG         = 2,
    TOKEN_REG_CS      = 5,
    TOKEN_EVENT       = 7,
    TOKEN_EVENT_CS    = 8,
    TOKEN_REG_CS_PRIV = 15,
};
}  // namespace rare_scan

// Linked from librocprof-trace-decoder.so (rare_scan_export.cpp).
extern "C" size_t
rocprof_trace_decoder_rare_scan_gfx9(const uint8_t* buf,
                                     size_t         size,
                                     void*          out,
                                     size_t         out_cap);

// ---------------------------------------------------------------------------
// PGM_LO / PGM_HI register addresses (gfx9 SQTT REG-token regaddr field).
// REG token encodes regaddr in bits 16..31 of the 64-bit token contents
// (see gfx9::Reg in source/gfx9/gfx9token.h). We keep a histogram of all REG
// addresses we observe and label the two PGM ones.
//
// COMPUTE_PGM_LO / COMPUTE_PGM_HI regaddr values in the SQTT REG token are
// 0xC and 0xD (offsets within the SH_REG block — the 0x2C00 base is implicit).
// ---------------------------------------------------------------------------
namespace
{
constexpr uint16_t COMPUTE_PGM_LO = 0xC;
constexpr uint16_t COMPUTE_PGM_HI = 0xD;

constexpr size_t SCAN_OUT_CAP = 1u << 23;  // 8M rare tokens / chunk
}  // namespace

namespace ScanState
{
// Triple-buffer mode delivers the 8-byte rocprof_trace_decoder_gfx9_header_t
// in its own callback before any payload chunk; we capture it but the rare
// scanner itself only needs post-header bytes.
std::atomic<bool>      have_header{false};
std::array<uint8_t, 8> header_bytes{};

// Stats. Atomics on the hot counters; the histogram is mutex-protected because
// even though rocprofiler-sdk's triple-buffer consumer thread is single-
// threaded today, that's an implementation detail.
std::atomic<uint64_t> chunks_processed{0};
std::atomic<uint64_t> bytes_processed{0};
std::atomic<uint64_t> rare_tokens_seen{0};
std::atomic<uint64_t> pgm_lo_count{0};
std::atomic<uint64_t> pgm_hi_count{0};

// Type distribution (16 token type IDs in gfx9 sqtt_token_type_t)
std::array<std::atomic<uint64_t>, 16> type_counts{};

// Trace-interrupt counters. END is benign (signals last record per SE);
// the BUFFER_FULL flags mean we lost data — GPU_FULL = drainer too slow,
// CPU_FULL = our callback too slow.
std::atomic<uint64_t> flag_end{0};
std::atomic<uint64_t> flag_gpu_full{0};
std::atomic<uint64_t> flag_cpu_full{0};

// Cumulative timing, in nanoseconds.
//   scanner_ns_total: only the rocprof_trace_decoder_rare_scan_gfx9 call.
//                     This is the AVX-512 / scalar scan throughput.
//   post_ns_total:    everything our callback does AFTER the scan returns
//                     (per-token switch + local map + mutex merge).
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
// Atomics above are trivially destructible so they survive; the map is not.
inline std::unordered_map<uint16_t, uint64_t>&
reg_addr_hist()
{
    static auto* h = new std::unordered_map<uint16_t, uint64_t>();
    return *h;
}

// PGM_LO / PGM_HI data-field histograms. Together LO|HI form the kernel
// program-start address; tracking the distinct data values written tells us
// how many distinct kernels SQTT saw (vs how often each was launched).
inline std::unordered_map<uint32_t, uint64_t>&
pgm_lo_data_hist()
{
    static auto* h = new std::unordered_map<uint32_t, uint64_t>();
    return *h;
}
inline std::unordered_map<uint32_t, uint64_t>&
pgm_hi_data_hist()
{
    static auto* h = new std::unordered_map<uint32_t, uint64_t>();
    return *h;
}

// Reusable scratch buffer for rare-scan output. Sized once at first use; we
// only ever touch it from the callback, so no synchronization needed.
thread_local std::vector<rare_scan::RareToken> tls_out;

void
scan_inline(const uint8_t* buf, size_t size)
{
    // Skip the standalone 8-byte header callback. rare_scan expects post-
    // header tokens.
    if(!have_header.load(std::memory_order_acquire) && size == sizeof(header_bytes))
    {
        std::memcpy(header_bytes.data(), buf, sizeof(header_bytes));
        have_header.store(true, std::memory_order_release);
        return;
    }

    if(tls_out.size() < SCAN_OUT_CAP) tls_out.resize(SCAN_OUT_CAP);

    auto t0 = std::chrono::steady_clock::now();
    size_t n = rocprof_trace_decoder_rare_scan_gfx9(buf, size, tls_out.data(), tls_out.size());
    auto t1 = std::chrono::steady_clock::now();

    chunks_processed.fetch_add(1, std::memory_order_relaxed);
    bytes_processed.fetch_add(size, std::memory_order_relaxed);
    rare_tokens_seen.fetch_add(n, std::memory_order_relaxed);

    // Aggregate into a small per-call buffer first, then merge under the
    // mutex once. Keeps the critical section tiny even at high token counts.
    // thread_local so the map's buckets are constructed once per consumer
    // thread and reused — clear() preserves the allocation.
    thread_local std::unordered_map<uint16_t, uint64_t> local;
    thread_local std::unordered_map<uint32_t, uint64_t> local_pgm_lo_data;
    thread_local std::unordered_map<uint32_t, uint64_t> local_pgm_hi_data;
    local.clear();
    local_pgm_lo_data.clear();
    local_pgm_hi_data.clear();
    uint64_t local_pgm_lo = 0;
    uint64_t local_pgm_hi = 0;

    for(size_t i = 0; i < n; i++)
    {
        const auto& tok = tls_out[i];
        if(tok.type < type_counts.size())
            type_counts[tok.type].fetch_add(1, std::memory_order_relaxed);
        uint16_t addr;
        uint32_t data;
        bool     is_regcs;
        switch(tok.type)
        {
            case rare_scan::TOKEN_REG:
                addr = static_cast<uint16_t>((tok.contents >> 16) & 0xFFFFu);
                data = static_cast<uint32_t>((tok.contents >> 32) & 0xFFFFFFFFu);
                is_regcs = false;
                break;
            case rare_scan::TOKEN_REG_CS:
            case rare_scan::TOKEN_REG_CS_PRIV:
                addr = static_cast<uint16_t>((tok.contents >> 9) & 0x7Fu);
                data = static_cast<uint32_t>((tok.contents >> 16) & 0xFFFFFFFFu);
                is_regcs = true;
                break;
            default:
                continue;  // EVENT / EVENT_CS — no register address
        }

        ++local[addr];
        // PGM_LO/HI only meaningful on REG_CS tokens — see
        // gfx9wave.cpp:499-506 in rocprof-trace-decoder, which only routes
        // REG_CS / REG_CS_PRIV through CSRegisterHandler::UpdateRegCS().
        if(!is_regcs) continue;
        if(addr == COMPUTE_PGM_LO)
        {
            ++local_pgm_lo;
            ++local_pgm_lo_data[data];
        }
        else if(addr == COMPUTE_PGM_HI)
        {
            ++local_pgm_hi;
            ++local_pgm_hi_data[data];
        }
    }

    if(local_pgm_lo) pgm_lo_count.fetch_add(local_pgm_lo, std::memory_order_relaxed);
    if(local_pgm_hi) pgm_hi_count.fetch_add(local_pgm_hi, std::memory_order_relaxed);

    if(!local.empty() || !local_pgm_lo_data.empty() || !local_pgm_hi_data.empty())
    {
        std::lock_guard<std::mutex> lock(hist_mu);
        auto& gh = reg_addr_hist();
        for(const auto& [addr, count] : local)
            gh[addr] += count;
        auto& glo = pgm_lo_data_hist();
        for(const auto& [d, c] : local_pgm_lo_data) glo[d] += c;
        auto& ghi = pgm_hi_data_hist();
        for(const auto& [d, c] : local_pgm_hi_data) ghi[d] += c;
    }

    auto t2 = std::chrono::steady_clock::now();
    scanner_ns_total.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(),
        std::memory_order_relaxed);
    post_ns_total.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count(),
        std::memory_order_relaxed);
    callback_ns_total.fetch_add(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t0).count(),
        std::memory_order_relaxed);
}
}  // namespace ScanState

namespace ATTClient
{
rocprof_trace_decoder_handle_t decoder{};
rocprofiler_context_id_t              agent_ctx{};
rocprofiler_context_id_t              tracing_ctx{};

constexpr uint64_t TARGET_CU       = 1;
constexpr uint64_t SHADER_MASK     = 0x1;
constexpr uint64_t GPU_BUFFER_SIZE_DEFAULT = 256ul << 20;  // 64MB triple-buffer slot

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
                     void*                                        se_data,
                     size_t                                       data_size,
                     rocprofiler_thread_trace_shader_data_flags_t flags,
                     rocprofiler_user_data_t /*userdata*/)
{
    if(flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_END)
        ScanState::flag_end.fetch_add(1, std::memory_order_relaxed);
    if(flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_GPU_BUFFER_FULL)
        ScanState::flag_gpu_full.fetch_add(1, std::memory_order_relaxed);
    if(flags & ROCPROFILER_THREAD_TRACE_SHADER_DATA_FLAGS_CPU_BUFFER_FULL)
        ScanState::flag_cpu_full.fetch_add(1, std::memory_order_relaxed);

    if(data_size == 0 || se_data == nullptr) return;
    ScanState::scan_inline(static_cast<const uint8_t*>(se_data), data_size);
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
    // NOTE: do NOT set NO_DETAIL=1 — it enables OCCUPANCY_MODE in aqlprofile,
    // which strips REG/INST/PERF tokens and emits only wave-occupancy EVENTs.
    // We need REG tokens for the PGM_LO/PGM_HI histogram, so leave detail on.
    parameters.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFERING_MODE,
                          {ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFERING_MODE_TRIPLE_BUFFER}});

    for(size_t idx = 0; idx < num_agents; idx++)
    {
        const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents[idx]);
        if(agent->type != ROCPROFILER_AGENT_TYPE_GPU) continue;

        rocprofiler_user_data_t user{};
        user.ptr = nullptr;
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
    // The decoder handle is required by the rocprofiler-sdk thread-trace path
    // even though we run our own scanner — without it, configure rejects
    // triple-buffer mode.
    rocprof_trace_decoder_create_handle(&decoder);

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

void
tool_fini(void* /*tool_data*/)
{
    rocprof_trace_decoder_destroy_handle(decoder);

    auto&  hist        = ScanState::reg_addr_hist();
    size_t total       = ScanState::rare_tokens_seen.load();
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
              << " rare_tokens=" << total
              << " unique_reg_addrs=" << hist.size() << "\n";
    std::cout << "[scan] PGM_LO (0x" << std::hex << COMPUTE_PGM_LO << std::dec
              << ") writes=" << ScanState::pgm_lo_count.load() << "\n";
    std::cout << "[scan] PGM_HI (0x" << std::hex << COMPUTE_PGM_HI << std::dec
              << ") writes=" << ScanState::pgm_hi_count.load() << "\n";

    // Throughput, broken into:
    //   scanner: rocprof_trace_decoder_rare_scan_gfx9 (the AVX-512 path)
    //   post:    our per-token switch + local map + global merge
    //   total:   sum, the full callback budget
    auto report = [bytes, chunks](const char* label, size_t ns) {
        if(ns == 0 || bytes == 0) return;
        double seconds  = ns / 1e9;
        double gb_per_s = (double) bytes / seconds / (1024.0 * 1024.0 * 1024.0);
        double ns_per_b = (double) ns / (double) bytes;
        double per_chk  = chunks ? ((double) ns / (double) chunks / 1000.0) : 0.0;
        std::printf(
            "[scan] %-8s  wall=%.3fs  throughput=%.2f GB/s  %.3f ns/byte"
            "  avg_chunk=%.1f us\n",
            label, seconds, gb_per_s, ns_per_b, per_chk);
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
    uint64_t end_n  = ScanState::flag_end.load();
    uint64_t gpu_n  = ScanState::flag_gpu_full.load();
    uint64_t cpu_n  = ScanState::flag_cpu_full.load();
    std::cout << "[scan] flags: END=" << end_n << " GPU_BUFFER_FULL=" << gpu_n
              << " CPU_BUFFER_FULL=" << cpu_n;
    if(gpu_n != 0 || cpu_n != 0)
        std::cout << "  *** TRACE INTERRUPTED — data was lost ***";
    std::cout << "\n";

    std::cout << "[scan] type counts:";
    for(size_t i = 0; i < ScanState::type_counts.size(); i++)
    {
        uint64_t c = ScanState::type_counts[i].load();
        if(c != 0) std::cout << " t" << i << "=" << c;
    }
    std::cout << "\n";

    // The decoder reconstructs the wave-start PC as
    //   ((HI<<32 | LO) << 8) & ((1<<48)-1)            [trace_parser.hpp:396]
    // i.e. the data field stores the address with its low 8 bits dropped.
    // Show the raw data and the implied byte address (data << 8) so two
    // values 1 unit apart are visibly 256 bytes apart in the actual PC space.
    auto print_data_hist = [](const char*                                  label,
                              const std::unordered_map<uint32_t, uint64_t>& h) {
        std::vector<std::pair<uint32_t, uint64_t>> v(h.begin(), h.end());
        std::sort(v.begin(), v.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        std::cout << "[scan] all " << label << " data values (" << v.size()
                  << " distinct):\n";
        for(size_t i = 0; i < v.size(); i++)
            std::printf("  data=0x%08x  pc_part=0x%010lx  count=%lu\n",
                        v[i].first,
                        static_cast<unsigned long>(v[i].first) << 8,
                        static_cast<unsigned long>(v[i].second));
    };
    print_data_hist("PGM_LO", ScanState::pgm_lo_data_hist());
    print_data_hist("PGM_HI", ScanState::pgm_hi_data_hist());
}
}  // namespace ATTClient

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/,
                      const char* /*runtime_version*/,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    if(priority > 0) return nullptr;
    id->name = "Thread Trace Rare-Scan Sample";
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &ATTClient::tool_init,
                                            &ATTClient::tool_fini,
                                            nullptr};
    return &cfg;
}
