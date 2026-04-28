// HSA hooks for kernel-trace and memory-copy-trace.
//
// kernel-trace: wrap hsa_amd_queue_intercept_create_fn so that every newly
// created intercept queue gets our packet writer registered. The writer
// inspects each AQL packet, attaches a per-dispatch completion signal, and on
// completion records dispatch start/end ticks via
// hsa_amd_profiling_get_dispatch_time_fn. Forwards packets unchanged via the
// supplied writer callback.
//
// memory-copy-trace: wrap hsa_amd_memory_async_copy_fn and
// hsa_amd_memory_async_copy_on_engine_fn. Each call gets a per-call signal
// chained behind whatever the caller passed; on completion we record
// start/end ticks and forward via the original signal.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <hsa.h>
#include <hsa_ext_amd.h>
#include <hsa_api_trace.h>
#include <amd_hsa_kernel_code.h>
#include <amd_hsa_queue.h>

#include "trace_buffer.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofv3_min {

extern bool g_kernel_trace_enabled;
extern bool g_memory_copy_trace_enabled;

// Saved originals (we keep separate snapshots of the sub-tables since the
// broker's HsaApiTable points to live tables we mutate in-place).
namespace hsa_orig {
    static HsaApiTable* table = nullptr; // for sub-table addressing only
    static AmdExtTable  amd_ext_snap{};
    static CoreApiTable core_snap{};
    static decltype(::hsa_amd_queue_intercept_create)*       queue_intercept_create   = nullptr;
    static decltype(::hsa_amd_queue_intercept_register)*     queue_intercept_register = nullptr;
    static decltype(::hsa_queue_create)*                     queue_create             = nullptr;
    static decltype(::hsa_queue_destroy)*                    queue_destroy            = nullptr;
    static decltype(::hsa_amd_memory_async_copy)*            mem_copy                 = nullptr;
    static decltype(::hsa_amd_memory_async_copy_on_engine)*  mem_copy_engine          = nullptr;
    static decltype(::hsa_amd_signal_create)*                amd_signal_create        = nullptr;
    static decltype(::hsa_signal_create)*                    signal_create            = nullptr;
    static decltype(::hsa_signal_destroy)*                   signal_destroy           = nullptr;
    static decltype(::hsa_amd_signal_async_handler)*         signal_async_handler     = nullptr;
    static decltype(::hsa_amd_profiling_get_dispatch_time)*  profiling_get_dispatch   = nullptr;
    static decltype(::hsa_amd_profiling_get_async_copy_time)* profiling_get_copy      = nullptr;
    static decltype(::hsa_amd_profiling_set_profiler_enabled)* profiling_set_enabled  = nullptr;
    static decltype(::hsa_amd_profiling_async_copy_enable)*    profiling_async_copy_enable = nullptr;
    static decltype(::hsa_signal_store_screlease)*           signal_store_screlease   = nullptr;
    static decltype(::hsa_signal_store_relaxed)*             signal_store_relaxed     = nullptr;
    static decltype(::hsa_iterate_agents)*                   iterate_agents           = nullptr;
    static decltype(::hsa_executable_freeze)*                executable_freeze        = nullptr;
    static decltype(::hsa_executable_iterate_agent_symbols)* iterate_agent_symbols   = nullptr;
    static decltype(::hsa_executable_symbol_get_info)*       symbol_get_info          = nullptr;
}

static std::atomic<uint64_t> g_dispatch_id{0};

// Forward decls (definitions live further down with the agent/kernel-name
// registries).
static std::string lookup_kernel_name(uint64_t kernel_object);
static bool name_matches_blit_copy(const std::string& n);

static const char* classify_direction(hsa_agent_t src, hsa_agent_t dst) {
    auto get_info = hsa_orig::core_snap.hsa_agent_get_info_fn;
    if (!get_info) return "UNKNOWN";
    hsa_device_type_t src_type = HSA_DEVICE_TYPE_CPU;
    hsa_device_type_t dst_type = HSA_DEVICE_TYPE_CPU;
    if (get_info(src, HSA_AGENT_INFO_DEVICE, &src_type) != HSA_STATUS_SUCCESS) return "UNKNOWN";
    if (get_info(dst, HSA_AGENT_INFO_DEVICE, &dst_type) != HSA_STATUS_SUCCESS) return "UNKNOWN";
    const bool src_gpu = (src_type == HSA_DEVICE_TYPE_GPU);
    const bool dst_gpu = (dst_type == HSA_DEVICE_TYPE_GPU);
    if (!src_gpu && dst_gpu) return "H2D";
    if (src_gpu && !dst_gpu) return "D2H";
    if (src_gpu && dst_gpu)  return "D2D";
    return "H2H";
}

struct CopyCtx {
    hsa_signal_t orig_signal; // 0 if caller didn't supply one
    hsa_signal_t our_signal;
    hsa_agent_t  src_agent;
    hsa_agent_t  dst_agent;
    uint64_t     bytes;
    uint64_t     correlation_id;
    const char*  direction;
};

static bool copy_signal_handler(hsa_signal_value_t /*value*/, void* arg) {
    auto* ctx = static_cast<CopyCtx*>(arg);
    hsa_amd_profiling_async_copy_time_t t{};
    if (hsa_orig::profiling_get_copy) {
        hsa_orig::profiling_get_copy(ctx->our_signal, &t);
    }
    CopyRow row{};
    row.kind = "MEMORY_COPY";
    row.direction = ctx->direction;
    row.source_agent_id = ctx->src_agent.handle;
    row.destination_agent_id = ctx->dst_agent.handle;
    row.correlation_id = ctx->correlation_id;
    row.start_ns = t.start;
    row.end_ns = t.end;
    row.bytes = ctx->bytes;
    TraceBuffers::instance().push_copy(std::move(row));

    // Chain: signal the caller's original signal, then destroy ours.
    if (ctx->orig_signal.handle && hsa_orig::core_snap.hsa_signal_subtract_screlease_fn) {
        hsa_orig::core_snap.hsa_signal_subtract_screlease_fn(ctx->orig_signal, 1);
    }
    if (hsa_orig::signal_destroy) hsa_orig::signal_destroy(ctx->our_signal);
    delete ctx;
    return false; // unregister handler
}

static hsa_status_t W_hsa_amd_memory_async_copy(
    void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
    size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal)
{
    if (!hsa_orig::amd_signal_create || !hsa_orig::signal_async_handler) {
        return hsa_orig::mem_copy(dst, dst_agent, src, src_agent, size,
                                  num_dep_signals, dep_signals, completion_signal);
    }
    auto* ctx = new CopyCtx{};
    ctx->orig_signal = completion_signal;
    ctx->src_agent = src_agent;
    ctx->dst_agent = dst_agent;
    ctx->bytes = size;
    ctx->correlation_id = TraceBuffers::instance().next_correlation_id();
    ctx->direction = classify_direction(src_agent, dst_agent);
    if (hsa_orig::amd_signal_create(1, 0, nullptr, 0, &ctx->our_signal) != HSA_STATUS_SUCCESS) {
        delete ctx;
        return hsa_orig::mem_copy(dst, dst_agent, src, src_agent, size,
                                  num_dep_signals, dep_signals, completion_signal);
    }
    hsa_status_t st = hsa_orig::mem_copy(dst, dst_agent, src, src_agent, size,
                                         num_dep_signals, dep_signals, ctx->our_signal);
    if (st != HSA_STATUS_SUCCESS) {
        if (hsa_orig::signal_destroy) hsa_orig::signal_destroy(ctx->our_signal);
        delete ctx;
        return st;
    }
    hsa_orig::signal_async_handler(ctx->our_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                   &copy_signal_handler, ctx);
    return st;
}

static hsa_status_t W_hsa_amd_memory_async_copy_on_engine(
    void* dst, hsa_agent_t dst_agent, const void* src, hsa_agent_t src_agent,
    size_t size, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal, hsa_amd_sdma_engine_id_t engine_id, bool force_copy_on_sdma)
{
    if (!hsa_orig::amd_signal_create || !hsa_orig::signal_async_handler) {
        return hsa_orig::mem_copy_engine(dst, dst_agent, src, src_agent, size,
                                         num_dep_signals, dep_signals, completion_signal,
                                         engine_id, force_copy_on_sdma);
    }
    auto* ctx = new CopyCtx{};
    ctx->orig_signal = completion_signal;
    ctx->src_agent = src_agent;
    ctx->dst_agent = dst_agent;
    ctx->bytes = size;
    ctx->correlation_id = TraceBuffers::instance().next_correlation_id();
    ctx->direction = classify_direction(src_agent, dst_agent);
    if (hsa_orig::amd_signal_create(1, 0, nullptr, 0, &ctx->our_signal) != HSA_STATUS_SUCCESS) {
        delete ctx;
        return hsa_orig::mem_copy_engine(dst, dst_agent, src, src_agent, size,
                                         num_dep_signals, dep_signals, completion_signal,
                                         engine_id, force_copy_on_sdma);
    }
    hsa_status_t st = hsa_orig::mem_copy_engine(dst, dst_agent, src, src_agent, size,
                                                num_dep_signals, dep_signals, ctx->our_signal,
                                                engine_id, force_copy_on_sdma);
    if (st != HSA_STATUS_SUCCESS) {
        if (hsa_orig::signal_destroy) hsa_orig::signal_destroy(ctx->our_signal);
        delete ctx;
        return st;
    }
    hsa_orig::signal_async_handler(ctx->our_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                   &copy_signal_handler, ctx);
    return st;
}

// --- kernel-trace ---

struct KernelCtx {
    hsa_signal_t our_signal;
    hsa_signal_t orig_signal; // 0 if caller's packet had no completion signal
    uint64_t     queue_id;
    uint64_t     agent_id;
    uint32_t     thread_id;
    uint64_t     dispatch_id;
    uint64_t     kernel_id;        // kernel_object handle
    uint64_t     correlation_id;
    uint32_t     private_segment_size;
    uint32_t     group_segment_size;
    uint16_t     workgroup_size_x;
    uint16_t     workgroup_size_y;
    uint16_t     workgroup_size_z;
    uint32_t     grid_size_x;
    uint32_t     grid_size_y;
    uint32_t     grid_size_z;
    std::string  kernel_name;
    bool         is_blit_copy;
};

static bool kernel_signal_handler(hsa_signal_value_t /*value*/, void* arg) {
    auto* ctx = static_cast<KernelCtx*>(arg);
    hsa_amd_profiling_dispatch_time_t t{};
    if (hsa_orig::profiling_get_dispatch) {
        hsa_agent_t a; a.handle = ctx->agent_id;
        hsa_orig::profiling_get_dispatch(a, ctx->our_signal, &t);
    }
    if (ctx->is_blit_copy) {
        CopyRow row{};
        row.kind = "MEMORY_COPY";
        row.direction = "BLIT_KERNEL";
        row.source_agent_id = ctx->agent_id;
        row.destination_agent_id = ctx->agent_id;
        row.correlation_id = ctx->correlation_id;
        row.start_ns = t.start;
        row.end_ns = t.end;
        row.bytes = 0;
        TraceBuffers::instance().push_copy(std::move(row));
    } else {
        KernelRow row{};
        row.kind = "KERNEL_DISPATCH";
        row.agent_id = ctx->agent_id;
        row.queue_id = ctx->queue_id;
        row.thread_id = ctx->thread_id;
        row.dispatch_id = ctx->dispatch_id;
        row.kernel_id = ctx->kernel_id;
        row.kernel_name = ctx->kernel_name;
        row.correlation_id = ctx->correlation_id;
        row.start_ns = t.start;
        row.end_ns = t.end;
        row.private_segment_size = ctx->private_segment_size;
        row.group_segment_size = ctx->group_segment_size;
        row.workgroup_size_x = ctx->workgroup_size_x;
        row.workgroup_size_y = ctx->workgroup_size_y;
        row.workgroup_size_z = ctx->workgroup_size_z;
        row.grid_size_x = ctx->grid_size_x;
        row.grid_size_y = ctx->grid_size_y;
        row.grid_size_z = ctx->grid_size_z;
        TraceBuffers::instance().push_kernel(std::move(row));
    }

    // Chain: signal the caller's original completion signal so apps that
    // wait on their own dispatch signal don't hang. Then destroy ours.
    if (ctx->orig_signal.handle && hsa_orig::core_snap.hsa_signal_subtract_screlease_fn) {
        hsa_orig::core_snap.hsa_signal_subtract_screlease_fn(ctx->orig_signal, 1);
    }
    if (hsa_orig::signal_destroy) hsa_orig::signal_destroy(ctx->our_signal);
    delete ctx;
    return false;
}

struct QueueState {
    hsa_queue_t* queue;
    uint64_t     queue_id;
    hsa_agent_t  agent;
};

static void our_packet_writer(const void* pkts, uint64_t pkt_count,
                              uint64_t /*user_pkt_index*/, void* data,
                              hsa_amd_queue_intercept_packet_writer writer)
{
    auto* qs = static_cast<QueueState*>(data);
    const auto* aql = static_cast<const hsa_kernel_dispatch_packet_t*>(pkts);

    for (uint64_t i = 0; i < pkt_count; ++i) {
        uint8_t type = (aql[i].header >> HSA_PACKET_HEADER_TYPE) & ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u);
        if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH && hsa_orig::amd_signal_create
            && hsa_orig::signal_async_handler)
        {
            auto* ctx = new KernelCtx{};
            ctx->queue_id = qs->queue_id;
            ctx->agent_id = qs->agent.handle;
            ctx->thread_id = GetCurrentThreadId();
            ctx->dispatch_id = ++g_dispatch_id;
            ctx->kernel_id = aql[i].kernel_object;
            ctx->correlation_id = TraceBuffers::instance().next_correlation_id();
            ctx->private_segment_size = aql[i].private_segment_size;
            ctx->group_segment_size = aql[i].group_segment_size;
            ctx->workgroup_size_x = aql[i].workgroup_size_x;
            ctx->workgroup_size_y = aql[i].workgroup_size_y;
            ctx->workgroup_size_z = aql[i].workgroup_size_z;
            ctx->grid_size_x = aql[i].grid_size_x;
            ctx->grid_size_y = aql[i].grid_size_y;
            ctx->grid_size_z = aql[i].grid_size_z;
            ctx->kernel_name  = lookup_kernel_name(ctx->kernel_id);
            ctx->is_blit_copy = !ctx->kernel_name.empty() && name_matches_blit_copy(ctx->kernel_name);
            if (hsa_orig::amd_signal_create(1, 0, nullptr, 0, &ctx->our_signal) == HSA_STATUS_SUCCESS) {
                // Capture the caller's original completion signal so we can
                // signal it back from our handler; replace the packet's slot
                // with ours to get accurate dispatch-time profiling.
                auto pkt_copy = aql[i];
                ctx->orig_signal = pkt_copy.completion_signal;
                pkt_copy.completion_signal = ctx->our_signal;
                hsa_orig::signal_async_handler(ctx->our_signal,
                                               HSA_SIGNAL_CONDITION_LT, 1,
                                               &kernel_signal_handler, ctx);
                writer(&pkt_copy, 1);
                continue;
            } else {
                delete ctx;
            }
        }
        // Non-dispatch or fallback: forward unchanged.
        writer(&aql[i], 1);
    }
}

static std::atomic<uint64_t> g_diag_writer_calls{0};
static std::atomic<uint64_t> g_diag_pkts_forwarded{0};

// Dump a 64-byte AQL packet: structured kernel-dispatch fields (assuming the
// packet is a KERNEL_DISPATCH; fields are still valid bytes for any other type)
// followed by the raw 64 bytes in hex. Used to compare what rocclr produces
// across the InterceptQueue path vs the regular doorbell-hook path.
static void dump_aql_packet(const char* tag, const void* pkt) {
    const auto* d = static_cast<const hsa_kernel_dispatch_packet_t*>(pkt);
    const uint8_t* b = static_cast<const uint8_t*>(pkt);
    std::fprintf(stderr,
        "[%s] hdr=0x%04x setup=0x%04x wg=%ux%ux%u grid=%ux%ux%u "
        "priv=%u group=%u kobj=0x%llx kargs=%p comp_sig=0x%llx\n",
        tag,
        (unsigned)d->header, (unsigned)d->setup,
        (unsigned)d->workgroup_size_x, (unsigned)d->workgroup_size_y, (unsigned)d->workgroup_size_z,
        (unsigned)d->grid_size_x, (unsigned)d->grid_size_y, (unsigned)d->grid_size_z,
        (unsigned)d->private_segment_size, (unsigned)d->group_segment_size,
        (unsigned long long)d->kernel_object,
        d->kernarg_address,
        (unsigned long long)d->completion_signal.handle);
    for (int row = 0; row < 4; ++row) {
        std::fprintf(stderr, "[%s] hex %02x:", tag, row * 16);
        for (int col = 0; col < 16; ++col) {
            std::fprintf(stderr, " %02x", b[row * 16 + col]);
        }
        std::fprintf(stderr, "\n");
    }
}

static void diag_packet_writer(const void* pkts, uint64_t pkt_count,
                               uint64_t user_pkt_index, void* /*data*/,
                               hsa_amd_queue_intercept_packet_writer writer)
{
    const uint64_t call_n = g_diag_writer_calls.fetch_add(1) + 1;
    std::fprintf(stderr, "[diag-writer] call#%llu pkts=%llu user_idx=%llu\n",
                 (unsigned long long)call_n,
                 (unsigned long long)pkt_count,
                 (unsigned long long)user_pkt_index);
    if (pkt_count >= 1) {
        dump_aql_packet("intercept-in", pkts);
    }
    // ROCPROFV3_DIAG_NO_FORWARD=1: skip writer() to test whether DXG's 0x100d
    // is triggered by something OTHER than the bytes we forward (e.g., the
    // wrapped queue's existence, or the InterceptQueue's async-doorbell
    // signaling the wrapped queue independently).
    const bool no_forward = []() {
        const char* v = std::getenv("ROCPROFV3_DIAG_NO_FORWARD");
        return v && v[0] == '1';
    }();
    if (no_forward) {
        std::fprintf(stderr, "[diag-writer] call#%llu NO_FORWARD: skipping writer()\n",
                     (unsigned long long)call_n);
        return;
    }
    if (writer) {
        writer(pkts, pkt_count);
        g_diag_pkts_forwarded.fetch_add(pkt_count);
        std::fprintf(stderr, "[diag-writer] call#%llu forwarded ok\n",
                     (unsigned long long)call_n);
    } else {
        std::fprintf(stderr, "[diag-writer] call#%llu writer cb is NULL!\n",
                     (unsigned long long)call_n);
    }
}

static hsa_status_t W_hsa_queue_create_intercept(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue)
{
    auto fn = hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_create_fn;
    if (!fn) {
        return hsa_orig::queue_create(agent, size, type, callback, data,
                                      private_segment_size, group_segment_size, queue);
    }

    hsa_status_t st = fn(agent, size, type, callback, data,
                        private_segment_size, group_segment_size, queue);
    if (st != HSA_STATUS_SUCCESS || !queue || !*queue) return st;

    auto reg = hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_register_fn;
    if (!reg) return st;

    auto* qs = new QueueState{*queue, (*queue)->id, agent};
    reg(*queue, &our_packet_writer, qs);

    if (hsa_orig::profiling_set_enabled) {
        hsa_orig::profiling_set_enabled(*queue, 1);
    }

    return st;
}

static hsa_status_t W_hsa_amd_queue_intercept_create(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue)
{
    hsa_status_t st = hsa_orig::queue_intercept_create(agent_handle, size, type, callback, data,
                                                       private_segment_size, group_segment_size,
                                                       queue);
    if (st == HSA_STATUS_SUCCESS && queue && *queue && hsa_orig::queue_intercept_register) {
        auto* qs = new QueueState{*queue, (*queue)->id, agent_handle};
        hsa_orig::queue_intercept_register(*queue, &our_packet_writer, qs);
    }
    return st;
}

// Windows-DXG path: do NOT substitute hsa_queue_create with intercept-create.
// The WSL thunk's AqlToPm4Thread cannot pump the intercept extension's soft
// queue and traps with HSA_STATUS_ERROR_EXCEPTION (status 0x1016) on the
// first packet. Instead, record the queue's doorbell signal and let the
// regular queue stand. We intercept dispatches via the doorbell store path
// (W_hsa_signal_store_screlease) below.

struct DoorbellInfo {
    hsa_queue_t* queue;
    hsa_agent_t  agent;
    uint64_t     queue_id;
    std::mutex   mu;
    uint64_t     next_to_process; // next AQL slot index we have not yet seen
};

static std::mutex g_doorbell_map_mu;
static std::unordered_map<uint64_t, std::unique_ptr<DoorbellInfo>> g_doorbells;

static void register_doorbell(hsa_queue_t* q, hsa_agent_t agent) {
    if (!q) return;
    const uint64_t handle = q->doorbell_signal.handle;
    if (!handle) return;
    auto info = std::make_unique<DoorbellInfo>();
    info->queue = q;
    info->agent = agent;
    info->queue_id = q->id;
    info->next_to_process = 0;
    std::lock_guard<std::mutex> lk(g_doorbell_map_mu);
    g_doorbells[handle] = std::move(info);
}

static hsa_status_t W_hsa_queue_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue)
{
    hsa_status_t st = hsa_orig::queue_create(agent, size, type, callback, data,
                                             private_segment_size, group_segment_size, queue);
    if (st == HSA_STATUS_SUCCESS && queue && *queue) {
        register_doorbell(*queue, agent);
        if (hsa_orig::profiling_set_enabled) {
            hsa_orig::profiling_set_enabled(*queue, 1);
        }
    }
    return st;
}

static hsa_status_t W_hsa_queue_destroy(hsa_queue_t* queue) {
    if (queue) {
        std::lock_guard<std::mutex> lk(g_doorbell_map_mu);
        g_doorbells.erase(queue->doorbell_signal.handle);
    }
    return hsa_orig::queue_destroy(queue);
}

// --- Agent + kernel-name registries ---

static std::mutex g_agents_mu;
static std::vector<hsa_agent_t> g_gpu_agents;

static std::mutex g_kernel_names_mu;
static std::unordered_map<uint64_t, std::string> g_kernel_names; // kernel_object -> mangled name

static bool name_matches_blit_copy(const std::string& n) {
    // rocclr's compute-shader copy kernels emitted by KernelBlitManager.
    // Match the prefixes; covers __amd_copyBuffer, __amd_copyBufferAligned,
    // __amd_copyBufferRect, __amd_copyImage*, __amd_fillBuffer, etc.
    static const char* prefixes[] = {
        "__amd_rocclr_copy", "__amd_rocclr_fill",
        "__amd_copy",        "__amd_fill"
    };
    for (auto* p : prefixes) {
        const size_t plen = std::strlen(p);
        if (n.size() >= plen && std::memcmp(n.data(), p, plen) == 0) return true;
    }
    return false;
}

static std::string lookup_kernel_name(uint64_t kernel_object) {
    std::lock_guard<std::mutex> lk(g_kernel_names_mu);
    auto it = g_kernel_names.find(kernel_object);
    if (it == g_kernel_names.end()) return std::string();
    return it->second;
}

static hsa_status_t agent_capture_cb(hsa_agent_t agent, void* /*data*/) {
    auto get_info = hsa_orig::core_snap.hsa_agent_get_info_fn;
    if (!get_info) return HSA_STATUS_SUCCESS;
    hsa_device_type_t dt = HSA_DEVICE_TYPE_CPU;
    if (get_info(agent, HSA_AGENT_INFO_DEVICE, &dt) != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;
    if (dt == HSA_DEVICE_TYPE_GPU) {
        std::lock_guard<std::mutex> lk(g_agents_mu);
        for (auto& a : g_gpu_agents) if (a.handle == agent.handle) return HSA_STATUS_SUCCESS;
        g_gpu_agents.push_back(agent);
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t W_hsa_iterate_agents(hsa_status_t (*callback)(hsa_agent_t, void*), void* data) {
    // Forward to the original first; then sweep our own capture pass to make
    // sure we have the agent list even if the runtime never iterates again.
    hsa_status_t st = hsa_orig::iterate_agents(callback, data);
    if (hsa_orig::iterate_agents) {
        hsa_orig::iterate_agents(&agent_capture_cb, nullptr);
    }
    return st;
}

static hsa_status_t symbol_capture_cb(hsa_executable_t /*exe*/, hsa_agent_t /*agent*/,
                                      hsa_executable_symbol_t sym, void* /*data*/) {
    auto get_info = hsa_orig::symbol_get_info;
    if (!get_info) return HSA_STATUS_SUCCESS;
    hsa_symbol_kind_t kind = HSA_SYMBOL_KIND_VARIABLE;
    if (get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (kind != HSA_SYMBOL_KIND_KERNEL) return HSA_STATUS_SUCCESS;

    uint64_t kernel_object = 0;
    if (get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    if (!kernel_object) return HSA_STATUS_SUCCESS;

    uint32_t name_len = 0;
    if (get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_len) != HSA_STATUS_SUCCESS)
        return HSA_STATUS_SUCCESS;
    std::string name(name_len, '\0');
    if (name_len > 0 &&
        get_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_NAME, name.data()) != HSA_STATUS_SUCCESS) {
        return HSA_STATUS_SUCCESS;
    }
    {
        std::lock_guard<std::mutex> lk(g_kernel_names_mu);
        g_kernel_names[kernel_object] = std::move(name);
    }
    return HSA_STATUS_SUCCESS;
}

static hsa_status_t W_hsa_executable_freeze(hsa_executable_t exe, const char* options) {
    hsa_status_t st = hsa_orig::executable_freeze(exe, options);
    if (st != HSA_STATUS_SUCCESS) return st;
    if (!hsa_orig::iterate_agent_symbols) return st;
    std::vector<hsa_agent_t> agents;
    {
        std::lock_guard<std::mutex> lk(g_agents_mu);
        agents = g_gpu_agents;
    }
    for (auto& a : agents) {
        hsa_orig::iterate_agent_symbols(exe, a, &symbol_capture_cb, nullptr);
    }
    return st;
}

// Inspect AQL packets in the ring from `from` (inclusive) to `to` (inclusive).
// On KERNEL_DISPATCH packets, swap completion_signal in-place with our chained
// signal and register an async handler that emits a KernelRow on completion.
// MUST be called before forwarding the doorbell store so the GPU/AqlToPm4Thread
// reads the modified packets.
static void process_doorbell_range(DoorbellInfo* info, uint64_t from, uint64_t to) {
    if (!hsa_orig::amd_signal_create || !hsa_orig::signal_async_handler) return;
    if (!info->queue || !info->queue->base_address || info->queue->size == 0) return;
    auto* ring = static_cast<hsa_kernel_dispatch_packet_t*>(info->queue->base_address);
    const uint64_t mask = static_cast<uint64_t>(info->queue->size) - 1;
    for (uint64_t idx = from; idx <= to; ++idx) {
        auto* slot = &ring[idx & mask];
        const uint8_t hdr = static_cast<uint8_t>(slot->header & 0xFFu);
        const uint8_t type = (hdr >> HSA_PACKET_HEADER_TYPE) &
                             ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u);
        if (type != HSA_PACKET_TYPE_KERNEL_DISPATCH) continue;

        dump_aql_packet("doorbell-orig", slot);

        auto* ctx = new KernelCtx{};
        ctx->queue_id            = info->queue_id;
        ctx->agent_id            = info->agent.handle;
        ctx->thread_id           = GetCurrentThreadId();
        ctx->dispatch_id         = ++g_dispatch_id;
        ctx->kernel_id           = slot->kernel_object;
        ctx->correlation_id      = TraceBuffers::instance().next_correlation_id();
        ctx->private_segment_size = slot->private_segment_size;
        ctx->group_segment_size   = slot->group_segment_size;
        ctx->workgroup_size_x    = slot->workgroup_size_x;
        ctx->workgroup_size_y    = slot->workgroup_size_y;
        ctx->workgroup_size_z    = slot->workgroup_size_z;
        ctx->grid_size_x         = slot->grid_size_x;
        ctx->grid_size_y         = slot->grid_size_y;
        ctx->grid_size_z         = slot->grid_size_z;

        if (hsa_orig::amd_signal_create(1, 0, nullptr, 0, &ctx->our_signal) != HSA_STATUS_SUCCESS) {
            delete ctx;
            continue;
        }
        ctx->kernel_name  = lookup_kernel_name(ctx->kernel_id);
        ctx->is_blit_copy = !ctx->kernel_name.empty() && name_matches_blit_copy(ctx->kernel_name);
        ctx->orig_signal = slot->completion_signal;
        // In-place swap: the GPU has not yet been told about this packet
        // (we have not forwarded the doorbell store).
        slot->completion_signal = ctx->our_signal;
        hsa_orig::signal_async_handler(ctx->our_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                       &kernel_signal_handler, ctx);
    }
}

static void W_hsa_signal_store_screlease(hsa_signal_t signal, hsa_signal_value_t value) {
    DoorbellInfo* info = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_doorbell_map_mu);
        auto it = g_doorbells.find(signal.handle);
        if (it != g_doorbells.end()) info = it->second.get();
    }
    if (info && value >= 0) {
        std::lock_guard<std::mutex> lk(info->mu);
        const uint64_t to = static_cast<uint64_t>(value);
        if (to >= info->next_to_process) {
            process_doorbell_range(info, info->next_to_process, to);
            info->next_to_process = to + 1;
        }
    }
    hsa_orig::signal_store_screlease(signal, value);
}

static void W_hsa_signal_store_relaxed(hsa_signal_t signal, hsa_signal_value_t value) {
    DoorbellInfo* info = nullptr;
    {
        std::lock_guard<std::mutex> lk(g_doorbell_map_mu);
        auto it = g_doorbells.find(signal.handle);
        if (it != g_doorbells.end()) info = it->second.get();
    }
    if (info && value >= 0) {
        std::lock_guard<std::mutex> lk(info->mu);
        const uint64_t to = static_cast<uint64_t>(value);
        if (to >= info->next_to_process) {
            process_doorbell_range(info, info->next_to_process, to);
            info->next_to_process = to + 1;
        }
    }
    hsa_orig::signal_store_relaxed(signal, value);
}

void install_hsa_wrappers(HsaApiTable* table) {
    if (!table) return;
    hsa_orig::table = table;

    // Snapshot sub-tables BEFORE mutating, so wrappers forward via originals.
    if (table->amd_ext_) {
        hsa_orig::amd_ext_snap          = *table->amd_ext_;
        hsa_orig::amd_signal_create     = hsa_orig::amd_ext_snap.hsa_amd_signal_create_fn;
        hsa_orig::signal_async_handler  = hsa_orig::amd_ext_snap.hsa_amd_signal_async_handler_fn;
        hsa_orig::profiling_get_dispatch = hsa_orig::amd_ext_snap.hsa_amd_profiling_get_dispatch_time_fn;
        hsa_orig::profiling_get_copy     = hsa_orig::amd_ext_snap.hsa_amd_profiling_get_async_copy_time_fn;
        hsa_orig::profiling_set_enabled  = hsa_orig::amd_ext_snap.hsa_amd_profiling_set_profiler_enabled_fn;
        hsa_orig::profiling_async_copy_enable = hsa_orig::amd_ext_snap.hsa_amd_profiling_async_copy_enable_fn;
        // Enable async-copy timing globally so any SDMA path we still wrap
        // produces non-zero start/end timestamps.
        if (hsa_orig::profiling_async_copy_enable) {
            hsa_orig::profiling_async_copy_enable(1);
        }
        std::fprintf(stderr,
            "[rocprofv3-min PREFLIGHT] amd_ext slots: "
            "intercept_create=%p intercept_register=%p "
            "signal_create=%p signal_async_handler=%p "
            "profiling_get_dispatch=%p profiling_set_enabled=%p\n",
            (void*)hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_create_fn,
            (void*)hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_register_fn,
            (void*)hsa_orig::amd_ext_snap.hsa_amd_signal_create_fn,
            (void*)hsa_orig::amd_ext_snap.hsa_amd_signal_async_handler_fn,
            (void*)hsa_orig::amd_ext_snap.hsa_amd_profiling_get_dispatch_time_fn,
            (void*)hsa_orig::amd_ext_snap.hsa_amd_profiling_set_profiler_enabled_fn);
    }
    if (table->core_) {
        hsa_orig::core_snap            = *table->core_;
        hsa_orig::signal_create        = hsa_orig::core_snap.hsa_signal_create_fn;
        hsa_orig::signal_destroy       = hsa_orig::core_snap.hsa_signal_destroy_fn;
        hsa_orig::signal_store_screlease = hsa_orig::core_snap.hsa_signal_store_screlease_fn;
        hsa_orig::signal_store_relaxed   = hsa_orig::core_snap.hsa_signal_store_relaxed_fn;
        hsa_orig::queue_create         = hsa_orig::core_snap.hsa_queue_create_fn;
        hsa_orig::queue_destroy        = hsa_orig::core_snap.hsa_queue_destroy_fn;
        hsa_orig::iterate_agents       = hsa_orig::core_snap.hsa_iterate_agents_fn;
        hsa_orig::executable_freeze    = hsa_orig::core_snap.hsa_executable_freeze_fn;
        hsa_orig::iterate_agent_symbols = hsa_orig::core_snap.hsa_executable_iterate_agent_symbols_fn;
        hsa_orig::symbol_get_info      = hsa_orig::core_snap.hsa_executable_symbol_get_info_fn;
        std::fprintf(stderr,
            "[rocprofv3-min PREFLIGHT] core slots: "
            "queue_create=%p queue_destroy=%p\n",
            (void*)hsa_orig::core_snap.hsa_queue_create_fn,
            (void*)hsa_orig::core_snap.hsa_queue_destroy_fn);
    }

    if (g_kernel_trace_enabled && table->amd_ext_) {
        hsa_orig::queue_intercept_create   = hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_create_fn;
        hsa_orig::queue_intercept_register = hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_register_fn;
        if (hsa_orig::queue_intercept_create) {
            table->amd_ext_->hsa_amd_queue_intercept_create_fn = &W_hsa_amd_queue_intercept_create;
        }
    }

    // Mode toggle: ROCPROFV3_USE_INTERCEPT=1 → install W_hsa_queue_create_intercept
    // (substitute intercept_create for queue_create, register diag_packet_writer).
    // Otherwise → install regular W_hsa_queue_create + doorbell-store hooks.
    // Both modes dump 64-byte AQL packets so the two captures can be diffed.
    const bool use_intercept = []() {
        const char* v = std::getenv("ROCPROFV3_USE_INTERCEPT");
        return v && v[0] == '1';
    }();
    std::fprintf(stderr, "[rocprofv3-min] kernel-trace mode: %s\n",
                 use_intercept ? "INTERCEPT" : "DOORBELL-HOOK");

    if (g_kernel_trace_enabled && table->core_) {
        if (use_intercept) {
            // Intercept path: route hsa_queue_create through our DIAG wrapper
            // which calls hsa_amd_queue_intercept_create + intercept_register.
            if (hsa_orig::queue_create &&
                hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_create_fn &&
                hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_register_fn) {
                table->core_->hsa_queue_create_fn = &W_hsa_queue_create_intercept;
            }
            if (hsa_orig::queue_destroy) {
                table->core_->hsa_queue_destroy_fn = &W_hsa_queue_destroy;
            }
        } else {
            // Doorbell-hook path: keep regular queue_create, record every
            // queue's doorbell, intercept signal_store stores to it.
            if (hsa_orig::queue_create) {
                table->core_->hsa_queue_create_fn = &W_hsa_queue_create;
            }
            if (hsa_orig::queue_destroy) {
                table->core_->hsa_queue_destroy_fn = &W_hsa_queue_destroy;
            }
            if (hsa_orig::signal_store_screlease) {
                table->core_->hsa_signal_store_screlease_fn = &W_hsa_signal_store_screlease;
            }
            if (hsa_orig::signal_store_relaxed) {
                table->core_->hsa_signal_store_relaxed_fn = &W_hsa_signal_store_relaxed;
            }
        }

        // Kernel-name resolution (same in both modes).
        if (hsa_orig::iterate_agents) {
            table->core_->hsa_iterate_agents_fn = &W_hsa_iterate_agents;
        }
        if (hsa_orig::executable_freeze && hsa_orig::iterate_agent_symbols &&
            hsa_orig::symbol_get_info) {
            table->core_->hsa_executable_freeze_fn = &W_hsa_executable_freeze;
        }
    }

    if (g_memory_copy_trace_enabled && table->amd_ext_) {
        hsa_orig::mem_copy        = hsa_orig::amd_ext_snap.hsa_amd_memory_async_copy_fn;
        hsa_orig::mem_copy_engine = hsa_orig::amd_ext_snap.hsa_amd_memory_async_copy_on_engine_fn;
        if (hsa_orig::mem_copy) {
            table->amd_ext_->hsa_amd_memory_async_copy_fn = &W_hsa_amd_memory_async_copy;
        }
        if (hsa_orig::mem_copy_engine) {
            table->amd_ext_->hsa_amd_memory_async_copy_on_engine_fn = &W_hsa_amd_memory_async_copy_on_engine;
        }
    }
}

} // namespace rocprofv3_min
