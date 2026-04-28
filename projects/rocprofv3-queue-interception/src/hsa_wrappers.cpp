// Queue-interception kernel-trace.
//
// Substitutes core_->hsa_queue_create_fn so that every newly created HW queue
// is built via hsa_amd_queue_intercept_create + intercept_register. The
// registered packet writer attaches a per-dispatch completion signal, fires
// hsa_amd_profiling_get_dispatch_time on completion, records a KernelRow,
// then chains the caller's original completion signal. Forwards packets
// unchanged via the supplied writer callback.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <hsa.h>
#include <hsa_ext_amd.h>
#include <hsa_api_trace.h>

#include "trace_buffer.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofv3_min {

extern bool g_kernel_trace_enabled;

namespace hsa_orig {
    static AmdExtTable  amd_ext_snap{};
    static CoreApiTable core_snap{};
    static decltype(::hsa_amd_queue_intercept_create)*       queue_intercept_create   = nullptr;
    static decltype(::hsa_amd_queue_intercept_register)*     queue_intercept_register = nullptr;
    static decltype(::hsa_queue_create)*                     queue_create             = nullptr;
    static decltype(::hsa_queue_destroy)*                    queue_destroy            = nullptr;
    static decltype(::hsa_amd_signal_create)*                amd_signal_create        = nullptr;
    static decltype(::hsa_signal_destroy)*                   signal_destroy           = nullptr;
    static decltype(::hsa_amd_signal_async_handler)*         signal_async_handler     = nullptr;
    static decltype(::hsa_amd_profiling_get_dispatch_time)*  profiling_get_dispatch   = nullptr;
    static decltype(::hsa_amd_profiling_set_profiler_enabled)* profiling_set_enabled  = nullptr;
    static decltype(::hsa_iterate_agents)*                   iterate_agents           = nullptr;
    static decltype(::hsa_executable_freeze)*                executable_freeze        = nullptr;
    static decltype(::hsa_executable_iterate_agent_symbols)* iterate_agent_symbols    = nullptr;
    static decltype(::hsa_executable_symbol_get_info)*       symbol_get_info          = nullptr;
}

static std::atomic<uint64_t> g_dispatch_id{0};

static std::mutex g_kernel_names_mu;
static std::unordered_map<uint64_t, std::string> g_kernel_names;

static std::mutex g_agents_mu;
static std::vector<hsa_agent_t> g_gpu_agents;

static std::string lookup_kernel_name(uint64_t kernel_object) {
    std::lock_guard<std::mutex> lk(g_kernel_names_mu);
    auto it = g_kernel_names.find(kernel_object);
    return it == g_kernel_names.end() ? std::string() : it->second;
}

struct KernelCtx {
    hsa_signal_t our_signal;
    hsa_signal_t orig_signal;
    uint64_t     queue_id;
    uint64_t     agent_id;
    uint32_t     thread_id;
    uint64_t     dispatch_id;
    uint64_t     kernel_id;
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
};

static bool kernel_signal_handler(hsa_signal_value_t /*value*/, void* arg) {
    auto* ctx = static_cast<KernelCtx*>(arg);
    hsa_amd_profiling_dispatch_time_t t{};
    if (hsa_orig::profiling_get_dispatch) {
        hsa_agent_t a; a.handle = ctx->agent_id;
        hsa_orig::profiling_get_dispatch(a, ctx->our_signal, &t);
    }
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
        uint8_t type = (aql[i].header >> HSA_PACKET_HEADER_TYPE) &
                       ((1u << HSA_PACKET_HEADER_WIDTH_TYPE) - 1u);
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
            ctx->kernel_name = lookup_kernel_name(ctx->kernel_id);
            if (hsa_orig::amd_signal_create(1, 0, nullptr, 0, &ctx->our_signal) == HSA_STATUS_SUCCESS) {
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
        writer(&aql[i], 1);
    }
}

// Substitute hsa_queue_create with intercept_create + intercept_register so
// every queue the runtime/HIP creates becomes a soft intercept queue.
static hsa_status_t W_hsa_queue_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue)
{
    auto fn = hsa_orig::queue_intercept_create;
    if (!fn) {
        return hsa_orig::queue_create(agent, size, type, callback, data,
                                      private_segment_size, group_segment_size, queue);
    }

    hsa_status_t st = fn(agent, size, type, callback, data,
                         private_segment_size, group_segment_size, queue);
    if (st != HSA_STATUS_SUCCESS || !queue || !*queue) return st;

    if (hsa_orig::queue_intercept_register) {
        auto* qs = new QueueState{*queue, (*queue)->id, agent};
        hsa_orig::queue_intercept_register(*queue, &our_packet_writer, qs);
    }
    if (hsa_orig::profiling_set_enabled) {
        hsa_orig::profiling_set_enabled(*queue, 1);
    }
    return st;
}

static hsa_status_t W_hsa_queue_destroy(hsa_queue_t* queue) {
    return hsa_orig::queue_destroy(queue);
}

// Kernel-name resolution.
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

void install_hsa_wrappers(HsaApiTable* table) {
    if (!table) return;

    if (table->amd_ext_) {
        hsa_orig::amd_ext_snap           = *table->amd_ext_;
        hsa_orig::amd_signal_create      = hsa_orig::amd_ext_snap.hsa_amd_signal_create_fn;
        hsa_orig::signal_async_handler   = hsa_orig::amd_ext_snap.hsa_amd_signal_async_handler_fn;
        hsa_orig::profiling_get_dispatch = hsa_orig::amd_ext_snap.hsa_amd_profiling_get_dispatch_time_fn;
        hsa_orig::profiling_set_enabled  = hsa_orig::amd_ext_snap.hsa_amd_profiling_set_profiler_enabled_fn;
        hsa_orig::queue_intercept_create   = hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_create_fn;
        hsa_orig::queue_intercept_register = hsa_orig::amd_ext_snap.hsa_amd_queue_intercept_register_fn;
        std::fprintf(stderr,
            "[rocprofv3-qi PREFLIGHT] amd_ext: intercept_create=%p intercept_register=%p "
            "signal_create=%p signal_async_handler=%p profiling_get_dispatch=%p\n",
            (void*)hsa_orig::queue_intercept_create,
            (void*)hsa_orig::queue_intercept_register,
            (void*)hsa_orig::amd_signal_create,
            (void*)hsa_orig::signal_async_handler,
            (void*)hsa_orig::profiling_get_dispatch);
    }
    if (table->core_) {
        hsa_orig::core_snap            = *table->core_;
        hsa_orig::signal_destroy       = hsa_orig::core_snap.hsa_signal_destroy_fn;
        hsa_orig::queue_create         = hsa_orig::core_snap.hsa_queue_create_fn;
        hsa_orig::queue_destroy        = hsa_orig::core_snap.hsa_queue_destroy_fn;
        hsa_orig::iterate_agents       = hsa_orig::core_snap.hsa_iterate_agents_fn;
        hsa_orig::executable_freeze    = hsa_orig::core_snap.hsa_executable_freeze_fn;
        hsa_orig::iterate_agent_symbols = hsa_orig::core_snap.hsa_executable_iterate_agent_symbols_fn;
        hsa_orig::symbol_get_info      = hsa_orig::core_snap.hsa_executable_symbol_get_info_fn;

        if (hsa_orig::queue_create && hsa_orig::queue_intercept_create &&
            hsa_orig::queue_intercept_register) {
            table->core_->hsa_queue_create_fn = &W_hsa_queue_create;
        }
        if (hsa_orig::queue_destroy) {
            table->core_->hsa_queue_destroy_fn = &W_hsa_queue_destroy;
        }
        if (hsa_orig::iterate_agents) {
            table->core_->hsa_iterate_agents_fn = &W_hsa_iterate_agents;
        }
        if (hsa_orig::executable_freeze && hsa_orig::iterate_agent_symbols &&
            hsa_orig::symbol_get_info) {
            table->core_->hsa_executable_freeze_fn = &W_hsa_executable_freeze;
        }
    }
}

} // namespace rocprofv3_min
