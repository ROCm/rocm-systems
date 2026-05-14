/* Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * APEX (Adaptive Prefetch EXtensions) - HIP Integration
 *
 * GPU memory tracking and proactive prefetch for MoE inference.
 * Zero-cost when APEX_GPU_ENABLE is not set.
 */

#include "hip_apex.h"

#include <hip/hip_runtime.h>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cstdio>

namespace apex {

// ============================================================
// Global state
// ============================================================

static std::atomic<int> g_enabled{-1};  // -1 = not checked, 0 = off, 1 = on
static std::atomic<int> g_debug{-1};

struct Allocation {
    void* ptr;
    size_t size;
    unsigned int flags;
    bool managed;
    int preferred_device;  // -1 = no preference
};

struct Expert {
    uint32_t id;
    void* ptr;
    size_t size;
    int current_device;     // -1 = system RAM, 0+ = GPU
    uint64_t last_access;   // monotonic counter
};

static std::mutex g_mutex;
static std::unordered_map<void*, Allocation> g_allocs;
static std::unordered_map<uint32_t, Expert> g_experts;
static Stats g_stats = {};
static uint64_t g_access_counter = 0;

static bool debug_enabled() {
    int d = g_debug.load(std::memory_order_relaxed);
    if (d < 0) {
        const char* env = getenv("APEX_GPU_DEBUG");
        d = (env && env[0] == '1') ? 1 : 0;
        g_debug.store(d, std::memory_order_relaxed);
    }
    return d == 1;
}

// ============================================================
// Public API
// ============================================================

bool enabled() {
    int e = g_enabled.load(std::memory_order_relaxed);
    if (e < 0) {
        const char* env = getenv("APEX_GPU_ENABLE");
        e = (env && env[0] == '1') ? 1 : 0;
        g_enabled.store(e, std::memory_order_relaxed);
        if (e) {
            fprintf(stderr, "[APEX] GPU integration enabled\n");
        }
    }
    return e == 1;
}

void track_alloc(void* ptr, size_t size, unsigned int flags, bool managed) {
    if (!ptr || !size) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    g_allocs[ptr] = Allocation{ptr, size, flags, managed, -1};
    g_stats.allocs++;
    if (managed) {
        g_stats.managed_bytes += size;
    }

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] alloc %p size=%zu managed=%d flags=0x%x\n",
                ptr, size, managed, flags);
    }
}

void track_free(void* ptr) {
    if (!ptr) return;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_allocs.find(ptr);
    if (it != g_allocs.end()) {
        if (it->second.managed) {
            g_stats.managed_bytes -= it->second.size;
        }
        g_allocs.erase(it);
        g_stats.frees++;

        if (debug_enabled()) {
            fprintf(stderr, "[APEX] free %p\n", ptr);
        }
    }
}

void pre_launch(const void* host_function, void** args) {
    // In a full implementation, this would:
    // 1. Look up the kernel by host_function pointer
    // 2. Inspect kernel arguments to find pointers to tracked allocations
    // 3. For MoE expert kernels, prefetch selected expert weights to HBM
    //
    // For now, we just count launches for statistics
    (void)host_function;
    (void)args;

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] pre_launch func=%p\n", host_function);
    }
}

void record_prefetch(const void* ptr, size_t count, int device) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_stats.prefetches_triggered++;

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] prefetch %p size=%zu device=%d\n", ptr, count, device);
    }
}

void register_expert(uint32_t expert_id, void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_experts[expert_id] = Expert{expert_id, ptr, size, -1, 0};
    g_stats.experts_registered++;

    if (debug_enabled()) {
        fprintf(stderr, "[APEX] register expert %u ptr=%p size=%zu\n",
                expert_id, ptr, size);
    }
}

void select_experts(const uint32_t* expert_ids, uint32_t count) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_access_counter++;

    for (uint32_t i = 0; i < count; i++) {
        auto it = g_experts.find(expert_ids[i]);
        if (it == g_experts.end()) continue;

        Expert& exp = it->second;
        exp.last_access = g_access_counter;

        if (exp.current_device >= 0) {
            // Expert already in HBM — cache hit
            g_stats.expert_cache_hits++;
        } else {
            // Expert in system RAM — need to prefetch
            g_stats.expert_cache_misses++;

            // Proactively prefetch to GPU 0
            hipError_t err = hipMemPrefetchAsync(exp.ptr, exp.size, 0, nullptr);
            if (err == hipSuccess) {
                exp.current_device = 0;
                g_stats.prefetches_triggered++;
                if (debug_enabled()) {
                    fprintf(stderr, "[APEX] prefetch expert %u (%zu bytes) to GPU 0\n",
                            exp.id, exp.size);
                }
            } else if (debug_enabled()) {
                fprintf(stderr, "[APEX] prefetch expert %u FAILED: %s\n",
                        exp.id, hipGetErrorString(err));
            }
        }
    }
}

Stats get_stats() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_stats;
}

}  // namespace apex
