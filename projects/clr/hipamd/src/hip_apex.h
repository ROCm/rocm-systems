/* Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * APEX (Adaptive Prefetch EXtensions) - HIP Integration
 *
 * Provides GPU memory tracking, expert weight caching, and
 * proactive prefetch hooks for MoE inference workloads.
 *
 * Enabled via: APEX_GPU_ENABLE=1 (zero-cost when disabled)
 * Requires: HSA_XNACK=1 and amdgpu noretry=0
 */

#ifndef HIP_SRC_HIP_APEX_H
#define HIP_SRC_HIP_APEX_H

#include <cstddef>
#include <cstdint>

namespace apex {

// Check if APEX GPU integration is enabled (cached env var check)
bool enabled();

// Allocation tracking — called from ihipMalloc/ihipFree/ihipMallocManaged
void track_alloc(void* ptr, size_t size, unsigned int flags, bool managed);
void track_free(void* ptr);

// Pre-launch hook — called from ihipLaunchKernel before GPU dispatch
// Can trigger proactive prefetch of kernel arguments
void pre_launch(const void* host_function, void** args);

// Record prefetch — called from ihipMemPrefetchAsync for policy feedback
void record_prefetch(const void* ptr, size_t count, int device);

// Expert weight management — called by inference frameworks via HIP extension
void register_expert(uint32_t expert_id, void* ptr, size_t size);
void select_experts(const uint32_t* expert_ids, uint32_t count);

// Statistics
struct Stats {
    uint64_t allocs;
    uint64_t frees;
    uint64_t managed_bytes;
    uint64_t prefetches_triggered;
    uint64_t experts_registered;
    uint64_t expert_cache_hits;
    uint64_t expert_cache_misses;
};

Stats get_stats();

}  // namespace apex

#endif  // HIP_SRC_HIP_APEX_H
