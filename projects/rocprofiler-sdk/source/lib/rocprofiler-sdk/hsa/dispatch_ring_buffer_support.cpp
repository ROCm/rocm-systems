// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#include "lib/rocprofiler-sdk/hsa/dispatch_ring_buffer_support.hpp"

#include <hsa/hsa.h>

#include <atomic>
#include <dlfcn.h>

namespace rocprofiler
{
namespace hsa
{
namespace
{
void*            g_iterate_sym              = nullptr;
void*            g_get_dispatch_records_sym = nullptr;
std::atomic<int> g_resolved{0};
}  // namespace

void
dispatch_ring_buffer_resolve_apis()
{
    int expected = 0;
    if(!g_resolved.compare_exchange_strong(expected, 1)) return;

    g_iterate_sym =
        dlsym(RTLD_DEFAULT, "hsa_amd_queue_iterate");  // NOLINT(concurrency-mt-unsafe)
    g_get_dispatch_records_sym = dlsym(                 // NOLINT(concurrency-mt-unsafe)
        RTLD_DEFAULT, "hsa_amd_profiling_get_dispatch_records");

    if(!g_iterate_sym || !g_get_dispatch_records_sym)
    {
        g_iterate_sym              = nullptr;
        g_get_dispatch_records_sym = nullptr;
    }
    g_resolved.store(2, std::memory_order_release);
}

bool
firmware_dispatch_ring_available()
{
    if(g_resolved.load(std::memory_order_acquire) != 2) dispatch_ring_buffer_resolve_apis();
    return g_iterate_sym != nullptr && g_get_dispatch_records_sym != nullptr;
}

hsa_amd_queue_iterate_fn_t
dispatch_ring_buffer_queue_iterate_fn_v()
{
    if(!firmware_dispatch_ring_available()) return nullptr;
    return reinterpret_cast<hsa_amd_queue_iterate_fn_t>(g_iterate_sym);
}

hsa_amd_profiling_get_dispatch_records_fn_t
dispatch_ring_buffer_get_dispatch_records_fn_v()
{
    if(!firmware_dispatch_ring_available()) return nullptr;
    return reinterpret_cast<hsa_amd_profiling_get_dispatch_records_fn_t>(
        g_get_dispatch_records_sym);
}

}  // namespace hsa
}  // namespace rocprofiler
