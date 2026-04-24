// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <hsa/hsa.h>

namespace rocprofiler
{
namespace hsa
{
using hsa_amd_queue_iterate_fn_t =
    hsa_status_t (*)(hsa_status_t (*callback)(hsa_queue_t* queue, void* userdata), void* userdata);

using hsa_amd_profiling_get_dispatch_records_fn_t = hsa_status_t (*)(
    hsa_queue_t* queue, void** buf, uint32_t* size, volatile uint32_t** wptr);

/** For pc_sampling dispatch ring drainer only. */
hsa_amd_queue_iterate_fn_t
dispatch_ring_buffer_queue_iterate_fn_v();

hsa_amd_profiling_get_dispatch_records_fn_t
dispatch_ring_buffer_get_dispatch_records_fn_v();

/**
 * True when ROCr exports the MEC firmware dispatch ring buffer API pair
 * (queue iteration + dispatch record buffer) resolved at runtime, independent of
 * the installed ::AmdExtTable size (Agent 3 / ROCr side may add these after our
 * SDK build).
 */
bool
firmware_dispatch_ring_available();

/** Called once HSA tables are ready; resolves optional symbols via dlsym. */
void
dispatch_ring_buffer_resolve_apis();
}  // namespace hsa
}  // namespace rocprofiler
