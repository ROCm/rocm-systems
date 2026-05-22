// projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp
#pragma once

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
namespace queue_hooks
{
// Compile-time identifiers used to tag inst_pkt_t entries by their producer subsystem.
// Replace the runtime-assigned ClientID values from QueueController::add_callback.
// The numeric values don't matter — only distinctness does. We use 1/2/3 for stable
// ordering in tests that inspect inst_pkt_t order.
constexpr int64_t COUNTERS_CLIENT_ID     = 1;
constexpr int64_t THREAD_TRACE_CLIENT_ID = 2;
constexpr int64_t PC_SAMPLING_CLIENT_ID  = 3;
}  // namespace queue_hooks
}  // namespace hsa
}  // namespace rocprofiler
