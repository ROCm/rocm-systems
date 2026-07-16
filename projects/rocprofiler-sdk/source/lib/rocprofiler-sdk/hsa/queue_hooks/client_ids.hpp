#pragma once

#include <cstdint>

namespace rocprofiler
{
namespace hsa
{
namespace queue_hooks
{
// Tags identifying the producer subsystem of each inst_pkt_t entry.
// Only distinctness matters; values are fixed for test-order stability.
constexpr int64_t COUNTERS_CLIENT_ID     = 1;
constexpr int64_t THREAD_TRACE_CLIENT_ID = 2;
constexpr int64_t PC_SAMPLING_CLIENT_ID  = 3;
constexpr int64_t SPM_CLIENT_ID          = 4;
}  // namespace queue_hooks
}  // namespace hsa
}  // namespace rocprofiler
