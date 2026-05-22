// projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/counters/queue_hooks.hpp
#pragma once

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <memory>

namespace rocprofiler
{
namespace counters
{
// Write-path hook called by WriteInterceptor. Internally enumerates active
// dispatch_counter_collection contexts and their callbacks vector.
// Pushes 0..N entries into inst_pkt (one per active callback that produces
// a packet) and ORs each callback's serialize bool into is_serialized.
void
write_hook(const hsa::Queue&                                        queue,
           const hsa::rocprofiler_packet&                           kernel_packet,
           rocprofiler_kernel_id_t                                  kernel_id,
           rocprofiler_dispatch_id_t                                dispatch_id,
           rocprofiler_user_data_t*                                 user_data,
           const hsa::queue_info_session_t::external_corr_id_map_t& ext_corr_ids,
           const context::correlation_id*                           correlation_id,
           /*out*/ hsa::inst_pkt_t&                                 inst_pkt,
           /*inout*/ bool&                                          is_serialized);

// Completion hook called by AsyncSignalHandler.
void
signal_completion_hook(const hsa::Queue&                           queue,
                       const hsa::rocprofiler_packet&              kernel_packet,
                       std::shared_ptr<hsa::queue_info_session_t>& session,
                       hsa::packet_data_t&                         packet,
                       hsa::inst_pkt_t&                            inst_pkt,
                       kernel_dispatch::profiling_time             dispatch_time);

// Returns true if any context with dispatch_counter_collection is active.
// Used by WriteInterceptor to compute should_batch_packets.
bool
is_any_active();
}  // namespace counters
}  // namespace rocprofiler
