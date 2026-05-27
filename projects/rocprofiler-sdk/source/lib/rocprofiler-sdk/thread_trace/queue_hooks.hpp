#pragma once

#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <memory>

namespace rocprofiler
{
namespace thread_trace
{
// Iterates active dispatch_thread_trace contexts and calls each tracer's
// pre_kernel_call; appends produced packets to inst_pkt, OR-folding each
// tracer's serialize flag into is_serialized.
void
write_hook(const hsa::Queue&                                        queue,
           const hsa::rocprofiler_packet&                           kernel_packet,
           rocprofiler_kernel_id_t                                  kernel_id,
           rocprofiler_dispatch_id_t                                dispatch_id,
           rocprofiler_user_data_t*                                 user_data,
           const hsa::queue_info_session_t::external_corr_id_map_t& ext_corr_ids,
           const context::correlation_id*                           correlation_id,
           hsa::inst_pkt_t&                                         inst_pkt,
           bool&                                                    is_serialized);

void
signal_completion_hook(const hsa::Queue&                           queue,
                       const hsa::rocprofiler_packet&              kernel_packet,
                       std::shared_ptr<hsa::queue_info_session_t>& session,
                       hsa::packet_data_t&                         packet,
                       hsa::inst_pkt_t&                            inst_pkt,
                       kernel_dispatch::profiling_time             dispatch_time);

bool
is_any_active();
}  // namespace thread_trace
}  // namespace rocprofiler
