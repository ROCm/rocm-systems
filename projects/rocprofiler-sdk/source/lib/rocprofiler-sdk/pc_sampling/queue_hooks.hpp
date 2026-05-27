#pragma once

#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <memory>
#include <optional>

namespace rocprofiler
{
namespace pc_sampling
{
// hsa types are fully qualified as ::rocprofiler::hsa::... because
// rocprofiler::pc_sampling::hsa is also a namespace (in hsa_adapter.hpp) and
// would shadow the intended one.

// Returns the marker packet if the queue's agent has PC sampling configured,
// else nullopt. WriteInterceptor emplaces this directly into the AQL stream
// (not via inst_pkt). Always returns nullopt in ROCPROFILER_SDK_HSA_PC_SAMPLING==0.
std::optional<::rocprofiler::hsa::rocprofiler_packet>
maybe_marker_packet(
    const ::rocprofiler::hsa::Queue&                                        queue,
    rocprofiler_dispatch_id_t                                               dispatch_id,
    const ::rocprofiler::hsa::queue_info_session_t::external_corr_id_map_t& ext_corr_ids,
    const context::correlation_id*                                          correlation_id);

void
signal_completion_hook(const ::rocprofiler::hsa::Queue&                          queue,
                       const ::rocprofiler::hsa::rocprofiler_packet&             kernel_packet,
                       std::shared_ptr<::rocprofiler::hsa::queue_info_session_t>& session,
                       ::rocprofiler::hsa::packet_data_t&                        packet,
                       ::rocprofiler::hsa::inst_pkt_t&                           inst_pkt,
                       kernel_dispatch::profiling_time                           dispatch_time);

// Always-available wrapper around is_pc_sample_service_configured; returns
// false in ROCPROFILER_SDK_HSA_PC_SAMPLING==0 builds. Lets callers (e.g.
// WriteInterceptor) query configured-ness without an #if guard.
bool
is_configured_on_agent(rocprofiler_agent_id_t agent_id);
}  // namespace pc_sampling
}  // namespace rocprofiler
