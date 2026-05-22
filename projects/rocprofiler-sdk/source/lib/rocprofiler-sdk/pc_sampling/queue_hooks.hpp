// projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/pc_sampling/queue_hooks.hpp
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
// All hsa:: types are fully qualified as ::rocprofiler::hsa::... below because
// rocprofiler::pc_sampling::hsa is also a real namespace (defined in
// hsa_adapter.hpp). Unqualified `hsa::Type` lookup from inside this namespace
// would bind to pc_sampling::hsa first per C++ nested-name-specifier rules and
// fail to find the type. See queue_hooks.cpp for the same pattern.

// Returns the marker packet for this dispatch if the agent is currently
// configured for PC sampling, else std::nullopt. The returned packet is
// emplaced directly into the AQL stream by WriteInterceptor (NOT through
// inst_pkt). This matches today's behavior at queue.cpp:650-657.
//
// In ROCPROFILER_SDK_HSA_PC_SAMPLING==0 builds, always returns std::nullopt.
std::optional<::rocprofiler::hsa::rocprofiler_packet>
maybe_marker_packet(
    const ::rocprofiler::hsa::Queue&                                        queue,
    rocprofiler_dispatch_id_t                                               dispatch_id,
    const ::rocprofiler::hsa::queue_info_session_t::external_corr_id_map_t& ext_corr_ids,
    const context::correlation_id*                                          correlation_id);

// Completion hook. Calls kernel_completion_cb if PC sampling is configured for
// this agent. No-op in disabled builds.
void
signal_completion_hook(const ::rocprofiler::hsa::Queue&                       queue,
                       const ::rocprofiler::hsa::rocprofiler_packet&          kernel_packet,
                       std::shared_ptr<::rocprofiler::hsa::queue_info_session_t>& session,
                       ::rocprofiler::hsa::packet_data_t&                     packet,
                       ::rocprofiler::hsa::inst_pkt_t&                        inst_pkt,
                       kernel_dispatch::profiling_time                        dispatch_time);

// PC sampling does not need is_any_active() — should_batch_packets only
// consults subsystems that REQUIRE per-packet mode.

// Always-available wrapper around is_pc_sample_service_configured.
// Returns false in ROCPROFILER_SDK_HSA_PC_SAMPLING==0 builds.
bool
is_configured_on_agent(rocprofiler_agent_id_t agent_id);
}  // namespace pc_sampling
}  // namespace rocprofiler
