#include "lib/rocprofiler-sdk/pc_sampling/queue_hooks.hpp"

#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
#    include "lib/rocprofiler-sdk/pc_sampling/hsa_adapter.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#endif

namespace rocprofiler
{
namespace pc_sampling
{
bool
is_configured_on_agent(rocprofiler_agent_id_t agent_id)
{
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    return is_pc_sample_service_configured(agent_id);
#else
    (void) agent_id;
    return false;
#endif
}

std::optional<::rocprofiler::hsa::rocprofiler_packet>
maybe_marker_packet(
    const ::rocprofiler::hsa::Queue&                                        queue,
    rocprofiler_dispatch_id_t                                               dispatch_id,
    const ::rocprofiler::hsa::queue_info_session_t::external_corr_id_map_t& ext_corr_ids,
    const context::correlation_id*                                          correlation_id)
{
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    auto agent_id = queue.get_agent().get_rocp_agent()->id;
    if(!is_configured_on_agent(agent_id)) return std::nullopt;
    // generate_marker_packet_for_kernel takes correlation_id* non-const
    // (calls add_ref_count); our signature is const-correct for callers.
    return pc_sampling::hsa::generate_marker_packet_for_kernel(
        const_cast<context::correlation_id*>(correlation_id), ext_corr_ids, dispatch_id);
#else
    (void) queue;
    (void) dispatch_id;
    (void) ext_corr_ids;
    (void) correlation_id;
    return std::nullopt;
#endif
}

void
signal_completion_hook(const ::rocprofiler::hsa::Queue&                           queue,
                       const ::rocprofiler::hsa::rocprofiler_packet&              kernel_packet,
                       std::shared_ptr<::rocprofiler::hsa::queue_info_session_t>& session,
                       ::rocprofiler::hsa::packet_data_t& /*packet*/,
                       ::rocprofiler::hsa::inst_pkt_t& /*inst_pkt*/,
                       kernel_dispatch::profiling_time /*dispatch_time*/)
{
#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0
    if(!session) return;
    // kernel_completion_cb takes a non-const packet& (legacy signature; unused).
    pc_sampling::hsa::kernel_completion_cb(
        queue.get_agent().get_rocp_agent(),
        const_cast<::rocprofiler::hsa::rocprofiler_packet&>(kernel_packet),
        *session);
#else
    (void) queue;
    (void) kernel_packet;
    (void) session;
#endif
}
}  // namespace pc_sampling
}  // namespace rocprofiler
