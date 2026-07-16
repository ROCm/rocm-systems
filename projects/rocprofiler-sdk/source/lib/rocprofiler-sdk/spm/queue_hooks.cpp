#include "lib/rocprofiler-sdk/spm/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"
#include "lib/rocprofiler-sdk/spm/dispatch_handlers.hpp"

namespace rocprofiler
{
namespace spm
{
namespace
{
auto
active_spm_contexts_filter()
{
    return [](const context::context* ctx) -> bool { return ctx && ctx->dispatch_spm != nullptr; };
}
}  // namespace

void
write_hook(const hsa::Queue&                                        queue,
           const hsa::rocprofiler_packet&                           kernel_packet,
           rocprofiler_kernel_id_t                                  kernel_id,
           rocprofiler_dispatch_id_t                                dispatch_id,
           rocprofiler_user_data_t*                                 user_data,
           const hsa::queue_info_session_t::external_corr_id_map_t& ext_corr_ids,
           const context::correlation_id*                           correlation_id,
           hsa::inst_pkt_t&                                         inst_pkt,
           bool&                                                    is_serialized)
{
    auto active = context::get_active_contexts(active_spm_contexts_filter());
    for(const auto* ctx : active)
    {
        for(auto& cb : ctx->dispatch_spm->callbacks)
        {
            auto [packet, bSerial] = pre_kernel_call(ctx,
                                                     cb,
                                                     queue,
                                                     kernel_packet,
                                                     kernel_id,
                                                     dispatch_id,
                                                     user_data,
                                                     ext_corr_ids,
                                                     correlation_id);
            if(packet) inst_pkt.emplace_back(std::move(packet), hsa::queue_hooks::SPM_CLIENT_ID);
            is_serialized |= bSerial;
        }
    }
}

void
signal_completion_hook(const hsa::Queue& /*queue*/,
                       const hsa::rocprofiler_packet& /*kernel_packet*/,
                       std::shared_ptr<hsa::queue_info_session_t>& session,
                       hsa::packet_data_t& /*packet*/,
                       hsa::inst_pkt_t&                inst_pkt,
                       kernel_dispatch::profiling_time dispatch_time)
{
    auto active = context::get_active_contexts(active_spm_contexts_filter());
    for(const auto* ctx : active)
    {
        for(auto& cb : ctx->dispatch_spm->callbacks)
        {
            post_kernel_call(ctx, cb, session, inst_pkt, dispatch_time);
        }
    }
}

bool
is_any_active()
{
    return !context::get_active_contexts(active_spm_contexts_filter()).empty();
}
}  // namespace spm
}  // namespace rocprofiler
