#include "lib/rocprofiler-sdk/thread_trace/queue_hooks.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_hooks/client_ids.hpp"
#include "lib/rocprofiler-sdk/thread_trace/core.hpp"

namespace rocprofiler
{
namespace thread_trace
{
namespace
{
auto
active_thread_trace_contexts_filter()
{
    return [](const context::context* ctx) -> bool {
        return ctx && ctx->dispatch_thread_trace != nullptr;
    };
}
}  // namespace

void
write_hook(const hsa::Queue&              queue,
           const hsa::rocprofiler_packet& /*kernel_packet*/,
           rocprofiler_kernel_id_t        kernel_id,
           rocprofiler_dispatch_id_t      dispatch_id,
           rocprofiler_user_data_t*       user_data,
           const hsa::queue_info_session_t::external_corr_id_map_t& /*ext_corr_ids*/,
           const context::correlation_id* correlation_id,
           hsa::inst_pkt_t&               inst_pkt,
           bool&                          is_serialized)
{
    auto active = context::get_active_contexts(active_thread_trace_contexts_filter());
    for(auto* ctx : active)
    {
        auto& tracer = *ctx->dispatch_thread_trace;
        auto [packet, bSerial] =
            tracer.pre_kernel_call(queue, kernel_id, dispatch_id, user_data, correlation_id);
        if(packet)
            inst_pkt.emplace_back(std::move(packet), hsa::queue_hooks::THREAD_TRACE_CLIENT_ID);
        is_serialized |= bSerial;
    }
}

void
signal_completion_hook(const hsa::Queue& /*queue*/,
                       const hsa::rocprofiler_packet& /*kernel_packet*/,
                       std::shared_ptr<hsa::queue_info_session_t>& session,
                       hsa::packet_data_t&                         packet_data,
                       hsa::inst_pkt_t&                            inst_pkt,
                       kernel_dispatch::profiling_time /*dispatch_time*/)
{
    auto active = context::get_active_contexts(active_thread_trace_contexts_filter());
    for(auto* ctx : active)
    {
        auto& tracer = *ctx->dispatch_thread_trace;
        tracer.post_kernel_call(inst_pkt, *session, packet_data);
    }
}

bool
is_any_active()
{
    return !context::get_active_contexts(active_thread_trace_contexts_filter()).empty();
}
}  // namespace thread_trace
}  // namespace rocprofiler
