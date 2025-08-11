// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/pc_sampling/hsa_adapter.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/common/logging.hpp"
#    include "lib/rocprofiler-sdk/context/context.hpp"
#    include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#    include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/parser/pc_record_interface.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/types.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/utils.hpp"

#    include <hsa/hsa.h>
#    include <hsa/hsa_ext_amd.h>
#    include <hsa/hsa_ven_amd_pc_sampling.h>

#    include <mutex>
#    include <optional>
#    include <shared_mutex>
#    include <stdexcept>

namespace rocprofiler
{
namespace pc_sampling
{
namespace hsa
{
namespace
{
const PCSAgentSession*
get_pcs_session_of(hsa_agent_t hsa_agent)
{
    // TODO: optimize this
    auto* service = get_configured_pc_sampling_service().load();
    for(const auto& [_, agent_session] : service->agent_sessions)
    {
        if(agent_session->hsa_agent->handle == hsa_agent.handle)
        {
            return agent_session.get();
        }
    }
    return nullptr;
}

// Called just before the dispatch packet is put inside the real hardware queue.
void
amd_intercept_marker_handler_callback(const struct amd_aql_intercept_marker_s* packet,
                                      hsa_queue_t*                             queue,
                                      uint64_t                                 packet_id)
{
    auto*       ext_table_ = rocprofiler::hsa::get_table().amd_ext_;
    hsa_agent_t hsa_agent;
    if(ext_table_->hsa_amd_queue_get_info_fn(queue, HSA_AMD_QUEUE_INFO_AGENT, &hsa_agent) !=
       HSA_STATUS_SUCCESS)
    {
        ROCP_CI_LOG(WARNING) << "Cannot map hsa_queue_t* to hsa_agent_t";
        return;
    }

    uint64_t doorbell_id = 0;
    if(ext_table_->hsa_amd_queue_get_info_fn(queue, HSA_AMD_QUEUE_INFO_DOORBELL_ID, &doorbell_id) !=
       HSA_STATUS_SUCCESS)
    {
        ROCP_CI_LOG(WARNING) << "Cannot map hsa_queue_t* to doorbell id";
        return;
    }

    auto internal_correlation = packet->user_data[0];
    auto external_correlation = rocprofiler_user_data_t{.value = packet->user_data[1]};

    auto const* pcs_session = get_pcs_session_of(hsa_agent);
    assert(pcs_session);

    dispatch_pkt_id_t dispatch_pkt;
    dispatch_pkt.type = (pcs_session->method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
                            ? AMD_HOST_TRAP_V1
                            : AMD_SNAPSHOT_V1;
    // Use rocp_agent handle to uniquely identify the GPU device
    dispatch_pkt.device      = device_handle{static_cast<uint32_t>(pcs_session->agent->id.handle)};
    dispatch_pkt.doorbell_id = doorbell_id;
    dispatch_pkt.queue_size  = queue->size;
    dispatch_pkt.write_index = packet_id;
    dispatch_pkt.correlation_id = {.internal = internal_correlation,
                                   .external = external_correlation};
    dispatch_pkt.dispatch_id    = packet->user_data[2];

    auto* parser = pcs_session->parser.get();
    if(parser->shouldFlipRocrBuffer(dispatch_pkt))
    {
        rocprofiler::hsa::get_table().pc_sampling_ext_->hsa_ven_amd_pcs_flush_fn(
            pcs_session->hsa_pc_sampling);
    }

    parser->newDispatch(dispatch_pkt);
}

/**
 * Callback called by HSA interceptor when the kernel has completed.
 */
void
kernel_completion_cb(const rocprofiler_agent_t* rocp_agent,
                     rocprofiler::hsa::rocprofiler_packet& /*kernel_pkt*/,
                     const rocprofiler::hsa::Queue::queue_info_session_t& session)
{
    // No internal correlation IDs, meaning there is no need to call CID manager.
    if(!session.correlation_id) return;

    // Check if the PC sampling service is configured on this agent.
    if(!is_pc_sample_service_configured(rocp_agent->id)) return;

    auto* service = get_configured_pc_sampling_service().load();
    assert(service);
    auto* agent_session = service->agent_sessions.at(rocp_agent->id).get();
    // Mark the correlation ID as completed
    agent_session->cid_manager->cid_async_activity_completed(session.correlation_id);
}

void
data_ready_callback(void*                                client_callback_data,
                    size_t                               data_size,
                    size_t                               lost_sample_count,
                    hsa_ven_amd_pcs_data_copy_callback_t data_copy_callback,
                    void*                                hsa_callback_data)
{
    (void) lost_sample_count;  // TODO: How is this exposed to the tool?

    auto* agent_session = static_cast<pc_sampling::PCSAgentSession*>(client_callback_data);

    // Wrap around the logic for copying PC samples from ROCr's buffer to the SDK's
    // PC sampling buffer inside the lambda function called by the CID manager,
    // a component responsible for managing the PC sampling related part of the
    // process of retiring correlation IDs.
    agent_session->cid_manager->manage_cids_implicit([&]() {
        size_t samples_num = data_size / sizeof(packet_union_t);
        // allocate a temporary buffer for copying PC samples
        // TODO: think about how to optimize this (e.g., introduce a buffer pool)
        auto buff = std::make_unique<packet_union_t[]>(samples_num);

        // copy all the data
        data_copy_callback(hsa_callback_data, data_size, buff.get());

        upcoming_samples_t upc;
        // rocp_agent handle uniquely identifies the device
        upc.device = device_handle{static_cast<uint32_t>(agent_session->agent->id.handle)};
        upc.which_sample_type = (agent_session->method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
                                    ? AMD_HOST_TRAP_V1
                                    : AMD_SNAPSHOT_V1;
        upc.num_samples       = samples_num;

        // TODO: how about using std::future
        std::condition_variable cv;

        auto pcs_parser_status =
            agent_session->parser->parse(upc,
                                         reinterpret_cast<const generic_sample_t*>(buff.get()),
                                         agent_session->agent->gfx_target_version,
                                         cv,
                                         false);

        if(pcs_parser_status != PCSAMPLE_STATUS_SUCCESS)
        {
            ROCP_INFO << "PCS Parser encountered samples from a blit kernel.\n";
        }
    });
}
}  // namespace

rocprofiler::hsa::rocprofiler_packet
generate_marker_packet_for_kernel(
    context::correlation_id*                      correlation_id,
    const tracing::external_correlation_id_map_t& external_correlation_ids,
    const rocprofiler_dispatch_id_t               dispatch_id)
{
    // This function executes for each kernel dispatched to the agent on which
    // the PC sampling service is configured.
    // By doing this, we allow the following scenario to happen:
    // A tool configures PC sampling on an agent and offloads some kernels on that agent.
    // In the middle of the kernel execution, a tool starts/activates PC sampling service
    // to collect samples. Although the PC sampling service was not started/activated
    // at the moment of dispatching kernels, the configured PC sampling service is aware of all
    // kernels dispatched on the agent and can recreate their correlation IDs.
    // The disadvantage of this approach is that it introduces overhead when PC sampling
    // service is inactive/stopped.
    amd_aql_intercept_marker_t marker_pkt;
    marker_pkt.header   = HSA_PACKET_TYPE_VENDOR_SPECIFIC;
    marker_pkt.format   = AMD_AQL_FORMAT_INTERCEPT_MARKER;
    marker_pkt.callback = amd_intercept_marker_handler_callback;

    if(correlation_id != nullptr)
    {
        correlation_id->add_ref_count();
        // Use the internal correlation ID generated by the tracing service.
        marker_pkt.user_data[0] = correlation_id->internal;

        // Find a context that holds PC sampling service.
        auto contexts = context::get_registered_contexts(
            [](const auto* ctx) { return ctx->pc_sampler != nullptr; });
        assert(contexts.size() == 1);
        const auto* pcs_context = contexts.at(0);

        // Get an external correlation that corresponds to the context
        // enclosing PC sampling service.
        auto external_corr    = tracing::empty_user_data;
        auto external_corr_it = external_correlation_ids.find(pcs_context);
        if(external_corr_it != external_correlation_ids.end())
            external_corr = external_corr_it->second;
        marker_pkt.user_data[1] = external_corr.value;
    }
    else
    {
        marker_pkt.user_data[0] = 0;
        // No external correlation ID
        marker_pkt.user_data[1] = 0;
    }

    // dispatch_id should always be present
    marker_pkt.user_data[2] = dispatch_id;

    return rocprofiler::hsa::rocprofiler_packet(marker_pkt);
}

void
pc_sampling_service_start(context::pc_sampling_service* service)
{
    auto* pc_sampling_table_ = rocprofiler::hsa::get_table().pc_sampling_ext_;
    for(const auto& [_, agent_session] : service->agent_sessions)
    {
        // If the agent has been hidden by the ROCR_VISIBLE_DEVICES, no need to start PC sampling.
        // Please check `pc_sampling_service_finish_configuration` for more information.
        if(!agent_session->hsa_agent.has_value()) continue;

        if(pc_sampling_table_->hsa_ven_amd_pcs_start_fn(agent_session->hsa_pc_sampling) !=
           HSA_STATUS_SUCCESS)
        {
            // Two concurrent calls to the pc_sampling::start_service are invoked on the same
            // service. The "faster" one succeeds and starts the PC sampling service on the HSA
            // level. Although the "slower fails", the service is started.
            ROCP_ERROR << "HSA runtime failed to start PC sampling on the agent "
                       << agent_session->agent->id.handle << "\n";
        }
    }
}

void
pc_sampling_service_stop(context::pc_sampling_service* service)
{
    auto* pc_sampling_table_ = rocprofiler::hsa::get_table().pc_sampling_ext_;
    for(const auto& [_, agent_session] : service->agent_sessions)
    {
        // If the agent has been hidden by the ROCR_VISIBLE_DEVICES, no need to stop PC sampling.
        // Please check `pc_sampling_service_finish_configuration` for more information.
        if(!agent_session->hsa_agent.has_value()) continue;

        if(pc_sampling_table_->hsa_ven_amd_pcs_stop_fn(agent_session->hsa_pc_sampling) !=
           HSA_STATUS_SUCCESS)
        {
            // Two concurrent calls to the pc_sampling::stop_serivce are invoked on the same
            // service. The "faster" one succeeds and stops the PC sampling service on the HSA
            // level. Although the "slower fails", the service is stopped. The "slower" continues,
            // while the "faster" tries flushing the ROCr's buffer below.
            ROCP_ERROR << "HSA runtime failed to stop PC sampling on the agent "
                       << agent_session->agent->id.handle << "\n";
            continue;
        };

        // Flush internal PC sampling buffers (ROCr + 2nd level trap handler buffers)
        flush_internal_agent_buffers(agent_session.get());
    }
}

void
pc_sampling_service_finish_configuration(context::pc_sampling_service* service)
{
    // This function is executed once by a single thread.
    // No synchronization needed.
    auto* pc_sampling_table_ = rocprofiler::hsa::get_table().pc_sampling_ext_;

    for(const auto& [_, agent_session] : service->agent_sessions)
    {
        // Get the HSA agent handle
        agent_session->hsa_agent = rocprofiler::agent::get_hsa_agent(agent_session->agent);

        // Check if HSA agent corresponding to the KFD node id is hidden via ROCR_VISIBLE_DEVICES,
        // If so, we cannot finish the configuration on the ROCr level.
        // Consequently, no PC samples will be delivered for this device.
        if(!agent_session->hsa_agent.has_value()) continue;

        // Create PC sampling session on the ROCr level.
        // ROCr reuses IOCTL session with `agent_session->ioctl_pcs_id`.
        hsa_status_t status = pc_sampling_table_->hsa_ven_amd_pcs_create_from_id_fn(
            agent_session->ioctl_pcs_id,
            agent_session->hsa_agent.value(),
            pc_sampling::utils::get_matching_hsa_pcs_method(agent_session->method),
            pc_sampling::utils::get_matching_hsa_pcs_units(agent_session->unit),
            agent_session->interval,
            pc_sampling::utils::get_hsa_pcs_latency(),
            pc_sampling::utils::get_hsa_pcs_buffer_size(),
            data_ready_callback,
            agent_session.get(),
            &agent_session->hsa_pc_sampling);

        if(status != HSA_STATUS_SUCCESS)
        {
            ROCP_ERROR << "HSA runtime failed to finish configuring PC sampling service"
                       << " on the agent with id: " << agent_session->agent->id.handle << "\n";
            std::runtime_error("PC sampling config on the HSA/ROCr level failed");
        }

        // TODO: any better way of informing the parser about what buffer is used for a
        // specific agent?
        if(!agent_session->parser->register_buffer_for_agent(agent_session->buffer_id,
                                                             agent_session->agent->id))
        {
            std::runtime_error("PCS parser does not accept buffer");
        }
    }

    // Register callbacks for the HSA's queue interceptor.
    // TODO: should we store callback ID in the service?
    rocprofiler::hsa::get_queue_controller()->add_callback(
        std::nullopt,
        [](const rocprofiler::hsa::Queue&,
           const rocprofiler::hsa::rocprofiler_packet&,
           rocprofiler_kernel_id_t /*kernel_id*/,
           rocprofiler_dispatch_id_t /*dispatch_id*/,
           rocprofiler_user_data_t*,
           const rocprofiler::hsa::Queue::queue_info_session_t::external_corr_id_map_t&,
           const context::correlation_id*) {
            return rocprofiler::hsa::Queue::pkt_and_serialize_t{};
        },
        // Completion CB
        [](const rocprofiler::hsa::Queue&                                  q,
           rocprofiler::hsa::rocprofiler_packet                            kern_pkt,
           std::shared_ptr<rocprofiler::hsa::Queue::queue_info_session_t>& session,
           rocprofiler::hsa::inst_pkt_t&,
           kernel_dispatch::profiling_time) {
            kernel_completion_cb(q.get_agent().get_rocp_agent(), kern_pkt, *session);
        });
}

rocprofiler_status_t
flush_internal_agent_buffers(const PCSAgentSession* agent_session)
{
    // If the agent has been hidden by the ROCR_VISIBLE_DEVICES,
    // there is no ROCr internal buffers to flush.
    if(!agent_session->hsa_agent.has_value()) return ROCPROFILER_STATUS_SUCCESS;

    auto* pc_sampling_table_ = rocprofiler::hsa::get_table().pc_sampling_ext_;

    // HSA table has not been loaded, so ROCr buffers does not exist yet.
    if(!pc_sampling_table_->hsa_ven_amd_pcs_flush_fn)
        return ROCPROFILER_STATUS_ERROR_HSA_NOT_LOADED;

    auto hsa_pcs_handle = agent_session->hsa_pc_sampling;
    // Explicitly flush ROCr's buffers and sync completed CIDs.
    agent_session->cid_manager->manage_cids_explicit([=]() {
        // TODO: investigate whether the ROCr should maintain an extra buffer
        // beyond the 2nd level trap handler buffers.
        if(pc_sampling_table_->hsa_ven_amd_pcs_flush_fn(hsa_pcs_handle) != HSA_STATUS_SUCCESS)
        {
            // TODO: Think if it is possible to recover from this error.
            std::runtime_error("Fail to flush ROCr's buffer explicitly");
        }
    });
    return ROCPROFILER_STATUS_SUCCESS;
}
}  // namespace hsa
}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
