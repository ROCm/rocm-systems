// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/spm/dispatch_handlers.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <rocprofiler-sdk/rocprofiler.h>

namespace rocprofiler
{
namespace spm
{
/**
 * @brief Async Handler for barrier=packet1 completion signal
 * Destroys the barrier-packet1 completion signal
 * Sets the dependency signal of barrier packet-2 to 0, so that barrier-packet2 can complete
 * This guarantees that SPM has been started before the dispatch
 **/
bool
AsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    auto* pkt    = CHECK_NOTNULL(static_cast<hsa::AQLPacket*>(data));
    auto* packet = CHECK_NOTNULL(dynamic_cast<hsa::SPMPacket*>(pkt));

    if(!packet->kfd_start())
    {
        ROCP_ERROR << "SPM KFD start failed in async signal handler";
    }

    CHECK_NOTNULL(hsa::get_queue_controller())
        ->get_core_table()
        .hsa_signal_destroy_fn(packet->before_krn_barrier_pkt.at(0).completion_signal);
    CHECK_NOTNULL(hsa::get_queue_controller())
        ->get_core_table()
        .hsa_signal_store_screlease_fn(packet->before_krn_barrier_pkt.at(1).dep_signal[0], 0);
    return false;
}

namespace
{
hsa_status_t
setup_spm_barrier_signals(hsa::SPMPacket& spm_pkt)
{
    auto& signal_to_start_kfd    = spm_pkt.before_krn_barrier_pkt.at(0).completion_signal;
    auto& signal_kfd_has_started = spm_pkt.before_krn_barrier_pkt.at(1).dep_signal[0];

    auto* queue_controller = CHECK_NOTNULL(hsa::get_queue_controller());

    auto status_signal_to_start_kfd = queue_controller->get_ext_table().hsa_amd_signal_create_fn(
        1, 0, nullptr, 0, &signal_to_start_kfd);
    auto status_signal_kfd_has_started = queue_controller->get_ext_table().hsa_amd_signal_create_fn(
        1, 0, nullptr, 0, &signal_kfd_has_started);

    if(status_signal_to_start_kfd != HSA_STATUS_SUCCESS ||
       status_signal_kfd_has_started != HSA_STATUS_SUCCESS)
    {
        if(status_signal_to_start_kfd == HSA_STATUS_SUCCESS)
            queue_controller->get_core_table().hsa_signal_destroy_fn(signal_to_start_kfd);
        if(status_signal_kfd_has_started == HSA_STATUS_SUCCESS)
            queue_controller->get_core_table().hsa_signal_destroy_fn(signal_kfd_has_started);
        return HSA_STATUS_ERROR;
    }

    queue_controller->get_core_table().hsa_signal_store_screlease_fn(signal_kfd_has_started, -1);
    queue_controller->get_core_table().hsa_signal_store_screlease_fn(signal_to_start_kfd, 0);

    auto status_async_handler = queue_controller->get_ext_table().hsa_amd_signal_async_handler_fn(
        signal_to_start_kfd,
        HSA_SIGNAL_CONDITION_EQ,
        -1,
        rocprofiler::spm::AsyncSignalHandler,
        &spm_pkt);

    if(status_async_handler != HSA_STATUS_SUCCESS && status_async_handler != HSA_STATUS_INFO_BREAK)
    {
        queue_controller->get_core_table().hsa_signal_destroy_fn(signal_to_start_kfd);
        queue_controller->get_core_table().hsa_signal_destroy_fn(signal_kfd_has_started);
        ROCP_WARNING << "hsa_amd_signal_async_handler failed with error code "
                     << status_async_handler
                     << " :: " << hsa::get_hsa_status_string(status_async_handler);
        return HSA_STATUS_ERROR;
    }

    return HSA_STATUS_SUCCESS;
}
}  // namespace

/**
 * @brief Callback we get from HSA interceptor when a kernel packet is being enqueued.
 * We return an AQLPacket containing the start/stop .
 * Barrier_packet1-barrier_packet2-SPM Start Packet - kernel packets- SPM stop packet
 * AsyncSignalHandler - barrier-packet1 completion signal handler
 * barrier-packet2 - dependency signal- initialized to value -1
 * Update callback, user data, and dispatch data in the SPM packet
 * handshake protocol with aqlprofile SPM start kfd - SPM start packet- kernel dispatch - SPM stop
 * packet - SPM stop KFD
 */
hsa::write_packet_t
pre_kernel_call(const context::context*                                  ctx,
                const std::shared_ptr<spm_counter_callback_info>&        info,
                const hsa::Queue&                                        queue,
                const hsa::rocprofiler_packet&                           pkt,
                uint64_t                                                 kernel_id,
                rocprofiler_dispatch_id_t                                dispatch_id,
                rocprofiler_user_data_t*                                 user_data,
                const hsa::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
                const context::correlation_id*                           correlation_id)
{
    CHECK(info && ctx);
    auto no_instrumentation = [&]() {
        auto ret_pkt = std::make_unique<rocprofiler::hsa::EmptyAQLPacket>();
        info->packet_return_map.wlock([&](auto& data) { data.emplace(ret_pkt.get(), nullptr); });
        // If we have a SPM counter collection context but it is not enabled, we still might need
        // to add barrier packets to transition from serialized -> unserialized execution. This
        // transition is coordinated by the serializer.
        return ret_pkt;
    };

    if(!ctx || !ctx->dispatch_spm) return {nullptr, false};

    bool is_enabled = false;
    ctx->dispatch_spm->enabled.rlock([&](const auto& collect_ctx) { is_enabled = collect_ctx; });

    if(!is_enabled || !info->user_cb) return {no_instrumentation(), true};

    auto _corr_id_v =
        rocprofiler_async_correlation_id_t{.internal = 0, .external = context::null_user_data};
    if(const auto* _corr_id = correlation_id)
    {
        _corr_id_v.internal = _corr_id->internal;
        if(const auto* external =
               rocprofiler::common::get_val(extern_corr_ids, info->internal_context))
        {
            _corr_id_v.external = *external;
        }
    }

    auto req_profile = rocprofiler_counter_config_id_t{.handle = 0};
    auto dispatch_data =
        common::init_public_api_struct(rocprofiler_spm_dispatch_counting_service_data_t{});

    dispatch_data.correlation_id = _corr_id_v;

    auto dispatch_info      = common::init_public_api_struct(rocprofiler_kernel_dispatch_info_t{});
    dispatch_info.kernel_id = kernel_id;
    dispatch_info.dispatch_id          = dispatch_id;
    dispatch_info.agent_id             = CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id;
    dispatch_info.queue_id             = queue.get_id();
    dispatch_info.private_segment_size = pkt.kernel_dispatch.private_segment_size;
    dispatch_info.group_segment_size   = pkt.kernel_dispatch.group_segment_size;
    dispatch_info.workgroup_size       = {pkt.kernel_dispatch.workgroup_size_x,
                                    pkt.kernel_dispatch.workgroup_size_y,
                                    pkt.kernel_dispatch.workgroup_size_z};
    dispatch_info.grid_size            = {pkt.kernel_dispatch.grid_size_x,
                               pkt.kernel_dispatch.grid_size_y,
                               pkt.kernel_dispatch.grid_size_z};
    dispatch_data.dispatch_info        = dispatch_info;

    info->user_cb(&dispatch_data, &req_profile, user_data, info->callback_args);

    if(req_profile.handle == 0) return {no_instrumentation(), true};

    auto prof_config = get_spm_counter_config(req_profile);
    CHECK(prof_config);

    std::unique_ptr<rocprofiler::hsa::AQLPacket> ret_pkt    = nullptr;
    auto                                         ret_status = get_spm_packet(ret_pkt, prof_config);

    CHECK_EQ(ret_status, ROCPROFILER_STATUS_SUCCESS) << rocprofiler_get_status_string(ret_status);

    if(!ret_pkt->empty)
    {
        auto* spm_pkt = CHECK_NOTNULL(dynamic_cast<hsa::SPMPacket*>(ret_pkt.get()));
        spm_pkt->clear();
        spm_pkt->populate_before();
        spm_pkt->populate_after();

        spm_pkt->cb.dispatch_data = dispatch_data;
        spm_pkt->cb.user_data     = *user_data;
        if(info->buffer)
            spm_pkt->cb.buffer = info->buffer;
        else
        {
            spm_pkt->cb.record_cb            = info->record_callback;
            spm_pkt->cb.record_callback_args = info->record_callback_args;
        }

        if(setup_spm_barrier_signals(*spm_pkt) != HSA_STATUS_SUCCESS)
        {
            ROCP_WARNING << "SPM signal creation failed, skipping instrumentation";
            return {no_instrumentation(), true};
        }

        info->packet_return_map.wlock(
            [&](auto& data) { data.emplace(ret_pkt.get(), prof_config); });
    }
    return {std::move(ret_pkt), true};
}

/**
 * @brief Callback called by HSA interceptor when the kernel has completed processing.
 * Destroys the depedency signal of barrier packet2
 * Invokes KFD SPM stop
 * Removes entry in packet_return_map
 * Puts the aql packet into config's packets cache for re-use
 */
void
post_kernel_call(const context::context*                           ctx,
                 const std::shared_ptr<spm_counter_callback_info>& info,
                 std::shared_ptr<hsa::queue_info_session_t>& /*ptr_session*/,
                 inst_pkt_t& pkts,
                 kernel_dispatch::profiling_time /*dispatch_time*/)
{
    CHECK(info && ctx);

    std::shared_ptr<spm_counter_config>          prof_config;
    std::unique_ptr<rocprofiler::hsa::AQLPacket> rel_pkt;
    // Get the Profile Config
    info->packet_return_map.wlock([&](auto& data) {
        for(auto& [aql_pkt, _] : pkts)
        {
            const auto& profile = rocprofiler::common::get_val(data, aql_pkt.get());
            if(profile)
            {
                prof_config = *profile;
                data.erase(aql_pkt.get());

                auto* pkt = dynamic_cast<hsa::SPMPacket*>(aql_pkt.get());
                if(!pkt) continue;

                CHECK_NOTNULL(hsa::get_queue_controller())
                    ->get_core_table()
                    .hsa_signal_destroy_fn(pkt->before_krn_barrier_pkt.at(1).dep_signal[0]);
                if(!pkt->kfd_stop())
                {
                    ROCP_ERROR << "SPM KFD stop failed in post-kernel completion";
                }
                if(pkt->cb.record_cb)
                {
                    pkt->cb.record_cb(&pkt->cb.dispatch_data,
                                      nullptr,
                                      0,
                                      ROCPROFILER_SPM_RECORD_FLAG_DISPATCH_END,
                                      pkt->cb.user_data,
                                      pkt->cb.record_callback_args);
                }
                pkt->clear();
                rel_pkt = std::move(aql_pkt);
                return;
            }
        }
    });
    if(rel_pkt)
        prof_config->packets.wlock(
            [&](auto& pkt_vector) { pkt_vector.emplace_back(std::move(rel_pkt)); });
}
}  // namespace spm
}  // namespace rocprofiler
