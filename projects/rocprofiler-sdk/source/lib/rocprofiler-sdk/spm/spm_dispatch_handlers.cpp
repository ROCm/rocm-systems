// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/spm/spm_dispatch_handlers.hpp"
#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace rocprofiler
{
namespace SPM
{
bool
AsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    auto* packet = CHECK_NOTNULL(static_cast<hsa::SPMPacket*>(data));
    packet->kfd_start();

    CHECK_NOTNULL(hsa::get_queue_controller())
        ->get_core_table()
        .hsa_signal_destroy_fn(packet->before_krn_barrier_pkt.at(0).completion_signal);
    CHECK_NOTNULL(hsa::get_queue_controller())
        ->get_core_table()
        .hsa_signal_store_screlease_fn(packet->before_krn_barrier_pkt.at(1).dep_signal[0], 0);
    return false;
}

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
hsa::Queue::pkt_and_serialize_t
pre_kernel_call(const context::context*                                         ctx,
                const std::shared_ptr<spm_counter_callback_info>&               info,
                const hsa::Queue&                                               queue,
                const hsa::rocprofiler_packet&                                  pkt,
                uint64_t                                                        kernel_id,
                rocprofiler_dispatch_id_t                                       dispatch_id,
                rocprofiler_user_data_t*                                        user_data,
                const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
                const context::correlation_id*                                  correlation_id)
{
    CHECK(info && ctx);

    if(!ctx || !ctx->dispatch_spm) return {nullptr, false};

    bool is_enabled = false;
    ctx->dispatch_spm->enabled.rlock([&](const auto& collect_ctx) { is_enabled = collect_ctx; });

    if(!is_enabled || !info->user_cb) return {nullptr, false};

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

    auto req_profile = rocprofiler_spm_counter_config_id_t{.handle = 0};
    auto dispatch_data =
        common::init_public_api_struct(rocprofiler_spm_dispatch_counting_service_data_t{});

    dispatch_data.correlation_id = _corr_id_v;
    {
        auto dispatch_info = common::init_public_api_struct(rocprofiler_kernel_dispatch_info_t{});
        dispatch_info.kernel_id            = kernel_id;
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
    }

    info->user_cb(dispatch_data, &req_profile, user_data, info->callback_args);

    if(req_profile.handle == 0) return {nullptr, true};

    auto prof_config = spm_get_controller().get_profile_cfg(req_profile);
    CHECK(prof_config);

    std::unique_ptr<rocprofiler::hsa::SPMPacket> ret_pkt;
    auto ret_status = info->get_spm_packet(ret_pkt, prof_config);
    CHECK_EQ(ret_status, ROCPROFILER_STATUS_SUCCESS) << rocprofiler_get_status_string(ret_status);

    if(!ret_pkt->empty)
    {
        ret_pkt->clear();
        ret_pkt->populate_before();
        ret_pkt->populate_after();

        ret_pkt->dispatch_data = dispatch_data;
        ret_pkt->user_data     = user_data;
        // ROCP_FATAL_IF(ret_pkt->before_krn_pkt.size() < 3) << "SPM Requires at least 3 packets";
        auto& signal_to_start_kfd    = ret_pkt->before_krn_barrier_pkt.at(0).completion_signal;
        auto& signal_kfd_has_started = ret_pkt->before_krn_barrier_pkt.at(1).dep_signal[0];

        // queue.create_signal(0, &signal_to_start_kfd);
        //  queue.create_signal(0, &signal_kfd_has_started);
        CHECK_NOTNULL(hsa::get_queue_controller())
            ->get_ext_table()
            .hsa_amd_signal_create_fn(1, 0, nullptr, 0, &signal_to_start_kfd);
        CHECK_NOTNULL(hsa::get_queue_controller())
            ->get_ext_table()
            .hsa_amd_signal_create_fn(1, 0, nullptr, 0, &signal_kfd_has_started);

        CHECK_NOTNULL(hsa::get_queue_controller())
            ->get_core_table()
            .hsa_signal_store_screlease_fn(signal_kfd_has_started, -1);
        CHECK_NOTNULL(hsa::get_queue_controller())
            ->get_core_table()
            .hsa_signal_store_screlease_fn(signal_to_start_kfd, 0);

        auto status = CHECK_NOTNULL(hsa::get_queue_controller())
                          ->get_ext_table()
                          .hsa_amd_signal_async_handler_fn(signal_to_start_kfd,
                                                           HSA_SIGNAL_CONDITION_EQ,
                                                           -1,
                                                           rocprofiler::SPM::AsyncSignalHandler,
                                                           ret_pkt.get());
        ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
            << "Error: hsa_amd_signal_async_handler failed with error code " << status
            << " :: " << hsa::get_hsa_status_string(status);
    }
    return {std::move(ret_pkt), true};
}

/**
 * Callback called by HSA interceptor when the kernel has completed processing.
 */
void
post_kernel_call(const context::context*                           ctx,
                 const std::shared_ptr<spm_counter_callback_info>& info,
                 std::shared_ptr<hsa::Queue::queue_info_session_t>& /*ptr_session*/,
                 inst_pkt_t& pkts,
                 kernel_dispatch::profiling_time /*dispatch_time*/)
{
    CHECK(info && ctx);

    std::shared_ptr<spm_counter_config> prof_config;
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
                CHECK_NOTNULL(hsa::get_queue_controller())
                    ->get_core_table()
                    .hsa_signal_destroy_fn(pkt->before_krn_barrier_pkt.at(1).dep_signal[0]);
                pkt->kfd_stop();
                auto rel_pkt = std::move(aql_pkt);
                return;
            }
        }
    });
}
}  // namespace SPM
}  // namespace rocprofiler
