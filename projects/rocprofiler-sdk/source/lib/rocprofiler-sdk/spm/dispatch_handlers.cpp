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

#include "lib/rocprofiler-sdk/spm/dispatch_handlers.hpp"
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
namespace spm
{
struct async_data
{
    hsa::AQLPacket*    packet{nullptr};
    spm_callback_data* callback_data{nullptr};
};

/**
 * @brief Async Handler for barrier=packet1 completion signal
 * Destroys the barrier-packet1 completion signal
 * Sets the dependency signal of barrier packet-2 to 0, so that barrier-packet2 can complete
 * This guarantees that SPM has been started before the dispatch
 **/
bool
AsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    auto* user_data = static_cast<async_data*>(data);
    auto* packet    = static_cast<hsa::SPMPacket*>(user_data->packet);
    auto* _data     = user_data->callback_data;

    if(_data->config_switch == true)
    {
        spm_get_controller()._agent_state_map.wlock([&](auto& map) {
            auto it = map.find(_data->dispatch_data.dispatch_info.agent_id.handle);
            ROCP_FATAL_IF(it == map.end())
                << "agent state map does not have an entnry for agent in async handler";
            auto& state_queue = it->second;
            if(state_queue.size() == 1)
                state_queue.front()->spm_packet->kfd_start();
            else
            {
                auto rel_pkt = std::move(state_queue.front()->spm_packet);
                rel_pkt->kfd_stop();
                state_queue.erase(state_queue.begin());
                ROCP_FATAL_IF(state_queue.empty())
                    << "agent state map has no entry for context switch async handler";
                state_queue.front()->spm_packet->kfd_start();
            }
        });
    }

    CHECK_NOTNULL(hsa::get_queue_controller())
        ->get_core_table()
        .hsa_signal_destroy_fn(packet->before_krn_barrier_pkt.at(0).completion_signal);
    CHECK_NOTNULL(hsa::get_queue_controller())
        ->get_core_table()
        .hsa_signal_store_screlease_fn(packet->before_krn_barrier_pkt.at(1).dep_signal[0], 0);
    spm_get_controller()._current_dispatch_data.wlock([&](auto& map) {
        auto agent_id        = _data->dispatch_data.dispatch_info.agent_id;
        map[agent_id.handle] = std::unique_ptr<spm_callback_data>(_data);
    });
    delete(user_data);
    return false;
}

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
    auto no_instrumentation = [&]() {
        auto ret_pkt = std::make_unique<rocprofiler::hsa::EmptyAQLPacket>();

        // If we have a SPM counter collection context but it is not enabled, we still might need
        // to add barrier packets to transition from serialized -> unserialized execution. This
        // transition is coordinated by the serializer.
        return ret_pkt;
    };

    if(!ctx || !ctx->dispatch_spm) return {nullptr, false};

    bool is_enabled = false;
    ctx->dispatch_spm->enabled.rlock([&](const auto& collect_ctx) { is_enabled = collect_ctx; });

    if(!is_enabled || !info->user_cb) return {no_instrumentation(), false};

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

    info->user_cb(&dispatch_data, &req_profile, user_data, info->callback_args);

    if(req_profile.handle == 0) return {nullptr, true};

    auto prof_config = spm_get_controller().get_profile_cfg(req_profile);
    CHECK(prof_config);

    std::unique_ptr<rocprofiler::hsa::AQLPacket> ret_pkt         = nullptr;
    auto                                         async_user_data = new async_data();
    async_user_data->callback_data                               = new spm_callback_data();

    async_user_data->callback_data->dispatch_data        = dispatch_data;
    async_user_data->callback_data->user_data            = user_data;
    async_user_data->callback_data->record_cb            = info->record_callback;
    async_user_data->callback_data->record_callback_args = info->record_callback_args;

    spm_get_controller()._agent_state_map.wlock([&](auto& map) {
        auto it = map.find(dispatch_data.dispatch_info.agent_id.handle);
        if(it == map.end())
        {
            std::unique_ptr<rocprofiler::hsa::SPMPacket> _pkt = nullptr;
            async_user_data->callback_data->config_switch     = true;
            _pkt = prof_config->pkt_generator->construct_packet(
                CHECK_NOTNULL(hsa::get_queue_controller())->get_core_table(),
                CHECK_NOTNULL(hsa::get_queue_controller())->get_ext_table());
            ret_pkt = std::make_unique<rocprofiler::hsa::SPMPacket>(*(_pkt.get()));
            map[dispatch_data.dispatch_info.agent_id.handle].push_back(
                std::make_unique<enqueue_dispatch_config_state>(prof_config->id, std::move(_pkt)));
        }
        else
        {
            auto& state_queue = it->second;
            if(state_queue.back()->config_id.handle == prof_config->id.handle)
            {
                ret_pkt = std::make_unique<rocprofiler::hsa::SPMPacket>(
                    *(state_queue.back()->spm_packet.get()));
            }
            else
            {
                std::unique_ptr<rocprofiler::hsa::SPMPacket> _pkt = nullptr;
                async_user_data->callback_data->config_switch     = true;
                _pkt = prof_config->pkt_generator->construct_packet(
                    CHECK_NOTNULL(hsa::get_queue_controller())->get_core_table(),
                    CHECK_NOTNULL(hsa::get_queue_controller())->get_ext_table());
                ret_pkt = std::make_unique<rocprofiler::hsa::SPMPacket>(*(_pkt.get()));
                state_queue.push_back(std::make_unique<enqueue_dispatch_config_state>(
                    prof_config->id, std::move(_pkt)));
            }
        }
    });

    if(!ret_pkt->empty)
    {
        auto* spm_pkt = dynamic_cast<hsa::SPMPacket*>(ret_pkt.get());
        // ROCP_FATAL_IF(pkt == nullptr)  << "NULL Packet returned from get spm packet: ";
        spm_pkt->clear();
        spm_pkt->populate_before();
        spm_pkt->populate_after();

        auto& signal_to_start_kfd    = spm_pkt->before_krn_barrier_pkt.at(0).completion_signal;
        auto& signal_kfd_has_started = spm_pkt->before_krn_barrier_pkt.at(1).dep_signal[0];

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

        async_user_data->packet = ret_pkt.get();
        auto status             = CHECK_NOTNULL(hsa::get_queue_controller())
                          ->get_ext_table()
                          .hsa_amd_signal_async_handler_fn(signal_to_start_kfd,
                                                           HSA_SIGNAL_CONDITION_EQ,
                                                           -1,
                                                           rocprofiler::spm::AsyncSignalHandler,
                                                           async_user_data);
        ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
            << "Error: hsa_amd_signal_async_handler failed with error code " << status
            << " :: " << hsa::get_hsa_status_string(status);
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
                 std::shared_ptr<hsa::Queue::queue_info_session_t>& /*ptr_session*/,
                 inst_pkt_t& pkts,
                 kernel_dispatch::profiling_time /*dispatch_time*/)
{
    CHECK(info && ctx);

    // Get the Profile Config

    for(auto& [aql_pkt, _] : pkts)
    {
        auto* pkt = dynamic_cast<hsa::SPMPacket*>(aql_pkt.get());
        if(!pkt) continue;
        auto rel_pkt = std::move(aql_pkt);
        ROCP_FATAL_IF(!pkt->sym.valid()) << "fatal";
        pkt->sym.spm_drain_counters_fn(pkt->handle);
        spm_get_controller()._current_dispatch_data.wlock([&](auto& map) {
            auto agent_id = rocprofiler::agent::get_rocprofiler_agent(pkt->hsa_agent)->id;
            auto it       = map.find(agent_id.handle);
            map.erase(it);
        });
        return;
    }
}

}  // namespace spm
}  // namespace rocprofiler
