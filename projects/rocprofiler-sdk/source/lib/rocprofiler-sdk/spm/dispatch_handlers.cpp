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
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

namespace rocprofiler
{
namespace spm
{

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
    auto callback_data                               = std::make_unique<spm_callback_data>();
    auto no_instrumentation = [&]() {
        auto ret_pkt = std::make_unique<rocprofiler::hsa::EmptyAQLPacket>();
        info->packet_return_map.wlock([&](auto& data) { data.emplace(ret_pkt.get(), nullptr); });
        callback_data->is_profiling = false;
        spm_get_controller()._callback_data.wlock([&](auto& map) {
            map[CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id.handle].push_back(std::move(callback_data));
        });
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

    if(req_profile.handle == 0) return {no_instrumentation(), true};
    
    auto prof_config = spm_get_controller().get_profile_cfg(req_profile);
    CHECK(prof_config);

    std::unique_ptr<rocprofiler::hsa::AQLPacket> ret_pkt         = nullptr;
     
    callback_data->dispatch_data        = dispatch_data;
    callback_data->user_data            = user_data;
    callback_data->record_cb            = info->record_callback;
    callback_data->record_callback_args = info->record_callback_args;
    callback_data->is_profiling            = true;

    spm_get_controller()._agent_state_map.wlock([&](auto& map) {
        auto it = map.find(dispatch_data.dispatch_info.agent_id.handle);
        if(it == map.end())
        {
            std::unique_ptr<rocprofiler::hsa::SPMPacket> _pkt = nullptr;
            callback_data->config_switch     = true;
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
                callback_data->config_switch     = true;
                _pkt = prof_config->pkt_generator->construct_packet(
                    CHECK_NOTNULL(hsa::get_queue_controller())->get_core_table(),
                    CHECK_NOTNULL(hsa::get_queue_controller())->get_ext_table());
                ret_pkt = std::make_unique<rocprofiler::hsa::SPMPacket>(*(_pkt.get()));
                state_queue.push_back(std::make_unique<enqueue_dispatch_config_state>(
                    prof_config->id, std::move(_pkt)));
            }
        }
    });

    info->packet_return_map.wlock([&](auto& data) { data.emplace(ret_pkt.get(), prof_config); });
    if(!ret_pkt->empty)
    {
        auto* spm_pkt = dynamic_cast<hsa::SPMPacket*>(ret_pkt.get());
        
        // ROCP_FATAL_IF(pkt == nullptr)  << "NULL Packet returned from get spm packet: ";
        spm_pkt->clear();
        spm_pkt->populate_before();
        spm_pkt->populate_after();
    }

    spm_get_controller()._callback_data.wlock([&](auto& map) {
            map[CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id.handle].push_back(std::move(callback_data));
    });
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
    std::shared_ptr<spm_counter_config> prof_config;
    // Get the Profile Config
    std::unique_ptr<rocprofiler::hsa::AQLPacket> ret_pkt = nullptr;
    info->packet_return_map.wlock([&](auto& data) {
        for(auto& [aql_pkt, _] : pkts)
        {
            const auto* profile = rocprofiler::common::get_val(data, aql_pkt.get());
           
            if(profile)
            {
                prof_config = *profile;
                auto* spm_pkt = dynamic_cast<hsa::SPMPacket*>(aql_pkt.get());
                spm_pkt->sym.spm_drain_counters_fn(spm_pkt->handle);
                data.erase(aql_pkt.get());
                ret_pkt = std::move(aql_pkt);
                return;
            }
        }
      
    });

    if(ret_pkt)
    {
      spm_get_controller()._callback_data.wlock([&](auto& map)
      {    
        auto* spm_pkt = dynamic_cast<hsa::SPMPacket*>(ret_pkt.get());
        auto agent_id = rocprofiler::agent::get_rocprofiler_agent(spm_pkt->hsa_agent)->id;
        auto it = map.find(agent_id.handle);
        auto current_dispatch = std::move(it->second.front());
        it->second.pop_front();
      });
    }
}


}  // namespace spm
}  // namespace rocprofiler
