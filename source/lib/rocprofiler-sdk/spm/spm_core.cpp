// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#define CHECK_HSA(fn, message)                                                                     \
    {                                                                                              \
        auto _status = (fn);                                                                       \
        if(_status != HSA_STATUS_SUCCESS)                                                          \
        {                                                                                          \
            ROCP_ERROR << "HSA Err: " << _status << '\n';                                          \
            throw std::runtime_error(message);                                                     \
        }                                                                                          \
    }

namespace rocprofiler
{
namespace SPM
{
CoreApiTable&
get_core()
{
    static CoreApiTable api{};
    return api;
}

AmdExtTable&
get_ext()
{
    static AmdExtTable api{};
    return api;
}

common::Synchronized<std::optional<int64_t>> client;

SPMQueue::SPMQueue(spm_parameter_pack _params, const hsa::AgentCache& cache)
: Queue(cache.get_hsa_agent())
, params(std::move(_params))
{
    auto pool = hsa::SPMMemoryPool{cache, get_ext(), get_core().hsa_memory_copy_fn};
    aql::SPMPacketFactory factory(*CHECK_NOTNULL(cache.get_rocp_agent()), params, pool);
    this->packet = factory.construct();
}

SPMQueue::~SPMQueue() { std::unique_lock<std::mutex> lk(mut); }

std::unique_ptr<hsa::internal_queue::Signal>
SPMQueue::start()
{
    std::unique_lock<std::mutex> lk(mut);
    ROCP_FATAL_IF(!packet) << "SPM packet not initialized";

    packet->data_fn   = params.data_fn;
    packet->user_data = params.user_data;
    packet->kfd_start();
    return SubmitAndSignalLast(packet->before_krn_pkt);
}

SPMQueue::StopSignal::~StopSignal()
{
    this->signal = nullptr;
    if(packet) packet->kfd_stop();
}

std::unique_ptr<SPMQueue::StopSignal>
SPMQueue::stop()
{
    std::unique_lock<std::mutex> lk(mut);
    ROCP_FATAL_IF(!packet) << "SPM packet not initialized";
    return std::make_unique<StopSignal>(packet.get(), SubmitAndSignalLast(packet->after_krn_pkt));
}

void
SPMAgentManager::resource_init()
{
    auto rocp_agents = rocprofiler::agent::get_agents();

    std::unique_lock<std::mutex> lk(agent_mut);
    queues.clear();

    for(const auto* rocp_agent : rocp_agents)
    {
        auto id = rocp_agent->id;
        if(params.find(rocp_agent->id) == params.end()) continue;
        auto& pack = params.at(id);

        auto queue = std::make_unique<SPMQueue>(
            pack, *CHECK_NOTNULL(rocprofiler::agent::get_agent_cache(rocp_agent)));
        auto& desc = CHECK_NOTNULL(queue->packet.get())->desc;

        pack.data_fn(id, ROCPROFILER_SPM_RECORD_TYPE_SPM_DESC, &desc, pack.user_data);
        queues[id] = std::move(queue);
    }
}

void
SPMAgentManager::resource_deinit()
{
    std::unique_lock<std::mutex> lk(agent_mut);
    queues.clear();
}

rocprofiler_status_t
SPMAgentManager::start_context()
{
    std::vector<std::unique_ptr<hsa::internal_queue::Signal>> wait_list{};
    std::unique_lock<std::mutex>                              lk(agent_mut);

    for(auto& [_, queue] : queues)
        wait_list.emplace_back(queue->start());

    return queues.empty() ? ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND : ROCPROFILER_STATUS_SUCCESS;
}

void
SPMAgentManager::stop_context()
{
    std::unique_lock<std::mutex> lk(agent_mut);

    {
        std::vector<std::unique_ptr<SPMQueue::StopSignal>> wait_list{};
        for(auto& [_, queue] : queues)
            wait_list.emplace_back(queue->stop());
    }

    for(auto& [id, queue] : queues)
    {
        const auto& pack = queue->params;
        pack.data_fn(id, ROCPROFILER_SPM_RECORD_TYPE_AGENT_DATA_END, nullptr, pack.user_data);
    }
}

void
SPMDispatchManager::resource_init()
{
    auto rocp_agents = rocprofiler::agent::get_agents();

    auto lk = std::unique_lock{agent_mut};

    for(const auto* rocp_agent : rocp_agents)
    {
        auto id = rocp_agent->id;
        if(params.find(id) == params.end()) continue;
        auto& pack = params.at(id);

        auto pool =
            hsa::SPMMemoryPool{*CHECK_NOTNULL(rocprofiler::agent::get_agent_cache(rocp_agent)),
                               get_ext(),
                               get_core().hsa_memory_copy_fn};
        aql::SPMPacketFactory factory(*rocp_agent, pack, pool);
        auto                  packet = factory.construct();

        pack.data_fn(id, ROCPROFILER_SPM_RECORD_TYPE_SPM_DESC, &packet->desc, pack.user_data);
        packets.push_back({id, std::move(packet)});
    }
}

void
SPMDispatchManager::resource_deinit()
{
    auto lk = std::unique_lock{agent_mut};
    packets.clear();
}

void
SPMDispatchManager::start_context()
{
    using corr_id_map_t = hsa::Queue::queue_info_session_t::external_corr_id_map_t;
    CHECK_NOTNULL(hsa::get_queue_controller())->enable_serialization();

    // Only one thread should be attempting to enable/disable this context
    client.wlock([&](auto& client_id) {
        if(client_id) return;

        client_id =
            CHECK_NOTNULL(hsa::get_queue_controller())
                ->add_callback(
                    std::nullopt,
                    [=](const hsa::Queue& q,
                        const hsa::rocprofiler_packet& /* kern_pkt */,
                        rocprofiler_kernel_id_t   kernel_id,
                        rocprofiler_dispatch_id_t dispatch_id,
                        rocprofiler_user_data_t*  user_data,
                        const corr_id_map_t& /* extern_corr_ids */,
                        const context::correlation_id* corr_id) {
                        return this->pre_kernel_call(q, kernel_id, dispatch_id, user_data, corr_id);
                    },
                    [=](const hsa::Queue& /* q */,
                        hsa::rocprofiler_packet /* kern_pkt */,
                        std::shared_ptr<hsa::Queue::queue_info_session_t>& session,
                        inst_pkt_t&                                        aql,
                        kernel_dispatch::profiling_time) {
                        this->post_kernel_call(aql, *session);
                    });
    });

    bActiveCtx.store(true);
}

void
SPMDispatchManager::stop_context()
{
    bActiveCtx.store(false);
    CHECK_NOTNULL(hsa::get_queue_controller())->disable_serialization();

    client.wlock([&](auto& client_id) {
        if(!client_id) return;

        // Remove our callbacks from HSA's queue controller
        CHECK_NOTNULL(hsa::get_queue_controller())->remove_callback(*client_id);
        client_id = std::nullopt;
    });
}

bool
AsyncSignalHandler(hsa_signal_value_t /*signal_v*/, void* data)
{
    auto* packet     = CHECK_NOTNULL(static_cast<hsa::SPMPacket*>(data));
    auto& before_krn = packet->before_krn_pkt;

    if(before_krn.size() < 2)
    {
        ROCP_ERROR << "Invalid before_krn packet" << std::endl;
        return true;
    }

    packet->kfd_start();

    get_core().hsa_signal_destroy_fn(before_krn.at(0).barrier_and.completion_signal);
    get_core().hsa_signal_store_screlease_fn(before_krn.at(1).barrier_and.dep_signal[0], 0);
    return false;
}

/**
 * Callback we get from HSA interceptor when a kernel packet is being enqueued.
 * We return an AQLPacket containing the start/stop/read packets for injection.
 */
hsa::Queue::pkt_and_serialize_t
SPMDispatchManager::pre_kernel_call(const hsa::Queue&         queue,
                                    rocprofiler_kernel_id_t   kernel_id,
                                    rocprofiler_dispatch_id_t dispatch_id,
                                    rocprofiler_user_data_t*  user_data,
                                    const context::correlation_id* /* corr_id */)
{
    auto agent_id = CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id;

    auto packet = [&]() {
        auto lk = std::shared_lock{agent_mut};
        for(auto& [id, pkt] : packets)
            if(id == agent_id && pkt != nullptr) return std::make_unique<hsa::SPMPacket>(*pkt);
        return std::unique_ptr<hsa::SPMPacket>{nullptr};
    }();

    if(!packet || !bActiveCtx) return {nullptr, false};

    auto& param = params.at(agent_id);

    auto control_flags = param.dispatch_fn(
        agent_id, queue.get_id(), kernel_id, dispatch_id, param.config_userdata, user_data);

    if(control_flags == 0) return {nullptr, true};

    packet->clear();
    packet->populate_before();
    packet->populate_after();

    ROCP_FATAL_IF(packet->before_krn_pkt.size() < 3) << "SPM Requires at least 3 packets";

    auto& signal_to_start_kfd    = packet->before_krn_pkt.at(0).barrier_and.completion_signal;
    auto& signal_kfd_has_started = packet->before_krn_pkt.at(1).barrier_and.dep_signal[0];

    queue.create_signal(0, &signal_to_start_kfd);
    queue.create_signal(0, &signal_kfd_has_started);

    get_core().hsa_signal_store_screlease_fn(signal_kfd_has_started, -1);
    get_core().hsa_signal_store_screlease_fn(signal_to_start_kfd, 0);

    auto status = get_ext().hsa_amd_signal_async_handler_fn(
        signal_to_start_kfd, HSA_SIGNAL_CONDITION_EQ, -1, AsyncSignalHandler, packet.get());
    ROCP_FATAL_IF(status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK)
        << "Error: hsa_amd_signal_async_handler failed with error code " << status
        << " :: " << hsa::get_hsa_status_string(status);

    packet->user_data = *user_data;
    packet->data_fn   = param.data_fn;
    return {std::move(packet), true};
}

void
SPMDispatchManager::post_kernel_call(SPMDispatchManager::inst_pkt_t& aql,
                                     const hsa::Queue::queue_info_session_t& /* session */)
{
    auto _lk = std::shared_lock{agent_mut};
    for(auto& aql_pkt : aql)
    {
        auto* pkt = dynamic_cast<hsa::SPMPacket*>(aql_pkt.first.get());
        if(!pkt) continue;

        get_core().hsa_signal_destroy_fn(pkt->before_krn_pkt.at(1).barrier_and.dep_signal[0]);
    }
}

void
initialize(HsaApiTable* table)
{
    hsa::internal_queue::initialize(table);

    assert(table->core_ && table->amd_ext_);
    get_core() = *table->core_;
    get_ext()  = *table->amd_ext_;

    for(auto& ctx : context::get_registered_contexts())
    {
        if(ctx->agent_spm) ctx->agent_spm->resource_init();
        if(ctx->dispatch_spm) ctx->dispatch_spm->resource_init();
    }
}

void
finalize()
{
    for(auto& ctx : context::get_registered_contexts())
    {
        if(ctx->agent_spm) ctx->agent_spm->resource_deinit();
        if(ctx->dispatch_spm) ctx->dispatch_spm->resource_deinit();
    }
}

}  // namespace SPM

}  // namespace rocprofiler
