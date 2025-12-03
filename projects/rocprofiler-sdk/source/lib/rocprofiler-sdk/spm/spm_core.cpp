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

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/spm/spm_core.hpp"
#include "lib/rocprofiler-sdk/spm/spm_dispatch_handlers.hpp"

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
/**
 * @brief The functions checks if the `ROCPROFILER_SPM_BETA_ENABLED` is set.
 * If so, it will enable SPM service. Otherwise, the API is reported
 * as not implemented.
 *
 * The SPM is in experimental phase .
   By enabling the `ROCPROFILER_SPM_BETA_ENABLED`,
 * user accepts all consequences of using early implementation of SPM API.
 */
bool
is_spm_explicitly_enabled()
{
    auto spm_sampling_enabled = rocprofiler::common::get_env("ROCPROFILER_SPM_BETA_ENABLED", false);

    if(!spm_sampling_enabled)
        ROCP_INFO << " SPM unavailable. The feature is implicitly disabled. "
                  << "To use it on a supported architecture, "
                  << "set ROCPROFILER_SPM_BETA_ENABLED=ON in the environment";

    return spm_sampling_enabled;
}

// Adds a counter collection profile to our global cache.
// Note: these profiles can be used across multiple contexts
//       and are independent of the context.
uint64_t
SpmCounterController::spm_add_profile(std::shared_ptr<spm_counter_config>&& config)
{
    static std::atomic<uint64_t> profile_val = 1;
    uint64_t                     ret         = 0;
    _configs.wlock([&](auto& data) {
        config->id = rocprofiler_spm_counter_config_id_t{.handle = profile_val};
        data.emplace(profile_val, std::move(config));
        ret = profile_val;
        profile_val++;
    });
    return ret;
}

void
SpmCounterController::spm_destroy_profile(uint64_t id)
{
    _configs.wlock([&](auto& data) { data.erase(id); });
}

std::shared_ptr<spm_counter_config>
SpmCounterController::get_profile_cfg(rocprofiler_spm_counter_config_id_t id)
{
    std::shared_ptr<spm_counter_config> cfg;
    _configs.rlock([&](const auto& map) { cfg = map.at(id.handle); });
    return cfg;
}

void
destroy_spm_counter_profile(uint64_t id)
{
    spm_get_controller().spm_destroy_profile(id);
}

SpmCounterController&
spm_get_controller()
{
    static auto* controller = rocprofiler::common::static_object<SpmCounterController>::construct();
    return *controller;
}

rocprofiler_status_t
spm_counter_callback_info::setup_spm_counter_config(std::shared_ptr<spm_counter_config>& profile)
{
    if(profile->pkt_generator)
    {
        return ROCPROFILER_STATUS_SUCCESS;
    }

    // Sets up the packet generator for the profile. This must be delayed until after HSA is loaded.
    // This call needs to be thread protected in that only one thread must be setting up profile at
    // the same time.

    auto& config           = *profile;
    auto  agent_name       = std::string(config.agent->name);
    profile->pkt_generator = std::make_unique<rocprofiler::aql::SPMPacketConstruct>(
        config.agent->id,
        std::vector<counters::Metric>{config.metrics.begin(), config.metrics.end()},
        config.sample_freq,
        config.buffer_size,
        config.timeout);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
spm_counter_callback_info::get_spm_packet(std::unique_ptr<rocprofiler::hsa::SPMPacket>& ret_pkt,
                                          std::shared_ptr<spm_counter_config>&          profile)
{
    rocprofiler_status_t status;
    status = spm_counter_callback_info::setup_spm_counter_config(profile);
    if(profile->packet && !profile->packet->empty && status == ROCPROFILER_STATUS_SUCCESS)
    {
        ret_pkt = std::make_unique<rocprofiler::hsa::SPMPacket>(*(profile->packet));
    }

    if(!ret_pkt)
    {
        // If we do not have a packet in the cache, create one.
        ret_pkt = profile->pkt_generator->construct_packet(
            CHECK_NOTNULL(hsa::get_queue_controller())->get_core_table(),
            CHECK_NOTNULL(hsa::get_queue_controller())->get_ext_table());

        profile->packet = std::make_unique<rocprofiler::hsa::SPMPacket>(*ret_pkt);
    };

    ret_pkt->record_cb            = record_callback;
    ret_pkt->record_callback_args = record_callback_args;

    ret_pkt->clear();

    packet_return_map.wlock([&](auto& data) { data.emplace(ret_pkt.get(), profile); });

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
create_spm_counter_profile(std::shared_ptr<spm_counter_config> config)
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    if(status = spm_counter_callback_info::setup_spm_counter_config(config);
       status != ROCPROFILER_STATUS_SUCCESS)
    {
        return status;
    }

    if(status = config->pkt_generator->can_collect(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        return status;
    }

    spm_get_controller().spm_add_profile(std::move(config));

    return status;
}

std::shared_ptr<spm_counter_config>
get_spm_counter_config(rocprofiler_spm_counter_config_id_t id)
{
    try
    {
        return spm_get_controller().get_profile_cfg(id);
    } catch(std::out_of_range&)
    {
        return nullptr;
    }
}

rocprofiler_status_t
configure_spm_dispatch(rocprofiler_context_id_t                       context_id,
                       rocprofiler_spm_dispatch_counting_service_cb_t callback,
                       void*                                          callback_data_args,
                       rocprofiler_spm_dispatch_counting_record_cb_t  record_callback,
                       void*                                          record_callback_args)
{
    return spm_get_controller().configure_dispatch(
        context_id, callback, callback_data_args, record_callback, record_callback_args);
}

rocprofiler_status_t
SpmCounterController::configure_dispatch(
    rocprofiler_context_id_t                       context_id,
    rocprofiler_spm_dispatch_counting_service_cb_t callback,
    void*                                          callback_args,
    rocprofiler_spm_dispatch_counting_record_cb_t  record_callback,
    void*                                          record_callback_args)
{
    auto* ctx_p = rocprofiler::context::get_mutable_registered_context(context_id);
    if(!ctx_p) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;

    auto& ctx = *ctx_p;

    // FIXME: Due to the clock gating issue, counter collection and PC sampling service
    // cannot coexist in the same context for now.
    if(ctx.pc_sampler) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
    if(ctx.counter_collection) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
    if(!ctx.dispatch_spm)
    {
        ctx.dispatch_spm =
            std::make_unique<rocprofiler::context::spm_dispatch_counter_collection_service>();
    }

    auto& cb = *ctx.dispatch_spm->callbacks.emplace_back(
        std::make_shared<SPM::spm_counter_callback_info>());

    cb.user_cb              = callback;
    cb.callback_args        = callback_args;
    cb.context              = context_id;
    cb.record_callback      = record_callback;
    cb.record_callback_args = record_callback_args;

    return ROCPROFILER_STATUS_SUCCESS;
}
void
start_context(const context::context* ctx)
{
    if(!ctx || !ctx->dispatch_spm) return;

    auto* controller = hsa::get_queue_controller();

    bool already_enabled = true;
    CHECK_NOTNULL(controller)->enable_serialization();
    ctx->dispatch_spm->enabled.wlock([&](auto& enabled) {
        if(enabled) return;
        already_enabled = false;
        enabled         = true;
    });

    if(!already_enabled)
    {
        for(auto& cb : ctx->dispatch_spm->callbacks)
        {
            if(cb->queue_id != rocprofiler::hsa::ClientID{-1}) return;
            // Insert our callbacks into HSA Interceptor. This
            // turns on counter instrumentation.

            cb->queue_id = controller->add_callback(
                std::nullopt,
                [=](const hsa::Queue&                                               q,
                    const hsa::rocprofiler_packet&                                  kern_pkt,
                    rocprofiler_kernel_id_t                                         kernel_id,
                    rocprofiler_dispatch_id_t                                       dispatch_id,
                    rocprofiler_user_data_t*                                        user_data,
                    const hsa::Queue::queue_info_session_t::external_corr_id_map_t& extern_corr_ids,
                    const context::correlation_id* correlation_id) {
                    return pre_kernel_call(ctx,
                                           cb,
                                           q,
                                           kern_pkt,
                                           kernel_id,
                                           dispatch_id,
                                           user_data,
                                           extern_corr_ids,
                                           correlation_id);
                },
                // Completion CB
                [=](const hsa::Queue& /* q */,
                    hsa::rocprofiler_packet /* kern_pkt */,
                    std::shared_ptr<hsa::Queue::queue_info_session_t>& session,
                    inst_pkt_t&                                        aql,
                    kernel_dispatch::profiling_time                    dispatch_time) {
                    post_kernel_call(ctx, cb, session, aql, dispatch_time);
                });
        }
    }
}

void
stop_context(const context::context* ctx)
{
    if(!ctx || !ctx->dispatch_spm) return;

    auto* controller = hsa::get_queue_controller();

    ctx->dispatch_spm->enabled.wlock([&](auto& enabled) {
        if(!enabled) return;
        enabled = false;
    });

    if(controller) controller->disable_serialization();
}

}  // namespace SPM

}  // namespace rocprofiler
