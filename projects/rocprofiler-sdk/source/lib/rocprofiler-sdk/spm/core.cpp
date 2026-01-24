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

#include "lib/rocprofiler-sdk/spm/core.hpp"
#include "lib/common/container/stable_vector.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/buffer.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/internal_threading.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/spm/dispatch_handlers.hpp"

#include <hsa/hsa_api_trace.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/rocprofiler.h>

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
namespace spm
{

    /**
 * Adds a counter collection profile to our global cache.
 * Note: these profiles can be used across multiple contexts and are independent of the context.
 * Note: these profiles are per agent
 * Assigns the config id and increments the monotonic counter.
 */
void
SpmCounterController::spm_add_profile(std::shared_ptr<spm_counter_config>&& config)
{
    static std::atomic<uint64_t> profile_val = 1;
    _configs.wlock([&](auto& data) {
        config->id = rocprofiler_spm_counter_config_id_t{.handle = profile_val};
        data.emplace(profile_val, std::move(config));
        profile_val++;
    });
}

void
SpmCounterController::state_map_fini()
{
    spm_get_controller()._agent_state_map.wlock([&](auto& map) {
        for(auto& [agent, state_queue] : map)
        {
            if(!state_queue.empty())
            {
                if(state_queue.size() != 1) ROCP_WARNING << "state queue greater than 1";
                auto rel_pkt = std::move(state_queue.front()->spm_packet);
                dynamic_cast<hsa::SPMPacket*>(rel_pkt.get())->kfd_stop();
                // state_queue.erase(state_queue.begin());

                state_queue.clear();
            }
        }
    });
}
/**
 * @brief Removes the profile entry from the global cache
 */
void
SpmCounterController::spm_destroy_profile(uint64_t id)
{
    _configs.wlock([&](auto& data) { data.erase(id); });
}

/**
 * @brief Queries the global cache for the config using config id
 */
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
    return *CHECK_NOTNULL(controller);
}

/**
 * @brief sets up packet generator in spm counter config
 * spm counter config  is instantiated for each config created by the user
 * Pkt generator is an instance of SPMPacketConstruct that contains SPM parameters and counters
 * Pkt generator is needed to construct aqlprofile packet for SPM
 */
rocprofiler_status_t
spm_counter_callback_info::setup_spm_counter_config(std::shared_ptr<spm_counter_config>& profile)
{
    /*
     * If the profile already has pkt generator set up then it returns success
     */
    if(profile->pkt_generator)
    {
        return ROCPROFILER_STATUS_SUCCESS;
    }

    // Sets up the packet generator for the profile.

    auto& config           = *profile;
    profile->pkt_generator = std::make_unique<rocprofiler::aql::SPMPacketConstruct>(
        config.agent->id,
        std::vector<counters::Metric>{config.metrics.begin(), config.metrics.end()},
        config.sample_freq,
        config.buffer_size,
        config.timeout);
    return ROCPROFILER_STATUS_SUCCESS;
}

void
state_map_fini()
{
    spm_get_controller().state_map_fini();
}
/** @brief  Creates spm the counter config
 * Checks if the input counters does not exceed hardware limit
 * Adds the config to configs cache
 */
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

/** @brief  Configures SPM dispatch for the context
 * Checks for conflicting services
 * Instantiates spm_dispatch_counter_collection_service
 */

rocprofiler_status_t
configure_callback_spm_dispatch(rocprofiler_context_id_t                       context_id,
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
        ctx.dispatch_spm->callback = std::make_shared<spm::spm_counter_callback_info>();
    }

    auto& cb                 = ctx.dispatch_spm->callback;
    cb->user_cb              = callback;
    cb->callback_args        = callback_args;
    cb->context              = context_id;
    cb->record_callback      = record_callback;
    cb->record_callback_args = record_callback_args;
    cb->internal_context     = ctx_p;

    return ROCPROFILER_STATUS_SUCCESS;
}

/** @brief start SPM dispatch context
 * Enables serialization
 * Returns if callback has already been added by checking the queue id
 * Adds a pre kernel and a post kernel callback
 * Enabled flag is used to check if context has already been enabled
 */

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
        auto& cb = ctx->dispatch_spm->callback;
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
                const context::correlation_id*                                  correlation_id) {
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

/** @brief stop SPM dispatch context
 * Disables serialization
 * Sets Enabled flag to false
 */

void
stop_context(const context::context* ctx)
{
    if(!ctx || !ctx->dispatch_spm) return;

    auto* controller = hsa::get_queue_controller();

    ctx->dispatch_spm->enabled.wlock([&](auto& enabled) {
        if(!enabled) return;
        enabled  = false;
        auto& cb = ctx->dispatch_spm->callback;
        if(!cb->queue_id) return;
        // Remove our callbacks from HSA's queue controller
        controller->remove_callback(cb->queue_id);
    });
    if(controller) controller->disable_serialization();
}

}  // namespace spm

}  // namespace rocprofiler
