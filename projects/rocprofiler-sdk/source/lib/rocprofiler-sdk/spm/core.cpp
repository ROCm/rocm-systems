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

#include "lib/rocprofiler-sdk/spm/core.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/spm/dispatch_handlers.hpp"

#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace rocprofiler
{
namespace spm
{
/**
 *This is a singleton class with lazy initialization
 */
class SpmCounterController
{
public:
    SpmCounterController() = default;
    // Adds a counter collection profile to our global cache.
    // Note: these profiles can be used across multiple contexts
    //       and are independent of the context.
    void spm_add_profile(std::shared_ptr<spm_counter_config>&& config);

    rocprofiler_status_t spm_destroy_profile(rocprofiler_counter_config_id_t id);

    std::shared_ptr<spm_counter_config> get_profile_cfg(rocprofiler_counter_config_id_t id);

private:
    // Cache to contain the map of config id handle to spm counter config
    common::Synchronized<
        std::unordered_map<rocprofiler_counter_config_id_t, std::shared_ptr<spm_counter_config>>>
        _configs;
};

SpmCounterController&
spm_get_controller()
{
    static auto* controller = rocprofiler::common::static_object<SpmCounterController>::construct();
    return *CHECK_NOTNULL(controller);
}

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

/**
 * Adds a counter collection profile to our global cache.
 * Note: these profiles can be used across multiple contexts and are independent of the context.
 * Note: these profiles are per agent
 * Assigns the config id and increments the monotonic counter.
 */
void
SpmCounterController::spm_add_profile(std::shared_ptr<spm_counter_config>&& config)
{
    // Offset from PMC counter IDs (which start at 1) to avoid config ID collision
    // since both share rocprofiler_counter_config_id_t
    static std::atomic<uint64_t> profile_val{1};

    _configs.wlock([&](auto& data) {
        config->id = rocprofiler_counter_config_id_t{.handle = profile_val};
        data.emplace(config->id, std::move(config));
        profile_val++;
    });
}

/**
 * @brief Removes the profile entry from the global cache
 */
rocprofiler_status_t
SpmCounterController::spm_destroy_profile(rocprofiler_counter_config_id_t id)
{
    return _configs.wlock([&](auto& data) {
        auto itr = data.find(id);
        if(itr == data.end()) return ROCPROFILER_STATUS_ERROR_PROFILE_NOT_FOUND;
        if(data.erase(id) != 1) return ROCPROFILER_STATUS_ERROR;
        return ROCPROFILER_STATUS_SUCCESS;
    });
}

/**
 * @brief Queries the global cache for the config using config id
 */
std::shared_ptr<spm_counter_config>
SpmCounterController::get_profile_cfg(rocprofiler_counter_config_id_t id)
{
    std::shared_ptr<spm_counter_config> cfg = nullptr;
    _configs.rlock([&](const auto& map) {
        auto it = map.find(id);
        if(it != map.end()) cfg = it->second;
    });
    return cfg;
}

rocprofiler_status_t
destroy_spm_counter_profile(rocprofiler_counter_config_id_t id)
{
    return spm_get_controller().spm_destroy_profile(id);
}

/**
 * @brief looks into the config's packet cache to re-use the packet
 * If not, constructs the packet using packet generator
 * updates packet_return map
 */
rocprofiler_status_t
get_spm_packet(std::unique_ptr<rocprofiler::hsa::AQLPacket>& ret_pkt,
               std::shared_ptr<spm_counter_config>&          profile)
{
    profile->packets.wlock([&](auto& pkt_vector) {
        if(!pkt_vector.empty())
        {
            ret_pkt = std::move(pkt_vector.back());
            pkt_vector.pop_back();
        }
    });

    if(!ret_pkt)
    {
        // If we do not have a packet in the cache, create one.
        ret_pkt = rocprofiler::aql::spm_construct_packet(
            CHECK_NOTNULL(profile->agent)->id,
            std::vector<counters::Metric>{profile->metrics.begin(), profile->metrics.end()},
            profile->spm_parameters);
        if(!ret_pkt)
        {
            ROCP_ERROR << "SPM packet construction failed";
            return ROCPROFILER_STATUS_ERROR;
        }
    };

    return ROCPROFILER_STATUS_SUCCESS;
}

/** @brief  Creates spm the counter config
 * Checks if the input counters does not exceed hardware limit
 * Adds the config to configs cache
 */
rocprofiler_status_t
create_spm_counter_profile(std::shared_ptr<spm_counter_config> config)
{
    auto status = ROCPROFILER_STATUS_SUCCESS;
    if(status = rocprofiler::aql::spm_can_collect(config->agent->id, config->metrics);
       status != ROCPROFILER_STATUS_SUCCESS)
    {
        return status;
    }

    spm_get_controller().spm_add_profile(std::move(config));

    return status;
}

std::shared_ptr<spm_counter_config>
get_spm_counter_config(rocprofiler_counter_config_id_t id)
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
    if(ctx.dispatch_counter_collection) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
    if(ctx.device_counter_collection) return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
    if(!ctx.dispatch_spm)
        ctx.dispatch_spm =
            std::make_unique<rocprofiler::context::spm_dispatch_counter_collection_service>();
    auto& cb = *ctx.dispatch_spm->callbacks.emplace_back(
        std::make_shared<rocprofiler::spm::spm_counter_callback_info>());

    cb.user_cb              = callback;
    cb.callback_args        = callback_args;
    cb.context              = context_id;
    cb.record_callback      = record_callback;
    cb.record_callback_args = record_callback_args;
    cb.internal_context     = ctx_p;

    return ROCPROFILER_STATUS_SUCCESS;
}

/** @brief start SPM dispatch context
 * Enables serialization
 * Returns if callback has already been added by checking the queue id
 * Adds a pre kernel and a post kernel callback
 * Enabled flag is used to check if context has already been enabled
 */

rocprofiler_status_t
start_context(const context::context* ctx)
{
    if(!ctx || !ctx->dispatch_spm) return ROCPROFILER_STATUS_ERROR;

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
        // Insert our callbacks into HSA Interceptor. This
        // turns on counter instrumentation.
        for(auto& cb : ctx->dispatch_spm->callbacks)
        {
            if(cb->queue_id != rocprofiler::hsa::ClientID{-1}) continue;
            cb->queue_id = controller->add_callback(
                std::nullopt,
                hsa::queue_callbacks_t{
                    .batch_packets = []() { return false; },
                    .write_interceptor =
                        [=](const hsa::Queue&              q,
                            const hsa::rocprofiler_packet& kern_pkt,
                            rocprofiler_kernel_id_t        kernel_id,
                            rocprofiler_dispatch_id_t      dispatch_id,
                            rocprofiler_user_data_t*       user_data,
                            const hsa::queue_info_session_t::external_corr_id_map_t&
                                                           extern_corr_ids,
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
                    .signal_completion =
                        [=](const hsa::Queue& /* q */,
                            const hsa::rocprofiler_packet& /* kern_pkt */,
                            std::shared_ptr<hsa::queue_info_session_t>& session,
                            hsa::packet_data_t& /* pkt_data */,
                            inst_pkt_t&                     aql,
                            kernel_dispatch::profiling_time dispatch_time) {
                            post_kernel_call(ctx, cb, session, aql, dispatch_time);
                        }});
        }
    }

    return ROCPROFILER_STATUS_SUCCESS;
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
        enabled = false;
    });

    if(controller)
    {
        hsa::queue_controller_sync();
        controller->disable_serialization();
        for(auto& cb : ctx->dispatch_spm->callbacks)
        {
            if(cb->queue_id != rocprofiler::hsa::ClientID{-1})
            {
                controller->remove_callback(cb->queue_id);
                cb->queue_id = rocprofiler::hsa::ClientID{-1};
            }
        }
    }
}

}  // namespace spm

}  // namespace rocprofiler
