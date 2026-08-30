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
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/metrics.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/spm/dispatch_handlers.hpp"

#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
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
 * @brief looks into the config's packet cache to reuse the packet
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

    // Scope serialization to the agents this context collects on. An empty set still means
    // every agent, so an unrestricted context serializes the whole machine as before.
    CHECK_NOTNULL(controller)->enable_serialization(ctx->dispatch_spm->agents);
    ctx->dispatch_spm->enabled.wlock([&](auto& enabled) {
        if(enabled) return;
        enabled = true;
    });

    // SPM no longer registers a per-queue callback with the queue controller; the HSA write
    // interceptor calls spm::write_hook / signal_completion_hook directly (see hsa/queue.cpp).
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
        // Drain in-flight dispatches, then disable serialization. The review of #8887 asked for
        // this sync to be kept and for SPM to stay visible to the enter hook and to serialization
        // through a "draining" window until both this and disable_serialization() complete.
        //
        // The visibility half comes from ordering rather than a separate flag:
        // context::stop_context calls this function while the context is still in the active list,
        // so for the duration of the drain the enter hook still reaches pre_kernel_call, whose
        // disabled path deliberately returns serialize=true to coordinate the
        // serialized->unserialized transition. Only once this function returns does the active slot
        // get cleared. Introducing a draining flag as well would duplicate that guarantee, but the
        // ordering is load-bearing: if the slot were cleared first, the enter hook would go blind
        // before disable_serialization() ran and a dispatch could be submitted unserialized while
        // the serializer was still enabled.
        //
        // A removal of this sync was considered on the grounds that its two original purposes --
        // avoiding dangling callback pointers once the per-queue callbacks were unregistered, and
        // letting in-flight dispatches complete -- are both addressed by the migration and by
        // routing completions over registered contexts. The first is genuinely gone. The second is
        // weaker than it looks: provenance routing ensures a completion that arrives is delivered,
        // whereas the drain bounds when completions arrive at all, which is what the rest of the
        // teardown depends on. Keeping it.
        hsa::queue_controller_sync();
        controller->disable_serialization(ctx->dispatch_spm->agents);
        // No per-queue callback to remove; spm::write_hook no-ops once dispatch_spm is
        // disabled above.
    }
}

rocprofiler_status_t
set_dispatch_agents(rocprofiler_context_id_t      context_id,
                    const rocprofiler_agent_id_t* agents,
                    size_t                        num_agents)
{
    if(num_agents > 0 && agents == nullptr) return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;

    auto* ctx_p = context::get_mutable_registered_context(context_id);
    if(!ctx_p) return ROCPROFILER_STATUS_ERROR_CONTEXT_INVALID;
    if(!ctx_p->dispatch_spm) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    for(const auto* itr : context::get_active_contexts())
    {
        if(itr && itr->context_idx == ctx_p->context_idx)
            return ROCPROFILER_STATUS_ERROR_CONFIGURATION_LOCKED;
    }

    auto selected = std::unordered_set<rocprofiler_agent_id_t>{};
    selected.reserve(num_agents);
    for(size_t i = 0; i < num_agents; ++i)
    {
        const auto* agent = rocprofiler::agent::get_agent(agents[i]);
        if(!agent || agent->type != ROCPROFILER_AGENT_TYPE_GPU)
            return ROCPROFILER_STATUS_ERROR_AGENT_NOT_FOUND;
        selected.emplace(agents[i]);
    }

    ctx_p->dispatch_spm->agents = std::move(selected);

    return ROCPROFILER_STATUS_SUCCESS;
}

}  // namespace spm

}  // namespace rocprofiler
