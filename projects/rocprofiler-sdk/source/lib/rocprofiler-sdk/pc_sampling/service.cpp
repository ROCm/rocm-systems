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

#include "lib/rocprofiler-sdk/pc_sampling/service.hpp"
#include "lib/rocprofiler-sdk/pc_sampling/defines.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/common/logging.hpp"
#    include "lib/common/static_object.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/hsa_adapter.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/utils.hpp"

#    include <optional>

namespace rocprofiler
{
namespace pc_sampling
{
using hsa_initialized_t = std::atomic<bool>;

hsa_initialized_t&
is_hsa_initialized()
{
    static auto _v = hsa_initialized_t{false};
    return _v;
}

common::Synchronized<global_pc_sampling_sessions_map_t>&
get_global_pc_sampling_sessions()
{
    static auto*& _v =
        common::static_object<common::Synchronized<global_pc_sampling_sessions_map_t>>::construct();
    return *_v;
}

rocprofiler_status_t
start_service(const context::context* ctx)
{
    auto* service = ctx->pc_sampler.get();

    // CAS to mark service as enabled
    bool expected = false;
    if(!service->enabled.compare_exchange_strong(expected, true))
    {
        // Service already started
        // TODO: Consider adding ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_STARTED
        return ROCPROFILER_STATUS_ERROR;
    }

    if(is_hsa_initialized().load())
    {
        hsa::pc_sampling_service_start(service);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
stop_service(const context::context* ctx)
{
    auto* service = ctx->pc_sampler.get();

    // CAS to mark service as disabled
    bool expected = true;
    if(!service->enabled.compare_exchange_strong(expected, false))
    {
        // Service not started
        // TODO: Consider adding ROCPROFILER_STATUS_ERROR_SERVICE_NOT_STARTED
        return ROCPROFILER_STATUS_ERROR;
    }

    if(is_hsa_initialized().load())
    {
        hsa::pc_sampling_service_stop(service);
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

void
post_hsa_init_start_active_service()
{
    // Called as part of the registration of the HSA table
    if(is_hsa_initialized().load())
    {
        // If there is a guarantee that the `rocprofiler_set_api_table`
        // can be called only once for the HSA, then this condition is redundant.
        return;
    }

    // Check if any PC sampling is configured
    {
        bool is_empty = get_global_pc_sampling_sessions().rlock(
            [](const auto& sessions) { return sessions.empty(); });
        if(is_empty) return;  // No PC sampling configured
    }

    // Configure PC sampling on the ROCr level for all services (only once)
    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        // Collect all unique services from the global sessions map
        std::unordered_set<context::pc_sampling_service*> services_to_configure;
        get_global_pc_sampling_sessions().rlock([&services_to_configure](const auto& sessions) {
            for(const auto& [_, agent_session] : sessions)
            {
                auto* ctx = rocprofiler::context::get_registered_context(agent_session->context_id);
                if(ctx && ctx->pc_sampler)
                {
                    services_to_configure.insert(ctx->pc_sampler.get());
                }
            }
        });

        // Configure each service on the ROCr level
        for(auto* service : services_to_configure)
        {
            hsa::pc_sampling_service_finish_configuration(service);
        }

        // Mark HSA as initialized
        is_hsa_initialized().store(true);

        // Start any services that are already enabled
        for(auto* service : services_to_configure)
        {
            if(service->enabled.load())
            {
                hsa::pc_sampling_service_start(service);
            }
        }
    });
}

rocprofiler_status_t
configure_pc_sampling_service(context::context*                ctx,
                              const rocprofiler_agent_t*       agent,
                              rocprofiler_pc_sampling_method_t method,
                              rocprofiler_pc_sampling_unit_t   unit,
                              uint64_t                         interval,
                              rocprofiler_buffer_id_t          buffer_id)
{
    // FIXME: PC Sampling cannot be used simultaneously with counter collection.
    // PC sampling requires clock gating to be disabled on MI2xx and MI3xx,
    // otherwise a weird GPU hang might appear and a machine must be rebooted.
    // Current implementation of (dispatch) counter collection service assumes disabling
    // the clock gating before dispatching a kernel and reenabling the clock gating
    // after kernel completion. Consequently, if PC sampling is active, (dispatch)
    // counter collection service can enable clock gating and hang might appear.
    // As a workaround, PC sampling and (dispatch) counter collection service
    // cannot coexist in the same context.
    if(ctx->dispatch_counter_collection || ctx->device_counter_collection)
    {
        return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
    }

    // Check if this agent is already configured by another context. If so, report an error.
    // Otherwise, register the session.
    return get_global_pc_sampling_sessions().wlock([&](auto& sessions) -> rocprofiler_status_t {
        if(auto it = sessions.find(agent->id); it != sessions.end())
        {
            // Agent already configured by another context
            return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
        }

        // calling KFD to check if the configuration is actually supported at the moment
        uint32_t ioctl_pcs_id;
        auto ioctl_status = ioctl::ioctl_pcs_create(agent, method, unit, interval, &ioctl_pcs_id);
        if(ioctl_status != ROCPROFILER_STATUS_SUCCESS) return ioctl_status;

        // Create a new session for this agent
        auto session = std::make_shared<PCSAgentSession>();

        // Fully initialize the session under the lock before making it visible
        session->context_id   = rocprofiler_context_id_t{.handle = ctx->context_idx};
        session->client_idx   = ctx->client_idx;
        session->agent        = agent;
        session->method       = method;
        session->unit         = unit;
        session->interval     = interval;
        session->buffer_id    = buffer_id;
        session->ioctl_pcs_id = ioctl_pcs_id;
        session->parser       = std::make_unique<PCSamplingParserContext>();
        session->cid_manager  = std::make_unique<PCSCIDManager>(session->parser.get());

        // Register session in global map
        sessions[agent->id] = session;

        // Initialize the PC sampler service for this context if needed
        if(!ctx->pc_sampler)
        {
            ctx->pc_sampler = std::make_unique<context::pc_sampling_service>();
        }

        // Register session in context map (shared ownership)
        // Keeping this under the same lock ensures atomicity of the entire operation
        ctx->pc_sampler->agent_sessions[agent->id] = session;

        // Note: This log reflects successful session creation and is intentionally
        // reported at ROCP_INFO (it was previously logged at ROCP_ERROR, which
        // was misleading for a non-error code path).
        ROCP_INFO << "PC sampling session with IOCTL id: " << session->ioctl_pcs_id
                  << " has been created!\n";

        return ROCPROFILER_STATUS_SUCCESS;
    });
}

bool
is_pc_sample_service_configured(rocprofiler_agent_id_t agent_id)
{
    return get_global_pc_sampling_sessions().rlock([agent_id](const auto& sessions) {
        // If the agent_id is in the global sessions map,
        // then the PC sampling service is configured on this agent.
        return sessions.find(agent_id) != sessions.end();
    });
}

PCSAgentSession*
get_agent_session(rocprofiler_agent_id_t agent_id)
{
    return get_global_pc_sampling_sessions().rlock(
        [agent_id](const auto& sessions) -> PCSAgentSession* {
            auto it = sessions.find(agent_id);
            if(it != sessions.end())
            {
                return it->second.get();
            }
            return nullptr;
        });
}

rocprofiler_status_t
flush_internal_agent_buffers(rocprofiler_buffer_id_t buffer_id)
{
    // checking if the buffer is registered
    auto const* buff = rocprofiler::buffer::get_buffer(buffer_id);
    if(!buff) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    // Checking if the context is registered
    const auto* ctx = rocprofiler::context::get_registered_context(
        rocprofiler_context_id_t{.handle = buff->context_id});
    if(!ctx) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;

    // To prevent a weird synthetic case where a client spawns multiple threads
    // inside tool_init where some of the threads call rocprofiler_flush_buffer
    // and other rocprofiler_configure_pc_sampling_service,
    // we're executing the following code under the read lock.
    // If we found that this is too restrictive for performance,
    // then we can remove the lock, as the case we explained above
    // sounds synthetic.
    auto* service = ctx->pc_sampler.get();
    if(service)
    {
        return get_global_pc_sampling_sessions().rlock([&](const auto& /*sessions*/) {
            rocprofiler_status_t status = ROCPROFILER_STATUS_SUCCESS;
            // The context `ctx` (that holds the buffer with `buffer_id`)
            // is the one containing PC sampling service.
            // The HSA interception table is registered.
            for(const auto& [_, agent_session] : service->agent_sessions)
            {
                // Find the agent that fills the buffer with `buffer_id`.
                // To prevent a weird case where one client tries emptying the buffer
                // of another client, ensure that client_idx of the agent_session
                // matches the client_idx of ctx.
                if(agent_session->buffer_id.handle == buffer_id.handle &&
                   agent_session->client_idx == ctx->client_idx)
                {
                    // Flush internal PC sampling buffers filled by the agent
                    // NOTE: one rocprofiler-SDK PC sampling buffer can be tied
                    // to multiple agent (agent sessions).
                    status = hsa::flush_internal_agent_buffers(agent_session.get());
                    if(status != ROCPROFILER_STATUS_SUCCESS) return status;
                }
            }
            return status;
        });
    }

    // PC sampling service not configured.
    return ROCPROFILER_STATUS_SUCCESS;
}

/**
 * @brief Flushes internal PC sampling buffers for agents.
 *
 * Loops over all agents that have PC sampling service configured and drains their
 * internal HSA buffers by flushing them to the SDK buffers.
 *
 * @param client_id Optional client ID to filter which agents to flush.
 * - If provided: Only flushes buffers for agents owned by this client.
 *   This prevents flushing buffers belonging to other clients' PC sampling sessions.
 * - If omitted (std::nullopt): Flushes buffers for all agents regardless of client
 *   ownership. Used during finalization to drain all remaining data.
 *
 * @return ROCPROFILER_STATUS_SUCCESS if successful, ROCPROFILER_STATUS_ERROR if no sessions exist,
 *         or the status of the last failed flush operation.
 *
 * @note One SDK buffer can consume data from multiple agents (multiple HSA runtime buffers).
 */
rocprofiler_status_t
flush_all_agents_buffers(std::optional<rocprofiler_client_id_t> client_id = std::nullopt)
{
    return get_global_pc_sampling_sessions().rlock(
        [&](const auto& sessions) -> rocprofiler_status_t {
            if(sessions.empty()) return ROCPROFILER_STATUS_ERROR;

            rocprofiler_status_t status = ROCPROFILER_STATUS_SUCCESS;
            for(const auto& [_, agent_session] : sessions)
            {
                // Filter by client if specified
                if(client_id && agent_session->client_idx != client_id->handle) continue;

                // Directly flush the agent's internal HSA buffers rather than going
                // through flush_internal_agent_buffers(), which would re-acquire rlock
                // on this same mutex (undefined behavior with std::shared_mutex).
                status = hsa::flush_internal_agent_buffers(agent_session.get());
                if(status != ROCPROFILER_STATUS_SUCCESS)
                {
                    ROCP_ERROR << "Failed to flush internal HSA buffers tied to rocp buffer "
                               << agent_session->buffer_id.handle;
                }
            }
            return status;
        });
}

void
service_sync(rocprofiler_client_id_t client_id)
{
    // Flush buffers only for agents owned by this specific client.
    // This ensures we don't interfere with PC sampling sessions
    // that other clients may have configured on different agents.
    // If this function is always called after `service_fini`,
    // then there should be no harm in flushing all buffers when
    // detaching each client. This would increase overhead of
    // the finalization a bit, but would reduce the complexity of the code.
    flush_all_agents_buffers(client_id);
}

void
service_fini()
{
    // Flush buffers for all agents regardless of client ownership.
    // During finalization, we need to drain all remaining PC sampling data
    // across all clients to ensure no data is lost.
    flush_all_agents_buffers();
}

}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
