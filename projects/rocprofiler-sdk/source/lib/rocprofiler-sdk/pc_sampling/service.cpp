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
#    include "lib/rocprofiler-sdk/pc_sampling/hsa_adapter.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/ioctl/ioctl_adapter.hpp"
#    include "lib/rocprofiler-sdk/pc_sampling/utils.hpp"

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

// The function returns the atomic pointer to the active PC sampling service.
// The nullptr means the PC sampling service is inactive.
atomic_pc_sampling_service_t&
get_active_pc_sampling_service()
{
    static auto _v = atomic_pc_sampling_service_t{nullptr};
    return _v;
}

// The function returns the atomic pointer to the configured pc sampling service.
// The nullptr means the PC sampling service is not configured.
atomic_pc_sampling_service_t&
get_configured_pc_sampling_service()
{
    static auto _v = atomic_pc_sampling_service_t{nullptr};
    return _v;
}

rocprofiler_status_t
start_service(const context::context* ctx)
{
    auto* service = ctx->pc_sampler.get();

    context::pc_sampling_service* _expected = nullptr;
    // If there is no active pc_sampling_service, mark `service` as activated.
    bool success = get_active_pc_sampling_service().compare_exchange_strong(_expected, service);

    if(!success)
    {
        // Some other context is active at the moment.
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

    if(get_active_pc_sampling_service().load() != service)
    {
        // Some other service is activated at the moment.
        return ROCPROFILER_STATUS_ERROR;
    }

    if(is_hsa_initialized().load())
    {
        hsa::pc_sampling_service_stop(service);
    }

    // No active PC sampling services
    bool success = get_active_pc_sampling_service().compare_exchange_strong(service, nullptr);

    return (success) ? ROCPROFILER_STATUS_SUCCESS : ROCPROFILER_STATUS_ERROR;
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

    // If the PC sampling service is not configured on any of the agents, return.
    if(!get_configured_pc_sampling_service().load()) return;

    static auto _once = std::once_flag{};
    std::call_once(_once, []() {
        // Configure PC sampling on the ROCr level only once.
        hsa::pc_sampling_service_finish_configuration(get_configured_pc_sampling_service().load());
    });

    // Theoretically, the remainder of the function
    // can execute concurrently with start_context/stop_context.

    context::pc_sampling_service* _expected   = nullptr;
    void*                         invalid_ptr = reinterpret_cast<void*>(0xDEADBEEF);
    context::pc_sampling_service* pseudo_sevice =
        static_cast<context::pc_sampling_service*>(invalid_ptr);

    if(get_active_pc_sampling_service().compare_exchange_strong(_expected, pseudo_sevice))
    {
        // At this point, we prevented any `start_context` instance from activating the service.
        is_hsa_initialized().store(true);
        // Now, allow `start_context` to active the service.
        get_active_pc_sampling_service().compare_exchange_strong(pseudo_sevice, nullptr);
    }
    else
    {
        // Someone already called `start_context` that activated service.
        // The pointer to this service is written inside `_expected`.
        // Start PC sampling service on the HSA level in the name of the
        // `start_context` caller.
        hsa::pc_sampling_service_start(_expected);
        // Although the caller of the `start_context` might try calling the hsa_start,
        // it will fail, which is fine, since the service is eventually started.
        is_hsa_initialized().store(true);
    }
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
    if(ctx->counter_collection || ctx->device_counter_collection)
    {
        return ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT;
    }

    if(!ctx->pc_sampler)
    {
        ctx->pc_sampler = std::make_unique<context::pc_sampling_service>();
    }

    if(ctx->pc_sampler->agent_sessions.count(agent->id) > 0)
    {
        // The service has already been configured for this agent.
        return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED;
    }

    // The restriction we agreed at the moment is that at most one context
    // can have PC sampling service configured, meaning
    // at most one instance of the `context::pc_sampling_service` can be configured
    // This `pc_sampling_service` contains at most one configuration per agent.
    context::pc_sampling_service* expected = nullptr;
    // Try registering the new instance of the `pc_sampling_service`.
    if(!get_configured_pc_sampling_service().compare_exchange_strong(expected,
                                                                     ctx->pc_sampler.get()))
    {
        // A `pc_sampling_service` instance has already been configured.
        // Note: the `expected` contains the pointer to the configured `pc_sampling_service`
        // instance.
        if(expected != ctx->pc_sampler.get())
        {
            // Someone tried configuring a new `pc_sampling_service instance`, which we do not
            // allow. Invalidate the `pc_sampling_service` from the `ctx` and return an error.
            ctx->pc_sampler = nullptr;
            // TODO: new status code needed
            return ROCPROFILER_STATUS_ERROR;
        }
        // Someone is trying to enable PC sampling on another agent, and we allow registering
        // new agent inside `pc_sampling_service` instance.
    }

    // calling KFD to check if the configuration is actually supported at the moment
    uint32_t ioctl_pcs_id;
    auto     ioctl_status = ioctl::ioctl_pcs_create(agent, method, unit, interval, &ioctl_pcs_id);
    if(ioctl_status != ROCPROFILER_STATUS_SUCCESS) return ioctl_status;

    ctx->pc_sampler->agent_sessions[agent->id] = std::make_unique<PCSAgentSession>();

    auto* session         = ctx->pc_sampler->agent_sessions[agent->id].get();
    session->agent        = agent;
    session->method       = method;
    session->unit         = unit;
    session->interval     = interval;
    session->buffer_id    = buffer_id;
    session->ioctl_pcs_id = ioctl_pcs_id;
    session->parser       = std::make_unique<PCSamplingParserContext>();
    session->cid_manager  = std::make_unique<PCSCIDManager>(session->parser.get());

    ROCP_ERROR << "PC sampling session with id: " << session->ioctl_pcs_id
               << " hsa been created!\n";

    return ROCPROFILER_STATUS_SUCCESS;
}

bool
is_pc_sample_service_configured(rocprofiler_agent_id_t agent_id)
{
    auto* service = get_configured_pc_sampling_service().load();
    if(service)
    {
        // If the agent_id is in the service->agent_sessions map,
        // then the PC sampling service is configured on this agent.
        return service->agent_sessions.find(agent_id) != service->agent_sessions.end();
    }
    // The PC sampling service is not configured on this agent
    return false;
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

    auto* service = get_configured_pc_sampling_service().load();
    if(service && ctx->pc_sampler.get() == service)
    {
        rocprofiler_status_t status = ROCPROFILER_STATUS_SUCCESS;
        // The context `ctx` (that holds the buffer with `buffer_id`)
        // is the one containing PC sampling service.
        // The HSA interception table is registered.
        for(const auto& [_, agent_session] : service->agent_sessions)
        {
            // Find the agent that fills the buffer with `buffer_id`
            if(agent_session->buffer_id.handle == buffer_id.handle)
            {
                // Flush internal PC sampling buffers filled by the agent
                // NOTE: one rocprofiler-SDK PC sampling buffer can be tied
                // to multiple agent (agent sessions).
                status = hsa::flush_internal_agent_buffers(agent_session.get());
                if(status != ROCPROFILER_STATUS_SUCCESS) return status;
            }
        }
    }

    // PC sampling service not configured.
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
flush_all_agent_buffers()
{
    auto* service = get_configured_pc_sampling_service().load();
    if(!service) return ROCPROFILER_STATUS_ERROR;

    rocprofiler_status_t status = ROCPROFILER_STATUS_SUCCESS;
    // Loop over all agents that have PC sampling service configured
    // and drain their internal buffers.
    // NOTE: one SDK buffer can consume data from multiple agents
    // (multiple HSA runtime buffers)
    for(const auto& [_, agent_session] : service->agent_sessions)
    {
        status = flush_internal_agent_buffers(agent_session->buffer_id);
        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            ROCP_ERROR << "Failed to flush internal HSA buffers tied to rocp buffer "
                       << agent_session->buffer_id.handle;
        }
    }
    return status;
}

void
service_sync()
{
    flush_all_agent_buffers();
}

void
service_fini()
{
    flush_all_agent_buffers();
}

}  // namespace pc_sampling
}  // namespace rocprofiler

#endif
