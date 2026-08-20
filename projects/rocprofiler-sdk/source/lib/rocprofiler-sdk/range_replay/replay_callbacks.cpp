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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/range_replay/replay_callbacks.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/local_context.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <atomic>

namespace rocprofiler
{
namespace range_replay
{
namespace
{
std::atomic<bool>&
range_replay_service_configured_flag()
{
    static auto*& _v = common::static_object<std::atomic<bool>>::construct(false);
    return *_v;
}

bool
context_has_range_replay(const tracing::context_t* ctx)
{
    return (CHECK_NOTNULL(ctx) && ctx->callback_tracer &&
            ctx->callback_tracer->domains(ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                          ROCPROFILER_RANGE_REPLAY_CONFIG));
}

range_replay_data_t
make_data(uint64_t range_id, rocprofiler_agent_id_t agent_id, uint64_t dispatch_count)
{
    auto data           = common::init_public_api_struct(range_replay_data_t{});
    data.range_id       = range_id;
    data.agent_id       = agent_id;
    data.dispatch_count = dispatch_count;
    return data;
}
}  // namespace

void
set_range_replay_service_configured(bool enabled)
{
    // Skip during finalization: the flag is a static_object that may already be destroyed.
    if(registration::get_fini_status() > 0) return;
    range_replay_service_configured_flag().store(enabled, std::memory_order_relaxed);
}

bool
has_active_range_replay_contexts()
{
    // Skip during finalization: the flag and the context registry are static_objects that may be
    // destroyed by then, and WriteInterceptor can still call this from HIP/HSA teardown.
    if(registration::get_fini_status() > 0) return false;
    if(!range_replay_service_configured_flag().load(std::memory_order_relaxed)) return false;
    return !context::get_active_contexts(context_has_range_replay).empty();
}

bool
has_registered_range_replay_context()
{
    if(registration::get_fini_status() > 0) return false;
    return !context::get_registered_contexts(context_has_range_replay).empty();
}

range_plan_t
execute_config_callback(uint64_t                range_id,
                        rocprofiler_thread_id_t thr_id,
                        uint64_t                internal_corr_id,
                        uint64_t                ancestor_corr_id)
{
    auto plan = range_plan_t{};

    auto config_contexts = tracing::callback_context_data_vec_t{};
    auto extern_corr_ids = tracing::external_correlation_id_map_t{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                               ROCPROFILER_RANGE_REPLAY_CONFIG,
                               config_contexts,
                               extern_corr_ids);

    if(config_contexts.empty()) return plan;

    plan.config_data = make_data(range_id, rocprofiler_agent_id_t{.handle = 0}, 0);

    tracing::execute_phase_enter_callbacks(config_contexts,
                                           thr_id,
                                           internal_corr_id,
                                           extern_corr_ids,
                                           ancestor_corr_id,
                                           ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                           ROCPROFILER_RANGE_REPLAY_CONFIG,
                                           plan.config_data);

    plan.pass_count_cb            = plan.config_data.pass_count_cb;
    plan.replay_continue_cb       = plan.config_data.replay_continue_cb;
    plan.config_contexts          = std::move(config_contexts);
    plan.external_correlation_ids = std::move(extern_corr_ids);
    plan.user_data                = plan.config_contexts.front().user_data;

    // Resolve CLOSE's contexts now, while the range is being admitted. See range_plan_t.
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                               ROCPROFILER_RANGE_REPLAY_CLOSE,
                               plan.close_contexts,
                               plan.close_correlation_ids);

    if(plan.pass_count_cb)
    {
        plan.total_passes = plan.pass_count_cb(range_id, plan.user_data);
        if(plan.total_passes == 0 && !plan.replay_continue_cb)
        {
            ROCP_WARNING << "range replay: pass_count_cb returned 0 without replay_continue_cb; "
                            "the range will not be replayed";
        }
        else
        {
            plan.indefinite       = (plan.total_passes == 0);
            plan.replay_requested = plan.indefinite || plan.total_passes > 1;
        }
    }

    // CONFIG is a synchronous exchange: close it here rather than holding the phase open across the
    // application code the range brackets.
    auto exit_contexts = plan.config_contexts;
    auto exit_corr_ids = plan.external_correlation_ids;
    auto exit_data     = plan.config_data;
    tracing::execute_phase_exit_callbacks(exit_contexts,
                                          exit_corr_ids,
                                          ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                          ROCPROFILER_RANGE_REPLAY_CONFIG,
                                          exit_data);

    return plan;
}

void
execute_pass_phase_enter(const range_plan_t&     plan,
                         uint64_t                current_pass,
                         rocprofiler_agent_id_t  agent_id,
                         uint64_t                dispatch_count,
                         rocprofiler_thread_id_t thr_id,
                         uint64_t                internal_corr_id,
                         uint64_t                ancestor_corr_id,
                         pass_context_state_t&   out_pass_state)
{
    out_pass_state = pass_context_state_t{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                               ROCPROFILER_RANGE_REPLAY_PASS,
                               out_pass_state.contexts,
                               out_pass_state.external_correlation_ids);

    for(auto& itr : out_pass_state.contexts)
        itr.user_data = plan.user_data;

    auto pass_data         = make_data(plan.config_data.range_id, agent_id, dispatch_count);
    pass_data.current_pass = current_pass;
    pass_data.total_passes = plan.indefinite ? 0 : plan.total_passes;

    // Localized context control, shared with kernel replay: the tool may enable/disable a context
    // for this range's replay loop from its PASS PHASE_ENTER callback. Legal only while armed, so
    // the arm window brackets the tool callback and is closed by a scope guard (a throwing tool
    // callback must not leave the toggles armed).
    pass_data.replay_local_start_context_cb = &kernel_replay::replay_local_start_context;
    pass_data.replay_local_stop_context_cb  = &kernel_replay::replay_local_stop_context;

    {
        kernel_replay::set_toggles_armed(true);
        const auto _disarm =
            common::scope_destructor{[]() { kernel_replay::set_toggles_armed(false); }};
        tracing::execute_phase_enter_callbacks(out_pass_state.contexts,
                                               thr_id,
                                               internal_corr_id,
                                               out_pass_state.external_correlation_ids,
                                               ancestor_corr_id,
                                               ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                               ROCPROFILER_RANGE_REPLAY_PASS,
                                               pass_data);
    }
}

void
execute_pass_phase_exit(const range_plan_t&    plan,
                        uint64_t               current_pass,
                        rocprofiler_agent_id_t agent_id,
                        uint64_t               dispatch_count,
                        pass_context_state_t&  pass_state)
{
    auto pass_data         = make_data(plan.config_data.range_id, agent_id, dispatch_count);
    pass_data.current_pass = current_pass;
    pass_data.total_passes = plan.indefinite ? 0 : plan.total_passes;

    tracing::execute_phase_exit_callbacks(pass_state.contexts,
                                          pass_state.external_correlation_ids,
                                          ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                          ROCPROFILER_RANGE_REPLAY_PASS,
                                          pass_data);
}

void
execute_close_callback(const range_plan_t&               plan,
                       rocprofiler_range_replay_status_t status,
                       rocprofiler_agent_id_t            agent_id,
                       uint64_t                          dispatch_count,
                       uint64_t                          divergence_count,
                       rocprofiler_thread_id_t           thr_id,
                       uint64_t                          internal_corr_id,
                       uint64_t                          ancestor_corr_id)
{
    // The contexts resolved when the range was admitted, not the ones active now.
    auto contexts        = plan.close_contexts;
    auto extern_corr_ids = plan.close_correlation_ids;

    if(contexts.empty()) return;

    for(auto& itr : contexts)
        itr.user_data = plan.user_data;

    auto close_data             = make_data(plan.config_data.range_id, agent_id, dispatch_count);
    close_data.status           = status;
    close_data.divergence_count = divergence_count;
    close_data.total_passes     = plan.indefinite ? 0 : plan.total_passes;

    tracing::execute_phase_enter_callbacks(contexts,
                                           thr_id,
                                           internal_corr_id,
                                           extern_corr_ids,
                                           ancestor_corr_id,
                                           ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                           ROCPROFILER_RANGE_REPLAY_CLOSE,
                                           close_data);

    tracing::execute_phase_exit_callbacks(contexts,
                                          extern_corr_ids,
                                          ROCPROFILER_CALLBACK_TRACING_RANGE_REPLAY,
                                          ROCPROFILER_RANGE_REPLAY_CLOSE,
                                          close_data);
}

bool
should_continue_replay(const range_plan_t& plan, uint64_t current_pass, bool is_final_pass)
{
    // Fixed-count loops never exceed total_passes; replay_continue_cb may only break early.
    if(!plan.indefinite && is_final_pass) return false;

    if(plan.replay_continue_cb)
        return plan.replay_continue_cb(plan.config_data.range_id,
                                       current_pass,
                                       plan.indefinite ? 0 : plan.total_passes,
                                       plan.user_data) != 0;

    return true;
}
}  // namespace range_replay
}  // namespace rocprofiler
