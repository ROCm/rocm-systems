// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/kernel_replay/replay_callbacks.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/experimental/kernel_replay.h>

namespace rocprofiler
{
namespace kernel_replay
{
namespace
{
template <typename Integral>
constexpr Integral
bit_extract(Integral x, int first, int last)
{
    return (x >> first) & ((Integral{1} << (last - first + 1)) - 1);
}

bool
context_has_kernel_replay(const tracing::context_t* ctx)
{
    return (CHECK_NOTNULL(ctx) && ctx->callback_tracer &&
            ctx->callback_tracer->domains(ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                          ROCPROFILER_KERNEL_REPLAY_CONFIG));
}
}  // namespace

bool
has_active_replay_contexts()
{
    return !context::get_active_contexts(context_has_kernel_replay).empty();
}

rocprofiler_kernel_dispatch_info_t
make_dispatch_info(const hsa::Queue& queue, const hsa::rocprofiler_packet& pkt)
{
    constexpr auto kernel_dispatch_info_rt_size =
        common::compute_runtime_sizeof<rocprofiler_kernel_dispatch_info_t>();

    auto info     = common::init_public_api_struct(rocprofiler_kernel_dispatch_info_t{});
    info.size     = kernel_dispatch_info_rt_size;
    info.queue_id = queue.get_id();
    // info.dispatch_id = set before callback is invoked;

    info.agent_id = CHECK_NOTNULL(queue.get_agent().get_rocp_agent())->id;

    const auto& original_packet = pkt.kernel_dispatch;
    const auto  packet_type     = bit_extract(original_packet.header,
                                         HSA_PACKET_HEADER_TYPE,
                                         HSA_PACKET_HEADER_TYPE + HSA_PACKET_HEADER_WIDTH_TYPE - 1);

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x0D
    if(packet_type == HSA_PACKET_TYPE_VENDOR_SPECIFIC)
    {
        const auto& ext_packet = pkt.ext_kernel_dispatch;
        if(ext_packet.amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH)
        {
            info.kernel_id            = code_object::get_kernel_id(ext_packet.kernel_object);
            info.private_segment_size = ext_packet.private_segment_size;
            info.group_segment_size   = ext_packet.group_segment_size;
            info.workgroup_size       = {ext_packet.workgroup_size_x,
                                   ext_packet.workgroup_size_y,
                                   ext_packet.workgroup_size_z};
            info.grid_size            = {static_cast<uint32_t>(ext_packet.cluster_count_x) *
                                  static_cast<uint32_t>(ext_packet.cluster_size_x) *
                                  static_cast<uint32_t>(ext_packet.workgroup_size_x),
                              static_cast<uint32_t>(ext_packet.cluster_count_y) *
                                  static_cast<uint32_t>(ext_packet.cluster_size_y) *
                                  static_cast<uint32_t>(ext_packet.workgroup_size_y),
                              static_cast<uint32_t>(ext_packet.cluster_count_z) *
                                  static_cast<uint32_t>(ext_packet.cluster_size_z) *
                                  static_cast<uint32_t>(ext_packet.workgroup_size_z)};
            return info;
        }
    }
#else
    (void) packet_type;
#endif

    const auto& s             = pkt.kernel_dispatch;
    info.kernel_id            = code_object::get_kernel_id(s.kernel_object);
    info.private_segment_size = s.private_segment_size;
    info.group_segment_size   = s.group_segment_size;
    info.workgroup_size       = {s.workgroup_size_x, s.workgroup_size_y, s.workgroup_size_z};
    info.grid_size            = {s.grid_size_x, s.grid_size_y, s.grid_size_z};
    return info;
}

replay_plan_t
execute_config_phase_enter(const hsa::Queue&              queue,
                           const hsa::rocprofiler_packet& pkt,
                           rocprofiler_thread_id_t        thr_id,
                           uint64_t                       internal_corr_id,
                           uint64_t                       ancestor_corr_id)
{
    auto plan = replay_plan_t{};

    auto config_contexts = tracing::callback_context_data_vec_t{};
    auto extern_corr_ids = tracing::external_correlation_id_map_t{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                               ROCPROFILER_KERNEL_REPLAY_CONFIG,
                               config_contexts,
                               extern_corr_ids);

    if(config_contexts.empty())
    {
        return plan;
    }

    plan.config_data =
        common::init_public_api_struct(rocprofiler_callback_tracing_kernel_replay_data_t{});
    plan.config_data.dispatch_info = make_dispatch_info(queue, pkt);

    tracing::execute_phase_enter_callbacks(config_contexts,
                                           thr_id,
                                           internal_corr_id,
                                           extern_corr_ids,
                                           ancestor_corr_id,
                                           ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                           ROCPROFILER_KERNEL_REPLAY_CONFIG,
                                           plan.config_data);

    plan.pass_count_cb            = plan.config_data.pass_count_cb;
    plan.replay_continue_cb       = plan.config_data.replay_continue_cb;
    plan.config_contexts          = std::move(config_contexts);
    plan.external_correlation_ids = std::move(extern_corr_ids);
    plan.user_data                = plan.config_contexts.empty() ? tracing::empty_user_data
                                                                 : plan.config_contexts.front().user_data;

    if(!plan.pass_count_cb)
    {
        execute_config_phase_exit(plan, thr_id, internal_corr_id, ancestor_corr_id);
        return plan;
    }

    plan.total_passes = plan.pass_count_cb(plan.config_data.dispatch_info, plan.user_data);
    if(plan.total_passes == 0 && !plan.replay_continue_cb)
    {
        ROCP_WARNING << "kernel replay: pass_count_cb returned 0 without replay_continue_cb; "
                        "dispatch will not be replayed";
        execute_config_phase_exit(plan, thr_id, internal_corr_id, ancestor_corr_id);
        return plan;
    }

    plan.indefinite       = (plan.total_passes == 0);
    plan.replay_requested = plan.indefinite || plan.total_passes > 1;
    if(!plan.replay_requested)
        execute_config_phase_exit(plan, thr_id, internal_corr_id, ancestor_corr_id);

    return plan;
}

void
execute_config_phase_exit(const replay_plan_t& plan,
                          rocprofiler_thread_id_t /*thr_id*/,
                          uint64_t /*internal_corr_id*/,
                          uint64_t /*ancestor_corr_id*/)
{
    if(plan.config_contexts.empty()) return;

    auto config_contexts = plan.config_contexts;
    auto extern_corr_ids = plan.external_correlation_ids;
    auto config_data     = plan.config_data;

    tracing::execute_phase_exit_callbacks(config_contexts,
                                          extern_corr_ids,
                                          ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                          ROCPROFILER_KERNEL_REPLAY_CONFIG,
                                          config_data);
}

void
execute_pass_phase_enter(const replay_plan_t&    plan,
                         uint64_t                current_pass,
                         rocprofiler_thread_id_t thr_id,
                         uint64_t                internal_corr_id,
                         uint64_t                ancestor_corr_id,
                         pass_context_state_t&   out_pass_state)
{
    out_pass_state = pass_context_state_t{};
    tracing::populate_contexts(ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                               ROCPROFILER_KERNEL_REPLAY_PASS,
                               out_pass_state.contexts,
                               out_pass_state.external_correlation_ids);

    // Deliver the sequence-wide user_data (captured at CONFIG PHASE_ENTER) to every PASS callback.
    for(auto& itr : out_pass_state.contexts)
        itr.user_data = plan.user_data;

    auto pass_data =
        common::init_public_api_struct(rocprofiler_callback_tracing_kernel_replay_data_t{});
    pass_data.dispatch_info = plan.config_data.dispatch_info;
    pass_data.current_pass  = current_pass;
    pass_data.total_passes  = plan.indefinite ? 0 : plan.total_passes;

    tracing::execute_phase_enter_callbacks(out_pass_state.contexts,
                                           thr_id,
                                           internal_corr_id,
                                           out_pass_state.external_correlation_ids,
                                           ancestor_corr_id,
                                           ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                           ROCPROFILER_KERNEL_REPLAY_PASS,
                                           pass_data);
}

void
execute_pass_phase_exit(const replay_plan_t&  plan,
                        uint64_t              current_pass,
                        pass_context_state_t& pass_state)
{
    // Reuse the contexts populated during PASS PHASE_ENTER so the exit record carries the same
    // thread id, correlation ids, operation, and user_data captured at enter.
    auto pass_data =
        common::init_public_api_struct(rocprofiler_callback_tracing_kernel_replay_data_t{});
    pass_data.dispatch_info = plan.config_data.dispatch_info;
    pass_data.current_pass  = current_pass;
    pass_data.total_passes  = plan.indefinite ? 0 : plan.total_passes;

    tracing::execute_phase_exit_callbacks(pass_state.contexts,
                                          pass_state.external_correlation_ids,
                                          ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
                                          ROCPROFILER_KERNEL_REPLAY_PASS,
                                          pass_data);
}

bool
should_continue_replay(const replay_plan_t& plan, uint64_t current_pass, bool is_final_pass)
{
    // Fixed-count loops never exceed total_passes; replay_continue_cb may only break early, not
    // extend the loop past N passes.
    //
    // TODO: optionally treat N (from pass_count_cb) as a *minimum* rather than a hard cap, so
    // replay_continue_cb can extend the loop beyond N ("run at least 4 passes, but keep going if I
    // still need more"). The app's completion signal is already fired once after the loop (see
    // WriteInterceptor), not at pass N-1, so lifting the cap below would be sufficient.
    if(!plan.indefinite && is_final_pass) return false;

    if(plan.replay_continue_cb)
        return plan.replay_continue_cb(plan.config_data.dispatch_info,
                                       current_pass,
                                       plan.indefinite ? 0 : plan.total_passes,
                                       plan.user_data) != 0;

    // No continue cb: indefinite loops require one (rejected at config), so this is a fixed loop
    // that has not yet reached its final pass.
    return true;
}
}  // namespace kernel_replay
}  // namespace rocprofiler
