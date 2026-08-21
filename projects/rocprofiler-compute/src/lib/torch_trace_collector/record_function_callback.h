// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// The RecordFunction callback pair. start_cb builds the marker path for one ATen
// operator and emits a ROCTX range; end_cb unwinds what start_cb pushed. Neither
// throws, so an error inside either is counted and swallowed.

#pragma once

#include "leaf_context.h"
#include "marker_stack.h"
#include "process_state.h"
#include "scope_guard.h"
#include "snapshot_store.h"
#include "stack_entry.h"
#include "stats.h"
#include "user_scope.h"
#include "wire_format.h"

#include <ATen/record_function.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

extern "C"
{
#include <rocprofiler-sdk-roctx/roctx.h>
}

namespace torch_trace_collector::detail
{

// Records what start_cb pushed so end_cb can unwind it.
struct RoctxObserverContext : public at::ObserverContext
{
    bool        pushed_roctx_range     = false;
    bool        pushed_leaf            = false;
    std::size_t pushed_snapshot_frames = 0;
};

// The RecordFunction tier instruments PyTorch ATen operators.
inline constexpr const char* kRecordFnBackend = "torch";

// Pops the ROCTX range, leaf frame, and snapshot frames recorded in
// observer_ctx. When count_pop is true, the pop is counted in Stats::pops.
inline void unwind_observer_context(const RoctxObserverContext& observer_ctx, bool count_pop)
{
    std::vector<StackEntry>& stack = thread_state().stack;
    if (observer_ctx.pushed_roctx_range)
    {
        roctxRangePop();
        if (count_pop)
        {
            inc(process_state().stats.pops);
        }
    }
    if (observer_ctx.pushed_leaf && !stack.empty())
    {
        stack.pop_back();
    }
    for (std::size_t i = 0; i < observer_ctx.pushed_snapshot_frames && !stack.empty(); ++i)
    {
        stack.pop_back();
    }
}

inline std::unique_ptr<at::ObserverContext> start_cb(const at::RecordFunction& record_fn)
{
    try
    {
        auto observer_ctx = std::make_unique<RoctxObserverContext>();
        auto rollback     = make_scope_guard(
            [&] { unwind_observer_context(*observer_ctx, /*count_pop=*/false); });

        ProcessState&            state = process_state();
        std::vector<StackEntry>& stack = thread_state().stack;

        const at::RecordScope scope  = record_fn.scope();
        const std::int64_t    seq_nr = record_fn.seqNr();
        const char*           name   = record_fn.name();
        if (name == nullptr || name[0] == '\0')
        {
            name = "<anonymous>";
        }

        const bool stack_was_empty          = stack.empty();
        bool       stack_was_empty_for_leaf = stack_was_empty;

        // On the first record seen on this thread, apply the TLS overlay to
        // re-seed autograd workers from the main-thread chain.
        if (stack_was_empty)
        {
            const std::size_t overlay_frames = apply_userscope_overlay();
            observer_ctx->pushed_snapshot_frames += overlay_frames;
            if (overlay_frames > 0)
            {
                stack_was_empty_for_leaf = false;
            }
        }

        if (scope == at::RecordScope::BACKWARD_FUNCTION && seq_nr >= 0)
        {
            // Autograd stamps the record with the id of the thread that built
            // the node. Zero means the record carries no forward identity, so
            // there is nothing to correlate.
            const std::uint64_t     forward_thread_id = record_fn.forwardThreadId();
            std::vector<StackEntry> snapshot;
            if (forward_thread_id != 0 && state.snapshots.consume(seq_nr, forward_thread_id, &snapshot))
            {
                observer_ctx->pushed_snapshot_frames += push_with_prefix_dedup(snapshot);
            }
        }

        StackEntry leaf;
        leaf.marker                  = name;
        const bool is_backward_scope = (scope == at::RecordScope::BACKWARD_FUNCTION);
        leaf.context = torch_trace_collector::default_leaf_context(is_backward_scope,
                                                                   seq_nr,
                                                                   stack_was_empty_for_leaf);
        stack.push_back(std::move(leaf));
        observer_ctx->pushed_leaf = true;

        if (scope == at::RecordScope::FUNCTION && seq_nr >= 0)
        {
            // Autograd records this same thread id when it builds the node.
            state.snapshots.save(seq_nr, at::RecordFunction::currentThreadId(), stack);
        }

        // Emit the ROCTX range. RecordFunction ops are torch-backed.
        std::string wire_string = build_marker_string(stack);
        wire_string += '|';
        wire_string += kRecordFnBackend;
        roctxRangePushA(wire_string.c_str());
        observer_ctx->pushed_roctx_range = true;
        inc(state.stats.pushes);

        rollback.dismiss();
        return observer_ctx;
    }
    catch (...)
    {
        inc(process_state().stats.callback_errors);
        return nullptr;
    }
}

inline void end_cb(const at::RecordFunction& /*record_fn*/, at::ObserverContext* obs_ctx)
{
    if (obs_ctx == nullptr)
    {
        return;
    }
    auto* observer_ctx = static_cast<RoctxObserverContext*>(obs_ctx);
    try
    {
        unwind_observer_context(*observer_ctx, /*count_pop=*/true);
    }
    catch (...)
    {
        inc(process_state().stats.callback_errors);
    }
}

}  // namespace torch_trace_collector::detail
