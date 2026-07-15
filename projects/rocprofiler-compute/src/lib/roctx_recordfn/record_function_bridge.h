// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "capture_buffer.h"
#include "install_state.h"
#include "leaf_context.h"
#include "marker_stack.h"
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
#include <mutex>
#include <string>
#include <utility>
#include <vector>

extern "C"
{
#include <rocprofiler-sdk-roctx/roctx.h>
}

namespace roctx_recordfn::detail
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
// observer_ctx. When count_pop is true, the pop is counted in g_stats.pops.
inline void unwind_observer_context(const RoctxObserverContext& observer_ctx, bool count_pop)
{
    if (observer_ctx.pushed_roctx_range)
    {
        roctxRangePop();
        if (count_pop)
        {
            inc(g_stats.pops);
        }
    }
    if (observer_ctx.pushed_leaf && !g_thread.stack.empty())
    {
        g_thread.stack.pop_back();
    }
    for (std::size_t i = 0; i < observer_ctx.pushed_snapshot_frames && !g_thread.stack.empty(); ++i)
    {
        g_thread.stack.pop_back();
    }
}

inline std::unique_ptr<at::ObserverContext> start_cb(const at::RecordFunction& record_fn)
{
    try
    {
        auto observer_ctx = std::make_unique<RoctxObserverContext>();
        auto rollback     = make_scope_guard(
            [&] { unwind_observer_context(*observer_ctx, /*count_pop=*/false); });

        const at::RecordScope scope  = record_fn.scope();
        const std::int64_t    seq_nr = record_fn.seqNr();
        const char*           name   = record_fn.name();
        if (name == nullptr || name[0] == '\0')
        {
            name = "<anonymous>";
        }

        const bool stack_was_empty          = g_thread.stack.empty();
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
            std::vector<StackEntry> snapshot;
            if (g_snapshots.consume(seq_nr, &snapshot))
            {
                observer_ctx->pushed_snapshot_frames += push_with_prefix_dedup(snapshot);
            }
        }

        StackEntry leaf;
        leaf.marker                  = name;
        const bool is_backward_scope = (scope == at::RecordScope::BACKWARD_FUNCTION);
        leaf.context = roctx_recordfn::default_leaf_context(is_backward_scope,
                                                            seq_nr,
                                                            stack_was_empty_for_leaf);
        g_thread.stack.push_back(std::move(leaf));
        observer_ctx->pushed_leaf = true;

        if (scope == at::RecordScope::FUNCTION && seq_nr >= 0)
        {
            g_snapshots.save(seq_nr, g_thread.stack);
        }

        // Emit the ROCTX range. RecordFunction ops are torch-backed.
        std::string wire_string = build_marker_string(g_thread.stack);
        wire_string += '|';
        wire_string += kRecordFnBackend;
        roctxRangePushA(wire_string.c_str());
        observer_ctx->pushed_roctx_range = true;
        g_capture.capture(wire_string);
        inc(g_stats.pushes);

        rollback.dismiss();
        return observer_ctx;
    }
    catch (...)
    {
        inc(g_stats.callback_errors);
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
        inc(g_stats.callback_errors);
    }
}

inline std::int64_t install()
{
    std::lock_guard<std::mutex> lock(g_install.mutex);
    const auto                  existing = g_install.handle.load();
    if (existing != at::INVALID_CALLBACK_HANDLE)
    {
        return static_cast<std::int64_t>(existing);
    }
    const auto handle = at::addGlobalCallback(
        at::RecordFunctionCallback(start_cb, end_cb)
            .scopes({at::RecordScope::FUNCTION, at::RecordScope::BACKWARD_FUNCTION}));
    g_install.handle.store(handle);
    g_install.installed.store(true);
    return static_cast<std::int64_t>(handle);
}

inline void uninstall()
{
    std::lock_guard<std::mutex> lock(g_install.mutex);
    const auto                  handle = g_install.handle.exchange(at::INVALID_CALLBACK_HANDLE);
    g_install.installed.store(false);
    if (handle != at::INVALID_CALLBACK_HANDLE)
    {
        at::removeCallback(handle);
    }
}

inline bool is_installed()
{
    return g_install.installed.load();
}

}  // namespace roctx_recordfn::detail
