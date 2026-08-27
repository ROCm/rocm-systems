// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "gsl_assert.h"
#include "leaf_context.h"
#include "marker_stack.h"
#include "process_state.h"
#include "record_function_callback.h"
#include "record_function_installation.h"
#include "snapshot_store.h"
#include "stack_entry.h"
#include "stats.h"
#include "user_scope.h"
#include "wire_format.h"

#include <ATen/record_function.h>
#include <c10/util/ThreadLocalDebugInfo.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C"
{
#include <rocprofiler-sdk-roctx/roctx.h>
}

namespace torch_trace_collector::detail
{
namespace
{

constexpr std::string_view kRoctxUserScopeKindName = "rocprofiler-compute.user_scope";

}  // namespace

const c10::DebugInfoKind kRoctxUserScopeKind{&kRoctxUserScopeKindName};

namespace
{

constexpr const char* kRecordFnBackend = "torch";

void encode_marker_segment(const std::string& name, std::string& out)
{
    for (char c : name)
    {
        if (c == '%')
            out += kEncodedPercent;
        else if (c == '/')
            out += kEncodedSlash;
        else
            out += c;
    }
}

std::unique_ptr<c10::DebugInfoGuard> publish_userscope_chain(const std::vector<StackEntry>& stack)
{
    try
    {
        auto info = std::make_shared<RoctxUserScopeChain>(stack);
        return std::make_unique<c10::DebugInfoGuard>(kRoctxUserScopeKind, std::move(info));
    }
    catch (...)
    {
        return nullptr;
    }
}

void unwind_observer_context(const RoctxObserverContext& observer_ctx, bool count_pop)
{
    std::vector<StackEntry>& stack = thread_state().stack;
    if (observer_ctx.pushed_roctx_range)
    {
        roctxRangePop();
        if (count_pop)
        {
            process_state().stats.pops.fetch_add(1, std::memory_order_relaxed);
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

}  // namespace

ProcessState& process_state()
{
    static ProcessState state;
    return state;
}

ThreadState& thread_state()
{
    static thread_local ThreadState state;
    return state;
}

std::size_t push_with_prefix_dedup(const std::vector<StackEntry>& chain)
{
    std::vector<StackEntry>& stack  = thread_state().stack;
    const std::size_t        maxc   = std::min(chain.size(), stack.size());
    std::size_t              common = 0;
    for (; common < maxc; ++common)
    {
        if (chain[common].marker != stack[common].marker || chain[common].context != stack[common].context)
        {
            break;
        }
    }
    std::size_t pushed = 0;
    for (std::size_t i = common; i < chain.size(); ++i)
    {
        stack.push_back(chain[i]);
        ++pushed;
    }
    return pushed;
}

std::string build_marker_string(const std::vector<StackEntry>& stack)
{
    std::size_t marker_len = 0;
    std::size_t ctx_len    = 0;
    for (const auto& entry : stack)
    {
        marker_len += entry.marker.size() + 1;
        for (char c : entry.marker)
            if (c == '%' || c == '/')
                marker_len += 2;
        ctx_len += entry.context.size() + 1;
    }
    std::string out;
    out.reserve(marker_len + ctx_len + 1);

    for (std::size_t i = 0; i < stack.size(); ++i)
    {
        if (i != 0)
            out += '/';
        encode_marker_segment(stack[i].marker, out);
    }
    out += ':';
    for (std::size_t i = 0; i < stack.size(); ++i)
    {
        if (i != 0)
            out += '/';
        out += stack[i].context;
    }
    return out;
}

std::size_t apply_userscope_overlay()
{
    auto* chain_info = dynamic_cast<const RoctxUserScopeChain*>(
        c10::ThreadLocalDebugInfo::get(kRoctxUserScopeKind));
    if (chain_info == nullptr || chain_info->chain.empty())
    {
        return 0;
    }
    const std::vector<StackEntry> chain_copy = chain_info->chain;
    const std::size_t             pushed     = push_with_prefix_dedup(chain_copy);
    if (pushed > 0)
    {
        process_state().stats.user_scope_inherits.fetch_add(1, std::memory_order_relaxed);
    }
    return pushed;
}

void push_user_scope(const std::string& marker, const std::string& context, const std::string& backend)
{
    ProcessState& state        = process_state();
    ThreadState&  thread       = thread_state();
    bool          pushed_frame = false;
    bool          pushed_guard = false;
    try
    {
        StackEntry entry;
        entry.marker  = marker;
        entry.context = context;
        thread.stack.push_back(std::move(entry));
        pushed_frame = true;

        thread.guards.push_back(publish_userscope_chain(thread.stack));
        pushed_guard = true;

        std::string wire_string = build_marker_string(thread.stack);
        if (!backend.empty())
        {
            wire_string += '|';
            wire_string += backend;
        }
        roctxRangePushA(wire_string.c_str());
        state.stats.user_scope_pushes.fetch_add(1, std::memory_order_relaxed);
        state.stats.pushes.fetch_add(1, std::memory_order_relaxed);
    }
    catch (...)
    {
        if (pushed_guard && !thread.guards.empty())
        {
            thread.guards.pop_back();
        }
        if (pushed_frame && !thread.stack.empty())
        {
            thread.stack.pop_back();
        }
        state.stats.callback_errors.fetch_add(1, std::memory_order_relaxed);
        throw;
    }
}

void pop_user_scope()
{
    try
    {
        ProcessState& state  = process_state();
        ThreadState&  thread = thread_state();

        if (thread.stack.empty() || thread.guards.empty())
        {
            state.stats.callback_errors.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        roctxRangePop();
        state.stats.user_scope_pops.fetch_add(1, std::memory_order_relaxed);
        state.stats.pops.fetch_add(1, std::memory_order_relaxed);
        thread.stack.pop_back();
        thread.guards.pop_back();
    }
    catch (...)
    {
        process_state().stats.callback_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

std::unique_ptr<at::ObserverContext> start_cb(const at::RecordFunction& record_fn)
{
    std::unique_ptr<RoctxObserverContext> observer_ctx;
    try
    {
        observer_ctx = std::make_unique<RoctxObserverContext>();

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
            state.snapshots.save(seq_nr, at::RecordFunction::currentThreadId(), stack);
        }

        std::string wire_string = build_marker_string(stack);
        wire_string += '|';
        wire_string += kRecordFnBackend;
        roctxRangePushA(wire_string.c_str());
        observer_ctx->pushed_roctx_range = true;
        state.stats.pushes.fetch_add(1, std::memory_order_relaxed);

        return observer_ctx;
    }
    catch (...)
    {
        if (observer_ctx)
        {
            try
            {
                unwind_observer_context(*observer_ctx, /*count_pop=*/false);
            }
            catch (...)
            {
            }
        }
        process_state().stats.callback_errors.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
}

void end_cb(const at::RecordFunction& /*record_fn*/, at::ObserverContext* obs_ctx)
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
        process_state().stats.callback_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

std::int64_t install()
{
    return process_state().install.wlock(
        [](InstallState& state)
        {
            if (state.handle != at::INVALID_CALLBACK_HANDLE)
            {
                return static_cast<std::int64_t>(state.handle);
            }
            state.handle = at::addGlobalCallback(
                at::RecordFunctionCallback(start_cb, end_cb)
                    .scopes({at::RecordScope::FUNCTION, at::RecordScope::BACKWARD_FUNCTION}));
            state.installed = true;
            return static_cast<std::int64_t>(state.handle);
        });
}

void uninstall()
{
    process_state().install.wlock(
        [](InstallState& state)
        {
            const auto handle = std::exchange(state.handle, at::INVALID_CALLBACK_HANDLE);
            state.installed   = false;
            if (handle != at::INVALID_CALLBACK_HANDLE)
            {
                at::removeCallback(handle);
            }
            process_state().snapshots.clear();
        });
}

bool is_installed()
{
    return process_state().install.rlock([](const InstallState& state) { return state.installed; });
}

synchronized_t<SnapshotStore::Shard>& SnapshotStore::shard_for(const SnapshotKey& key)
{
    return shards_[shard_index(key)];
}

void SnapshotStore::lru_remove(Shard& shard, const SnapshotKey& key)
{
    auto it = shard.lru_idx.find(key);
    if (it == shard.lru_idx.end())
        return;
    shard.lru_order.erase(it->second);
    shard.lru_idx.erase(it);
}

void SnapshotStore::lru_touch(Shard& shard, const SnapshotKey& key)
{
    lru_remove(shard, key);
    shard.lru_order.push_back(key);
    auto tail = shard.lru_order.end();
    --tail;
    shard.lru_idx.emplace(key, tail);
}

void SnapshotStore::evict_oldest(Shard& shard)
{
    Expects(!shard.lru_order.empty());
    const SnapshotKey oldest = shard.lru_order.front();
    shard.lru_order.pop_front();
    shard.lru_idx.erase(oldest);
    shard.snapshots.erase(oldest);
    stats_.snapshots_dropped.fetch_add(1, std::memory_order_relaxed);
}

void SnapshotStore::save(std::int64_t seq_nr, std::uint64_t thread_id, const std::vector<StackEntry>& stack)
{
    const SnapshotKey key = {seq_nr, thread_id};
    shard_for(key).wlock(
        [&](Shard& shard)
        {
            auto it = shard.snapshots.find(key);
            if (it != shard.snapshots.end())
            {
                it->second = stack;
                lru_touch(shard, key);
                stats_.snapshots_overwritten.fetch_add(1, std::memory_order_relaxed);
                stats_.snapshots_saved.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            while (shard.snapshots.size() >= kShardSoftCap)
            {
                evict_oldest(shard);
            }
            shard.snapshots.emplace(key, stack);
            lru_touch(shard, key);
            stats_.snapshots_saved.fetch_add(1, std::memory_order_relaxed);
        });
}

bool SnapshotStore::consume(std::int64_t seq_nr, std::uint64_t thread_id, std::vector<StackEntry>* out_stack)
{
    const SnapshotKey key = {seq_nr, thread_id};
    return shard_for(key).wlock(
        [&](Shard& shard)
        {
            auto it = shard.snapshots.find(key);
            if (it == shard.snapshots.end())
                return false;
            Expects(shard.lru_idx.find(key) != shard.lru_idx.end());
            *out_stack = std::move(it->second);
            shard.snapshots.erase(it);
            lru_remove(shard, key);
            stats_.snapshots_consumed.fetch_add(1, std::memory_order_relaxed);
            return true;
        });
}

std::size_t SnapshotStore::pending() const
{
    std::size_t total = 0;
    for (const auto& guarded_shard : shards_)
    {
        total += guarded_shard.rlock([](const Shard& shard) { return shard.snapshots.size(); });
    }
    return total;
}

void SnapshotStore::clear()
{
    for (auto& guarded_shard : shards_)
    {
        guarded_shard.wlock(
            [](Shard& shard)
            {
                shard.snapshots.clear();
                shard.lru_order.clear();
                shard.lru_idx.clear();
            });
    }
}

}  // namespace torch_trace_collector::detail
