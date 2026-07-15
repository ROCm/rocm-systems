// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "capture_buffer.h"
#include "marker_stack.h"
#include "scope_guard.h"
#include "stack_entry.h"
#include "stats.h"
#include "wire_format.h"

#include <c10/util/ThreadLocalDebugInfo.h>

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

extern "C"
{
#include <rocprofiler-sdk-roctx/roctx.h>
}

namespace roctx_recordfn::detail
{

// Carries the main-thread USER_SCOPE chain to autograd workers.
class RoctxUserScopeChain : public c10::DebugInfoBase
{
public:
    explicit RoctxUserScopeChain(std::vector<StackEntry> c)
        : chain(std::move(c))
    {
    }

    std::vector<StackEntry> chain;
};

// DebugInfoKind slot used to publish the USER_SCOPE chain: a private
// string-keyed slot when available, otherwise TEST_INFO_2.
#ifdef ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND
inline constexpr std::string_view kRoctxUserScopeName = "ROCPROF_TORCHTRACE_INFO";
inline const c10::DebugInfoKind   kRoctxDbgKind(&kRoctxUserScopeName);
#else
inline constexpr c10::DebugInfoKind kRoctxDbgKind = c10::DebugInfoKind::TEST_INFO_2;
#endif

// Overlays the published USER_SCOPE chain onto the thread stack.
inline std::size_t apply_userscope_overlay()
{
    auto* base       = c10::ThreadLocalDebugInfo::get(kRoctxDbgKind);
    auto* chain_info = dynamic_cast<const RoctxUserScopeChain*>(base);
    if (chain_info == nullptr || chain_info->chain.empty())
    {
        return 0;
    }
    const std::vector<StackEntry> chain_copy = chain_info->chain;
    const std::size_t             pushed     = push_with_prefix_dedup(chain_copy);
    if (pushed > 0)
    {
        inc(g_stats.user_scope_inherits);
    }
    return pushed;
}

// Builds the DebugInfoGuard that publishes the chain into TLS, or a null
// guard on failure.
inline std::unique_ptr<c10::DebugInfoGuard> make_userscope_guard(const std::vector<StackEntry>& stack)
{
    try
    {
        auto info = std::make_shared<RoctxUserScopeChain>(stack);
        return std::make_unique<c10::DebugInfoGuard>(kRoctxDbgKind, std::move(info));
    }
    catch (...)
    {
        return nullptr;
    }
}

// Pushes a USER_SCOPE frame and emits a ROCTX range. When non-empty,
// backend is appended to the range as "|<backend>".
inline void push_user_scope(const std::string& marker, const std::string& context, const std::string& backend)
{
    try
    {
        StackEntry entry;
        entry.marker  = marker;
        entry.context = context;
        g_thread.stack.push_back(std::move(entry));
        auto stack_rollback = make_scope_guard([] { g_thread.stack.pop_back(); });

        g_thread.guards.push_back(make_userscope_guard(g_thread.stack));
        auto guards_rollback = make_scope_guard([] { g_thread.guards.pop_back(); });

        std::string wire_string = build_marker_string(g_thread.stack);
        if (!backend.empty())
        {
            wire_string += '|';
            wire_string += backend;
        }
        roctxRangePushA(wire_string.c_str());
        auto roctx_rollback = make_scope_guard([] { roctxRangePop(); });

        g_capture.capture(wire_string);
        inc(g_stats.user_scope_pushes);
        inc(g_stats.pushes);

        roctx_rollback.dismiss();
        guards_rollback.dismiss();
        stack_rollback.dismiss();
    }
    catch (...)
    {
        inc(g_stats.callback_errors);
        throw;
    }
}

inline void pop_user_scope()
{
    try
    {
        if (g_thread.stack.empty() || g_thread.guards.empty())
        {
            inc(g_stats.callback_errors);
            return;
        }
        roctxRangePop();
        inc(g_stats.user_scope_pops);
        inc(g_stats.pops);
        g_thread.stack.pop_back();
        g_thread.guards.pop_back();
    }
    catch (...)
    {
        inc(g_stats.callback_errors);
    }
}

}  // namespace roctx_recordfn::detail
