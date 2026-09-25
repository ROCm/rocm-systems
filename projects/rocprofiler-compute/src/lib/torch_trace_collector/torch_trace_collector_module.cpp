// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "process_state.h"
#include "record_function_installation.h"
#include "torch_trace_collector.h"
#include "user_scope.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <type_traits>

using torch_trace_collector::detail::process_state;

namespace
{

std::mutex collector_lifecycle_mutex;

}  // namespace

static_assert(std::is_standard_layout_v<torch_trace_collector_stats>);
static_assert(sizeof(torch_trace_collector_stats) == TORCH_TRACE_COLLECTOR_STATS_ABI_SIZE);

extern "C" std::uint32_t torch_trace_collector_abi_revision(void)
{
    return TORCH_TRACE_COLLECTOR_ABI_REVISION;
}

extern "C" std::int32_t torch_trace_collector_install(void)
{
    try
    {
        const std::lock_guard<std::mutex> lifecycle_lock{collector_lifecycle_mutex};
        if (torch_trace_collector::detail::is_installed())
        {
            return 0;
        }

        if (torch_trace_collector::detail::install() == at::INVALID_CALLBACK_HANDLE)
        {
            return 1;
        }
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" std::int32_t torch_trace_collector_uninstall(void)
{
    try
    {
        const std::lock_guard<std::mutex> lifecycle_lock{collector_lifecycle_mutex};
        torch_trace_collector::detail::uninstall();
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" std::int32_t torch_trace_collector_is_installed(void)
{
    try
    {
        return torch_trace_collector::detail::is_installed() ? 1 : 0;
    }
    catch (...)
    {
        return 0;
    }
}

extern "C" std::int32_t torch_trace_collector_push_user_scope(const char* marker,
                                                              const char* context,
                                                              const char* backend)
{
    if (marker == nullptr || context == nullptr || backend == nullptr)
    {
        return 1;
    }

    try
    {
        if (!torch_trace_collector::detail::is_installed())
        {
            return 1;
        }
        torch_trace_collector::detail::push_user_scope(std::string{marker},
                                                       std::string{context},
                                                       std::string{backend});
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" std::int32_t torch_trace_collector_pop_user_scope(void)
{
    try
    {
        return torch_trace_collector::detail::pop_user_scope() ? 0 : 1;
    }
    catch (...)
    {
        return 1;
    }
}

extern "C" std::int32_t torch_trace_collector_get_stats(torch_trace_collector_stats* stats)
{
    if (stats == nullptr || stats->struct_size < sizeof(torch_trace_collector_stats))
    {
        return 1;
    }

    try
    {
        const auto& state = process_state();

        torch_trace_collector_stats result{};
        result.struct_size           = static_cast<std::uint32_t>(sizeof(result));
        result.installed             = torch_trace_collector::detail::is_installed() ? 1U : 0U;
        result.pushes                = state.stats.pushes.load();
        result.pops                  = state.stats.pops.load();
        result.user_scope_pushes     = state.stats.user_scope_pushes.load();
        result.user_scope_pops       = state.stats.user_scope_pops.load();
        result.user_scope_inherits   = state.stats.user_scope_inherits.load();
        result.snapshots_saved       = state.stats.snapshots_saved.load();
        result.snapshots_consumed    = state.stats.snapshots_consumed.load();
        result.snapshots_dropped     = state.stats.snapshots_dropped.load();
        result.snapshots_overwritten = state.stats.snapshots_overwritten.load();
        result.callback_errors       = state.stats.callback_errors.load();
        result.snapshots_pending     = state.snapshots.pending();
        *stats                       = result;
        return 0;
    }
    catch (...)
    {
        return 1;
    }
}
