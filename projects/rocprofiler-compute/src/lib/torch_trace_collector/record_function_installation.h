// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Registration of the callback pair with ATen, and the state that records it.
// at::addGlobalCallback registers process-wide, so the state is one guarded
// global and install() is idempotent.

#pragma once

#include "record_function_callback.h"
#include "snapshot_store.h"
#include "synchronized.hpp"

#include <ATen/record_function.h>

#include <cstdint>
#include <utility>

namespace torch_trace_collector::detail
{

using rocprofiler_compute_tool::common::synchronized_t;

// Global RecordFunction callback registration state.
struct InstallState
{
    at::CallbackHandle handle    = at::INVALID_CALLBACK_HANDLE;
    bool               installed = false;
};

inline synchronized_t<InstallState> g_install;

inline std::int64_t install()
{
    return g_install.wlock(
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

inline void uninstall()
{
    g_install.wlock(
        [](InstallState& state)
        {
            const auto handle = std::exchange(state.handle, at::INVALID_CALLBACK_HANDLE);
            state.installed   = false;
            if (handle != at::INVALID_CALLBACK_HANDLE)
            {
                at::removeCallback(handle);
            }
            // Only the callback consumes snapshots.
            g_snapshots.clear();
        });
}

inline bool is_installed()
{
    return g_install.rlock([](const InstallState& state) { return state.installed; });
}

}  // namespace torch_trace_collector::detail
