// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Registration of the callback pair with ATen. at::addGlobalCallback registers
// process-wide, so install() is idempotent and its state lives in
// process_state().

#pragma once

#include "process_state.h"
#include "record_function_callback.h"

#include <ATen/record_function.h>

#include <cstdint>
#include <utility>

namespace torch_trace_collector::detail
{

inline std::int64_t install()
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

inline void uninstall()
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
            // Only the callback consumes snapshots.
            process_state().snapshots.clear();
        });
}

inline bool is_installed()
{
    return process_state().install.rlock([](const InstallState& state) { return state.installed; });
}

}  // namespace torch_trace_collector::detail
