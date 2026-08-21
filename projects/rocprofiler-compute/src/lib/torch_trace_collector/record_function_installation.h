// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Registration of the callback pair with ATen. at::addGlobalCallback registers
// process-wide, so install() is idempotent and its state lives in
// process_state().

#pragma once

#include "args_capture.h"
#include "process_state.h"
#include "record_function_callback.h"

#include <ATen/record_function.h>

#include <cstdint>
#include <utility>

namespace torch_trace_collector::detail
{

inline std::int64_t install(bool capture_args = true, bool capture_values = false)
{
    g_args_capture.capture_args.store(capture_args);
    g_args_capture.capture_values.store(capture_values);
    return process_state().install.wlock(
        [capture_args](InstallState& state)
        {
            if (state.handle != at::INVALID_CALLBACK_HANDLE)
            {
                // needsInputs is fixed when the callback is registered and cannot be
                // changed in place; call uninstall() before install() to change it.
                return static_cast<std::int64_t>(state.handle);
            }
            auto callback = at::RecordFunctionCallback(start_cb, end_cb)
                                .scopes({at::RecordScope::FUNCTION, at::RecordScope::BACKWARD_FUNCTION});
            if (capture_args)
            {
                callback.needsInputs(true);
            }
            state.handle    = at::addGlobalCallback(callback);
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
