// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ATen/record_function.h>

#include <atomic>
#include <mutex>

namespace torch_trace_collector::detail
{

// Global RecordFunction callback registration state.
struct InstallState
{
    std::atomic<at::CallbackHandle> handle{at::INVALID_CALLBACK_HANDLE};
    std::atomic<bool>               installed{false};
    std::mutex                      mutex;
};

inline InstallState g_install;

}  // namespace torch_trace_collector::detail
