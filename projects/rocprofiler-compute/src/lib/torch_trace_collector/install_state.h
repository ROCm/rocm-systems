// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "synchronized.hpp"

#include <ATen/record_function.h>

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

}  // namespace torch_trace_collector::detail
