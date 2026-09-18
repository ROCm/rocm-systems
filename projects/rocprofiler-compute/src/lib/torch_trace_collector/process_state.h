// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Process-wide collector state.

#pragma once

#include "snapshot_store.h"
#include "stats.h"
#include "synchronized.hpp"

#include <ATen/record_function.h>

namespace torch_trace_collector::detail
{

using rocprofiler_compute_tool::common::synchronized_t;

// One global callback per observed scope.
struct InstallState
{
    at::CallbackHandle forward_handle  = at::INVALID_CALLBACK_HANDLE;
    at::CallbackHandle backward_handle = at::INVALID_CALLBACK_HANDLE;
};

struct ProcessState
{
    Stats                        stats;
    synchronized_t<InstallState> install;
    SnapshotStore                snapshots{stats};
};

ProcessState& process_state();

}  // namespace torch_trace_collector::detail
