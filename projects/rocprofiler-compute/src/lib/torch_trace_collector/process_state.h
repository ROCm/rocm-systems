// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// The collector's process-wide state. at::addGlobalCallback registers one
// callback pair for the whole process, and RecordFunction passes the callbacks
// no state of its own, so they reach it through process_state().

#pragma once

#include "capture_buffer.h"
#include "snapshot_store.h"
#include "stats.h"
#include "synchronized.hpp"

#include <ATen/record_function.h>

namespace torch_trace_collector::detail
{

using rocprofiler_compute_tool::common::synchronized_t;

// Records the RecordFunction callback registration.
struct InstallState
{
    at::CallbackHandle handle    = at::INVALID_CALLBACK_HANDLE;
    bool               installed = false;
};

// State shared by every thread in the process. Per-thread state is separate;
// see ThreadState in marker_stack.h.
struct ProcessState
{
    Stats                        stats;
    synchronized_t<InstallState> install;
    CaptureBuffer                capture;
    SnapshotStore                snapshots{stats};
};

// The one instance, constructed on first use. Each binary that links the
// collector holds its own, which is what the callback registration expects.
inline ProcessState& process_state()
{
    static ProcessState state;
    return state;
}

}  // namespace torch_trace_collector::detail
