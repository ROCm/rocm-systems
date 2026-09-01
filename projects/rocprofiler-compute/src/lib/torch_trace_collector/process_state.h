// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Process-wide collector state.

#pragma once

#include "schema_arg_names.h"
#include "snapshot_store.h"
#include "stats.h"
#include "synchronized.hpp"

#include <ATen/record_function.h>

namespace torch_trace_collector::detail
{

using rocprofiler_compute_tool::common::synchronized_t;

struct InstallState
{
    at::CallbackHandle handle = at::INVALID_CALLBACK_HANDLE;
};

struct ProcessState
{
    Stats                        stats;
    synchronized_t<InstallState> install;
    SnapshotStore                snapshots{stats};
    SchemaArgNamesCache          schema_arg_names_cache;
};

ProcessState& process_state();

}  // namespace torch_trace_collector::detail
