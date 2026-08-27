// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// The RecordFunction callback pair. start_cb builds the marker path for one ATen
// operator and emits a ROCTX range; end_cb unwinds what start_cb pushed. Neither
// throws, so an error inside either is counted and swallowed.

#pragma once

#include <ATen/record_function.h>

#include <cstddef>
#include <memory>

namespace torch_trace_collector::detail
{

// Records what start_cb pushed so end_cb can unwind it.
struct RoctxObserverContext : public at::ObserverContext
{
    bool        pushed_roctx_range     = false;
    bool        pushed_leaf            = false;
    std::size_t pushed_snapshot_frames = 0;
};

std::unique_ptr<at::ObserverContext> start_cb(const at::RecordFunction& record_fn);
void end_cb(const at::RecordFunction& record_fn, at::ObserverContext* obs_ctx);

}  // namespace torch_trace_collector::detail
