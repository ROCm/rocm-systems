// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Definitions for the torch_trace_collector capture hooks.

#include "capture_buffer.h"
#include "process_state.h"

#include <string>
#include <vector>

namespace torch_trace_collector::detail
{

void start_capture()
{
    process_state().capture.start();
}

std::vector<std::string> stop_capture()
{
    return process_state().capture.stop();
}

}  // namespace torch_trace_collector::detail
