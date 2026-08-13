// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Definitions for the torch_trace_collector capture hooks.

#include "capture_buffer.h"
#include "capture_control.h"

#include <string>
#include <vector>

namespace torch_trace_collector::detail
{

void start_capture()
{
    g_capture.start();
}

std::vector<std::string> stop_capture()
{
    return g_capture.stop();
}

}  // namespace torch_trace_collector::detail
