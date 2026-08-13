// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>
#include <vector>

namespace torch_trace_collector::detail
{

// Start and stop recording of the emitted wire strings.
void                     start_capture();
std::vector<std::string> stop_capture();

}  // namespace torch_trace_collector::detail
