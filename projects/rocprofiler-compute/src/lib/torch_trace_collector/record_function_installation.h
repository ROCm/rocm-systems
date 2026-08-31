// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Global RecordFunction callback registration.

#pragma once

#include <cstdint>

namespace torch_trace_collector::detail
{

std::int64_t install(bool capture_args = true, bool capture_values = false);
void         uninstall();
bool         is_installed();

}  // namespace torch_trace_collector::detail
