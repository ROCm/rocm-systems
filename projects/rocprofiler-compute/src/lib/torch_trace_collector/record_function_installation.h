// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Registration of the callback pair with ATen. at::addGlobalCallback registers
// process-wide, so install() is idempotent and its state lives in
// process_state().

#pragma once

#include <cstdint>

namespace torch_trace_collector::detail
{

std::int64_t install();
void         uninstall();
bool         is_installed();

}  // namespace torch_trace_collector::detail
