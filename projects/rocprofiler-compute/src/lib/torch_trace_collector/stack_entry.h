// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

namespace torch_trace_collector::detail
{

struct StackEntry
{
    std::string marker;
    std::string context;
};

}  // namespace torch_trace_collector::detail
