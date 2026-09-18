// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ATen/core/operator_name.h>

#include <string>
#include <vector>

namespace torch_trace_collector::detail
{

std::vector<std::string> schema_arg_names(const c10::OperatorName& operator_name);

}  // namespace torch_trace_collector::detail
