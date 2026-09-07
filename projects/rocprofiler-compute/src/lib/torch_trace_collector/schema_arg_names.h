// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "synchronized.hpp"

#include <ATen/core/operator_name.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace torch_trace_collector::detail
{

using rocprofiler_compute_tool::common::synchronized_t;

// Schema argument names keyed by operator name and overload.
class SchemaArgNamesCache
{
public:
    std::vector<std::string> argument_names(const c10::OperatorName& operator_name);
    void                     clear();

private:
    using ArgumentNamesByOperator = std::unordered_map<c10::OperatorName, std::vector<std::string>>;

    synchronized_t<ArgumentNamesByOperator> names_by_operator_;
};

std::vector<std::string> schema_arg_names(const c10::OperatorName& operator_name);

}  // namespace torch_trace_collector::detail
