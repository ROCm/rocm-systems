// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "schema_arg_names.h"

#include <ATen/core/dispatch/Dispatcher.h>

#include <mutex>
#include <optional>
#include <unordered_map>

namespace torch_trace_collector::detail
{
namespace
{

std::mutex g_schema_arg_names_mutex;

std::unordered_map<std::string, std::vector<std::string>> g_schema_arg_names;

std::string operator_key(const c10::OperatorName& operator_name)
{
    return operator_name.name + '\0' + operator_name.overload_name;
}

std::optional<std::vector<std::string>> argument_names_from_dispatcher(
    const c10::OperatorName& operator_name)
{
    try
    {
        const auto operator_handle = c10::Dispatcher::singleton().findSchema(operator_name);
        std::vector<std::string> argument_names;
        if(!operator_handle.has_value())
        {
            return argument_names;
        }
        const auto& schema_arguments = operator_handle->schema().arguments();
        argument_names.reserve(schema_arguments.size());
        for(const auto& schema_argument : schema_arguments)
        {
            argument_names.push_back(schema_argument.name());
        }
        return argument_names;
    }
    catch(...)
    {
        return std::nullopt;
    }
}

}  // namespace

std::vector<std::string> schema_arg_names(const c10::OperatorName& operator_name)
{
    const std::string key = operator_key(operator_name);
    {
        std::lock_guard<std::mutex> lock(g_schema_arg_names_mutex);
        const auto                  found = g_schema_arg_names.find(key);
        if(found != g_schema_arg_names.end())
        {
            return found->second;
        }
    }

    const auto dispatcher_argument_names = argument_names_from_dispatcher(operator_name);
    const std::vector<std::string> names =
        dispatcher_argument_names.value_or(std::vector<std::string>{});

    std::lock_guard<std::mutex> lock(g_schema_arg_names_mutex);
    return g_schema_arg_names.emplace(key, names).first->second;
}

}  // namespace torch_trace_collector::detail
