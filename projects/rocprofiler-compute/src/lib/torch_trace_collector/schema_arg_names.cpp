// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "schema_arg_names.h"

#include "process_state.h"

#include <ATen/core/dispatch/Dispatcher.h>

#include <optional>

namespace torch_trace_collector::detail
{
namespace
{

std::optional<std::vector<std::string>> argument_names_from_dispatcher(const c10::OperatorName& operator_name)
{
    try
    {
        const auto operator_handle = c10::Dispatcher::singleton().findSchema(operator_name);
        std::vector<std::string> argument_names;
        if (!operator_handle.has_value())
        {
            return argument_names;
        }
        const auto& schema_arguments = operator_handle->schema().arguments();
        argument_names.reserve(schema_arguments.size());
        for (const auto& schema_argument : schema_arguments)
        {
            argument_names.push_back(schema_argument.name());
        }
        return argument_names;
    }
    catch (...)
    {
        return std::nullopt;
    }
}

}  // namespace

std::vector<std::string> SchemaArgNamesCache::argument_names(const c10::OperatorName& operator_name)
{
    if (auto cached_argument_names = names_by_operator_.rlock(
            [&](const ArgumentNamesByOperator& names_by_operator) -> std::optional<std::vector<std::string>>
            {
                const auto found = names_by_operator.find(operator_name);
                if (found == names_by_operator.end())
                {
                    return std::nullopt;
                }
                return found->second;
            }))
    {
        return *cached_argument_names;
    }

    const auto dispatcher_argument_names = argument_names_from_dispatcher(operator_name);
    if (!dispatcher_argument_names.has_value())
    {
        return {};
    }

    names_by_operator_.wlock(
        [&](ArgumentNamesByOperator& names_by_operator)
        { names_by_operator.emplace(operator_name, *dispatcher_argument_names); });
    return *dispatcher_argument_names;
}

void SchemaArgNamesCache::clear()
{
    names_by_operator_.wlock([](ArgumentNamesByOperator& names_by_operator)
                             { names_by_operator.clear(); });
}

std::vector<std::string> schema_arg_names(const c10::OperatorName& operator_name)
{
    return process_state().schema_arg_names_cache.argument_names(operator_name);
}

}  // namespace torch_trace_collector::detail
