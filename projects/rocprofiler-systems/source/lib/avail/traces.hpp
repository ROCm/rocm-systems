// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/records.hpp"

#include "common/delimit.hpp"
#include "common/string_utility.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofsys::avail
{

struct traces_listing_result
{
    std::vector<trace_record> records;
    std::vector<std::string>  defaults;
    std::optional<diagnostic> issue;
};

struct operations_listing_result
{
    std::vector<operation_record> records;
    std::optional<diagnostic>     issue;
};

[[nodiscard]] inline std::optional<std::string>
domain_from_operations_env(std::string_view env_name)
{
    constexpr std::string_view k_prefix = "ROCPROFSYS_ROCM_";
    constexpr std::string_view k_suffix = "_OPERATIONS";
    if(env_name.size() <= k_prefix.size() + k_suffix.size())
    {
        return std::nullopt;
    }
    if(!env_name.starts_with(k_prefix))
    {
        return std::nullopt;
    }
    if(!env_name.ends_with(k_suffix))
    {
        return std::nullopt;
    }

    const auto domain = env_name.substr(
        k_prefix.size(), env_name.size() - k_prefix.size() - k_suffix.size());
    if(domain.empty())
    {
        return std::nullopt;
    }
    return utility::string::to_lower(domain);
}

[[nodiscard]] inline std::string
unknown_trace_message(std::string_view name)
{
    return "unknown trace '" + std::string{ name } + "'";
}

[[nodiscard]] inline std::string
trace_has_no_operations_message(std::string_view name)
{
    return "trace '" + std::string{ name } + "' has no operations";
}

namespace inventory
{

[[nodiscard]] inline std::vector<std::string>
parse_domain_list(const std::string& csv)
{
    auto names = rocprofsys::delimit(csv, ",");
    for(auto& name : names)
    {
        utility::string::to_lower_in_place(name);
    }
    return names;
}

[[nodiscard]] inline bool
contains_choice(const std::vector<std::string>& choices, std::string_view requested)
{
    return std::ranges::any_of(
        choices, [requested](const std::string& choice) { return choice == requested; });
}

inline void
append_group_alias_to_member(std::vector<trace_record>&                          records,
                             const std::unordered_map<std::string, std::size_t>& by_name,
                             std::string_view group_name, std::string_view member)
{
    const auto found = by_name.find(std::string{ member });
    if(found == by_name.end())
    {
        return;
    }
    auto& leaf = records[found->second];
    if(leaf.name == group_name || contains_choice(leaf.aliases, group_name))
    {
        return;
    }
    leaf.aliases.emplace_back(group_name);
}

inline void
add_group_name_to_members(std::vector<trace_record>& records)
{
    std::unordered_map<std::string, std::size_t> by_name;
    by_name.reserve(records.size());
    for(std::size_t idx = 0; idx < records.size(); ++idx)
    {
        by_name.emplace(records[idx].name, idx);
    }

    for(const auto& group : records)
    {
        for(const auto& member : group.aliases)
        {
            append_group_alias_to_member(records, by_name, group.name, member);
        }
    }
}

template <typename TracingConfig>
[[nodiscard]] traces_listing_result
traces()
{
    traces_listing_result result;
    const auto            choices = TracingConfig::get_domain_choices();
    result.defaults = parse_domain_list(TracingConfig::get_domain_defaults());
    const auto default_set =
        std::unordered_set<std::string>{ result.defaults.begin(), result.defaults.end() };

    result.records.reserve(choices.size());
    for(const auto& name : choices)
    {
        result.records.push_back(trace_record{
            .name        = name,
            .description = TracingConfig::get_domain_description(name),
            .aliases     = TracingConfig::get_domain_members(name),
            .is_default  = default_set.contains(name),
        });
    }
    add_group_name_to_members(result.records);
    return result;
}

[[nodiscard]] inline bool
contains_operation(const std::vector<operation_record>& records,
                   std::string_view trace_name, std::string_view operation_name)
{
    return std::ranges::any_of(
        records, [trace_name, operation_name](const operation_record& existing) {
            return existing.name == operation_name && existing.trace_name == trace_name;
        });
}

template <typename OperationSpec>
void
append_unique_operations(operations_listing_result& result, std::string_view trace_name,
                         const OperationSpec& spec)
{
    for(const auto& operation_name : spec.operation_choices)
    {
        if(contains_operation(result.records, trace_name, operation_name))
        {
            continue;
        }
        result.records.push_back(operation_record{
            .trace_name  = std::string{ trace_name },
            .name        = operation_name,
            .description = {},
        });
    }
}

template <typename TracingConfig>
[[nodiscard]] operations_listing_result
operations(std::string_view name)
{
    const auto requested = utility::string::to_lower(name);
    const auto known = contains_choice(TracingConfig::get_domain_choices(), requested);

    operations_listing_result result;
    for(const auto& spec : TracingConfig::get_operation_settings())
    {
        const auto domain =
            domain_from_operations_env(spec.env_names.operations_include_env_name);
        if(domain != requested)
        {
            continue;
        }
        append_unique_operations(result, requested, spec);
    }

    if(!result.records.empty())
    {
        return result;
    }

    result.issue = diagnostic{
        .capability = capability_kind::trace_operations,
        .source     = source_id::rocprofiler_sdk,
        .message    = known ? trace_has_no_operations_message(requested)
                            : unknown_trace_message(requested),
    };
    return result;
}

}  // namespace inventory

[[nodiscard]] traces_listing_result
query_traces();

[[nodiscard]] operations_listing_result
query_operations(std::string_view name);

}  // namespace rocprofsys::avail
