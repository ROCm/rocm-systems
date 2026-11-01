// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/precedence.hpp"
#include "parser/config_store.hpp"
#include "parser/schema.hpp"
#include "parser/schema_validator.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
[[nodiscard]] constexpr std::uint8_t
source_rank(config_source source) noexcept
{
    return static_cast<std::uint8_t>(source);
}

std::vector<std::string>
rejection_messages(const schema& config_schema, const resolved_config& resolved)
{
    std::vector<std::string> errors;
    for(const auto& [param_id, item] : resolved)
    {
        const auto* param = config_schema.find_param(param_id);
        if(param == nullptr)
        {
            errors.push_back(fmt::format("unknown parameter '{}'", param_id));
            continue;
        }
        if(!is_valid_value(*param, item.value))
        {
            errors.push_back(fmt::format("invalid value '{}' for parameter '{}'",
                                         item.value, param_id));
        }
    }
    return errors;
}
}  // namespace

resolved_config
resolve_precedence(std::vector<sourced_values> sources)
{
    // libstdc++ has no std::ranges::stable_sort.
    std::stable_sort(  // NOLINT(modernize-use-ranges)
        sources.begin(), sources.end(),
        [](const sourced_values& lhs, const sourced_values& rhs) {
            return source_rank(lhs.source) < source_rank(rhs.source);
        });

    resolved_config resolved;
    for(const auto& item : sources)
    {
        for(const auto& [param_id, value] : item.values)
        {
            resolved.insert_or_assign(
                param_id, resolved_value{ .value = value, .source = item.source });
        }
    }
    return resolved;
}

void
apply_resolved(config_store& store, const resolved_config& resolved)
{
    const auto config_schema = store.config_schema();
    if(!config_schema)
    {
        throw std::runtime_error("configuration store has no schema");
    }
    const auto errors = rejection_messages(*config_schema, resolved);
    if(!errors.empty())
    {
        throw std::runtime_error(fmt::format("{}", fmt::join(errors, "; ")));
    }
    for(const auto& [param_id, item] : resolved)
    {
        store.set(param_id, item.value, item.source);
    }
}

}  // namespace rocprofsys::parser
