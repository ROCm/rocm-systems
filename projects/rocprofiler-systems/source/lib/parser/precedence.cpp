// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/precedence.hpp"
#include "parser/config_store.hpp"

#include <algorithm>
#include <cstdint>
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
    for(const auto& [param_id, item] : resolved)
    {
        store.set(param_id, item.value, item.source);
    }
}

}  // namespace rocprofsys::parser
