// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "library/pmc/collectors/gpu_perf_counter/types.hpp"

#include <functional>
#include <map>
#include <span>
#include <string>
#include <string_view>

namespace rocprofsys::pmc::collectors::gpu_perf_counter
{

/**
 * @brief Counter metadata by counter name; resolves the names used in expressions.
 *
 * Instances of one counter (one per dimension) share a definition, so the first
 * instance of each name is kept.
 */
class counter_definitions
{
public:
    counter_definitions() = default;

    explicit counter_definitions(std::span<const counter_metadata> counters)
    {
        for(const auto& meta : counters)
        {
            m_by_name.try_emplace(meta.name, meta);
        }
    }

    /// @brief The definition of @p name, or nullptr if no counter has that name.
    [[nodiscard]] const counter_metadata* find(std::string_view name) const
    {
        auto const itr = m_by_name.find(name);
        return itr == m_by_name.end() ? nullptr : &itr->second;
    }

private:
    std::map<std::string, counter_metadata, std::less<>> m_by_name;
};

}  // namespace rocprofsys::pmc::collectors::gpu_perf_counter
