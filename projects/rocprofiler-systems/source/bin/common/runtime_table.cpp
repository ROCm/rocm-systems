// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/runtime_table.hpp"

#include "common/string_utility.hpp"

#include <sstream>
#include <utility>
#include <vector>

namespace rocprofsys::runtime_table
{
namespace
{
[[nodiscard]] std::vector<std::string>
split_csv_lowercase(std::string_view input)
{
    std::vector<std::string> tokens;
    std::string              token;
    auto                     stream = std::istringstream{ std::string{ input } };
    while(std::getline(stream, token, ','))
    {
        auto trimmed = utility::string::to_lower(utility::string::trim(token));
        if(!trimmed.empty())
        {
            tokens.push_back(std::move(trimmed));
        }
    }
    return tokens;
}

void
enable_legacy_row(std::map<std::string, std::string>& result, const row& spec)
{
    if(spec.in_legacy_parallel)
    {
        result[std::string{ spec.env_key }] = "true";
    }
}

}  // namespace

std::map<std::string, std::string>
expand_legacy_parallel(std::string_view runtimes)
{
    std::map<std::string, std::string> result;
    // "all" is matched on the raw string. "ALL" is not the all-runtimes form.
    if(runtimes.empty() || runtimes == "all")
    {
        for(const auto& spec : rows())
        {
            enable_legacy_row(result, spec);
        }
        return result;
    }

    for(const auto& token : split_csv_lowercase(runtimes))
    {
        const auto* spec = find(token);
        if(spec == nullptr)
        {
            continue;
        }
        enable_legacy_row(result, *spec);
    }
    return result;
}

}  // namespace rocprofsys::runtime_table
