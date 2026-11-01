// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/value_provider.hpp"
#include "parser/schema.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofsys::parser
{
namespace
{
// Same delimiter set tracing_config uses for ROCPROFSYS_ROCM_DOMAINS.
constexpr std::string_view k_list_delimiters = " ,;:\t\n";

bool
is_listed(const std::vector<std::string>& choices, std::string_view value)
{
    return std::ranges::find(choices, value) != choices.end();
}

[[nodiscard]] bool
contains_delimiter(std::string_view value)
{
    return value.find_first_of(k_list_delimiters) != std::string_view::npos;
}

std::vector<std::string>
split_provider_list(std::string_view value)
{
    std::vector<std::string> parts;
    std::size_t              begin = 0;
    while(begin < value.size())
    {
        begin = value.find_first_not_of(k_list_delimiters, begin);
        if(begin == std::string_view::npos)
        {
            break;
        }
        const auto end = value.find_first_of(k_list_delimiters, begin);
        const auto count =
            end == std::string_view::npos ? std::string_view::npos : end - begin;
        const auto token = value.substr(begin, count);
        if(!token.empty())
        {
            parts.emplace_back(token);
        }
        if(end == std::string_view::npos)
        {
            break;
        }
        begin = end;
    }
    return parts;
}

bool
every_element_present(std::string_view value)
{
    if(value.empty())
    {
        return true;
    }
    return !split_provider_list(value).empty();
}

bool
elements_are_listed(const std::vector<std::string>& choices, std::string_view value)
{
    const auto parts = split_provider_list(value);
    if(parts.empty())
    {
        return is_listed(choices, value);
    }
    return std::ranges::all_of(parts, [&choices](const std::string& element) {
        return is_listed(choices, element);
    });
}
}  // namespace

bool
value_provider_interface::accepts(const schema_param& param, std::string_view value) const
{
    const bool is_list = param.type == param_type::enumeration_list ||
                         (param.type == param_type::string && contains_delimiter(value));
    if(param.allow_custom)
    {
        if(is_list)
        {
            return every_element_present(value);
        }
        return param.type == param_type::string || !value.empty();
    }
    if(value.empty())
    {
        return param.type != param_type::enumeration;
    }
    const auto available = choices(param);
    return is_list ? elements_are_listed(available, value) : is_listed(available, value);
}

bool
value_provider_registry::add(std::unique_ptr<value_provider_interface> provider)
{
    if(!provider || provider->name().empty())
    {
        return false;
    }
    const auto                                name = std::string{ provider->name() };
    const std::unique_lock<std::shared_mutex> lock{ m_mutex };
    if(m_providers.contains(name))
    {
        return false;
    }
    m_providers.emplace(name, std::move(provider));
    return true;
}

const value_provider_interface*
value_provider_registry::find(std::string_view name) const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    const auto                                found = m_providers.find(name);
    return found == m_providers.end() ? nullptr : found->second.get();
}

}  // namespace rocprofsys::parser
