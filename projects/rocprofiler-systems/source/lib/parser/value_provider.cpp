// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/value_provider.hpp"
#include "parser/schema.hpp"

#include <algorithm>
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
bool
is_listed(const std::vector<std::string>& choices, std::string_view value)
{
    return std::ranges::find(choices, value) != choices.end();
}

bool
every_element_present(std::string_view value, std::string_view separator)
{
    const auto parts = split_list(value, separator).value_or(std::vector<std::string>{});
    return std::ranges::all_of(
        parts, [](const std::string& element) { return !element.empty(); });
}

bool
elements_are_listed(const std::vector<std::string>& choices, std::string_view value,
                    std::string_view separator)
{
    if(separator.empty())
    {
        return is_listed(choices, value);
    }
    const auto parts = split_list(value, separator).value_or(std::vector<std::string>{});
    if(parts.empty())
    {
        return is_listed(choices, value);
    }
    return std::ranges::all_of(parts, [&choices](const std::string& element) {
        return !element.empty() && is_listed(choices, element);
    });
}
}  // namespace

bool
value_provider_interface::accepts(const schema_param& param, std::string_view value) const
{
    const auto separator = list_separator(param);
    const bool is_list   = param.type == param_type::enumeration_list ||
                         (param.type == param_type::string &&
                          value.find(separator) != std::string_view::npos);
    if(param.allow_custom)
    {
        if(is_list)
        {
            return every_element_present(value, separator);
        }
        return param.type == param_type::string || !value.empty();
    }
    if(value.empty())
    {
        return param.type != param_type::enumeration;
    }
    const auto available = choices(param);
    return is_list ? elements_are_listed(available, value, separator)
                   : is_listed(available, value);
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
