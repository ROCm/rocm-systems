// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/backend.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
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

[[nodiscard]] bool
is_listed(const std::vector<std::string>& choices, std::string_view value)
{
    for(const auto& choice : choices)
    {
        if(choice == value)
        {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool
contains_delimiter(std::string_view value)
{
    return value.find_first_of(k_list_delimiters) != std::string_view::npos;
}

[[nodiscard]] std::vector<std::string>
split_backend_list(std::string_view value)
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

[[nodiscard]] bool
has_an_element(std::string_view value)
{
    if(value.empty())
    {
        return true;
    }
    return !split_backend_list(value).empty();
}

[[nodiscard]] bool
elements_are_listed(const std::vector<std::string>& choices, std::string_view value)
{
    const auto parts = split_backend_list(value);
    if(parts.empty())
    {
        return is_listed(choices, value);
    }
    for(const auto& element : parts)
    {
        if(!is_listed(choices, element))
        {
            return false;
        }
    }
    return true;
}
}  // namespace

bool
backend::accepts(const schema_param& param, std::string_view value) const
{
    const bool is_list = param.type == param_type::enumeration_list ||
                         (param.type == param_type::string && contains_delimiter(value));
    if(param.allow_custom)
    {
        if(is_list)
        {
            return has_an_element(value);
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
backend_registry::add(std::unique_ptr<backend> item)
{
    if(!item || item->name().empty())
    {
        return false;
    }
    const auto                                name = std::string{ item->name() };
    const std::unique_lock<std::shared_mutex> lock{ m_mutex };
    if(m_backends.find(name) != m_backends.end())
    {
        return false;
    }
    m_backends.emplace(name, std::move(item));
    return true;
}

const backend*
backend_registry::find(std::string_view name) const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    const auto                                found = m_backends.find(name);
    return found == m_backends.end() ? nullptr : found->second.get();
}

}  // namespace rocprofsys::parser
