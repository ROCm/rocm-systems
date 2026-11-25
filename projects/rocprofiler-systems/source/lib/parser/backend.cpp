// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "parser/backend.hpp"
#include "parser/schema.hpp"
#include "parser/value_text.hpp"

#include <fmt/format.h>

#include <memory>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace rocprofsys::parser
{
bool
backend_interface::accepts(const schema_param& param, std::string_view value) const
{
    if(param.backend != name())
    {
        throw std::logic_error(
            fmt::format("parameter '{}' does not name backend '{}'", param.id, name()));
    }
    // Without the backend name, parse_value() checks every name against the list.
    auto       listed = param;
    const auto names  = choices(param);
    listed.backend.reset();
    listed.choices.insert(listed.choices.end(), names.begin(), names.end());
    return accepts_text(listed, value);
}

bool
backend_registry::add(std::unique_ptr<backend_interface> item)
{
    if(!item || item->name().empty())
    {
        return false;
    }
    const auto                                name = std::string{ item->name() };
    const std::unique_lock<std::shared_mutex> lock{ m_mutex };
    if(m_backends.contains(name))
    {
        return false;
    }
    m_backends.emplace(name, std::move(item));
    return true;
}

const backend_interface*
backend_registry::find(std::string_view name) const
{
    const std::shared_lock<std::shared_mutex> lock{ m_mutex };
    const auto                                found = m_backends.find(name);
    return found == m_backends.end() ? nullptr : found->second.get();
}

}  // namespace rocprofsys::parser
