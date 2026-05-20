// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Engine-bound implementation of parsed_values. This is one of the two
// translation units that knows about tim::argparse — the other is the
// interpreter. A future engine swap means rewriting these specializations
// against the new engine's value-extraction API; nothing else needs to
// change.

#include "parsed_values.hpp"

#include "detail/parser_engine.hpp"

#include <cstdint>
#include <deque>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{
namespace argparse
{

namespace
{
[[nodiscard]] inline std::string
to_key(std::string_view key)
{
    return std::string{ key };
}
}  // namespace

bool
parsed_values::exists(std::string_view key) const
{
    return m_engine->exists(to_key(key));
}

void
parsed_values::set_use_color(bool enabled)
{
    m_engine->set_use_color(enabled);
}

template <>
bool
parsed_values::get<bool>(std::string_view key) const
{
    return m_engine->get<bool>(to_key(key));
}

template <>
int
parsed_values::get<int>(std::string_view key) const
{
    return m_engine->get<int>(to_key(key));
}

template <>
std::int64_t
parsed_values::get<std::int64_t>(std::string_view key) const
{
    return m_engine->get<std::int64_t>(to_key(key));
}

template <>
double
parsed_values::get<double>(std::string_view key) const
{
    return m_engine->get<double>(to_key(key));
}

template <>
std::string
parsed_values::get<std::string>(std::string_view key) const
{
    return m_engine->get<std::string>(to_key(key));
}

template <>
std::set<std::string>
parsed_values::get<std::set<std::string>>(std::string_view key) const
{
    return m_engine->get<std::set<std::string>>(to_key(key));
}

template <>
std::vector<std::string>
parsed_values::get<std::vector<std::string>>(std::string_view key) const
{
    return m_engine->get<std::vector<std::string>>(to_key(key));
}

template <>
std::deque<std::string>
parsed_values::get<std::deque<std::string>>(std::string_view key) const
{
    return m_engine->get<std::deque<std::string>>(to_key(key));
}

template <>
std::vector<std::int64_t>
parsed_values::get<std::vector<std::int64_t>>(std::string_view key) const
{
    return m_engine->get<std::vector<std::int64_t>>(to_key(key));
}

}  // namespace argparse
}  // namespace rocprofsys
