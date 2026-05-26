// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "query_builder_base.hpp"

namespace profiler_hub::queries
{

query_builder_base::query_builder_base(std::stringstream& ss)
: m_stream{ ss }
{}

std::string
query_builder_base::get_query_string() const
{
    return m_stream.str();
}

void
query_builder_base::append_keyword(std::string_view keyword)
{
    m_stream << " " << keyword << " ";
}

void
query_builder_base::append_raw(std::string_view text)
{
    m_stream << text;
}

}  // namespace profiler_hub::queries
