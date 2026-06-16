// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sample_types.h"

#include <utility>

using namespace rocprofiler_compute_tool;

size_t pc_string_table_t::insert(const std::string& instruction_text, const std::string& comment)
{
    // Look up via views over the caller's strings so a cache hit copies nothing.
    if (const auto it = m_index.find(
            std::make_pair(std::string_view{instruction_text}, std::string_view{comment}));
        it != m_index.end())
        return it->second;

    const size_t idx = m_instructions.size();
    // deque element addresses are stable, so the views stored below stay valid.
    const std::string& stored_text    = m_instructions.emplace_back(instruction_text);
    const std::string& stored_comment = m_comments.emplace_back(comment);
    m_index.emplace(std::make_pair(std::string_view{stored_text}, std::string_view{stored_comment}), idx);
    return idx;
}

const std::deque<std::string>& pc_string_table_t::instructions() const
{
    return m_instructions;
}

const std::deque<std::string>& pc_string_table_t::comments() const
{
    return m_comments;
}
