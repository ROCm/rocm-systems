// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/python/rocpd/source/serialization/sql.hpp"

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/units.hpp"
#include "lib/output/domain_type.hpp"
#include "lib/output/generator.hpp"
#include "lib/output/sql/common.hpp"

#include <sqlite3.h>

#include <chrono>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace rocpd
{
namespace tool = ::rocprofiler::tool;

template <typename Tp>
struct sql_generator : public tool::generator<Tp>
{
    using base_type = tool::generator<Tp>;

    sql_generator(sqlite3*         conn,
                  std::string_view query,
                  std::string_view order_by   = {},
                  int64_t          chunk_size = compute_chunk_size());

    sql_generator()           = delete;
    ~sql_generator() override = default;

    sql_generator(const sql_generator&)     = delete;
    sql_generator(sql_generator&&) noexcept = delete;
    sql_generator& operator=(const sql_generator&) = delete;
    sql_generator& operator=(sql_generator&&) noexcept = delete;

    std::vector<Tp> get(size_t itr) const override;

    static int64_t compute_chunk_size();

private:
    static std::string sanitize_query(std::string_view query);

private:
    using archive_t = cereal::SQLite3InputArchive;

    sqlite3*            m_conn        = nullptr;
    std::string         m_query       = {};
    std::string         m_order       = {};
    int64_t             m_chunk_size  = 0;
    int64_t             m_num_entries = 0;
    int64_t             m_num_chunks  = 0;
    std::vector<size_t> m_expected    = {};
    archive_t           m_archive;
};

template <typename Tp>
std::string
sql_generator<Tp>::sanitize_query(std::string_view query)
{
    if(auto pos = query.find(';'); pos != std::string_view::npos)
        return std::string{query.substr(0, pos)};

    return std::string{query};
}

template <typename Tp>
int64_t
sql_generator<Tp>::compute_chunk_size()
{
    return (16 * ::rocprofiler::common::units::get_page_size()) / sizeof(Tp);
}

template <typename Tp>
sql_generator<Tp>::sql_generator(sqlite3*         conn,
                                 std::string_view query,
                                 std::string_view order_by,
                                 int64_t          chunk_size)
: base_type{tool::defer_size{}}
, m_conn{conn}
, m_query{sanitize_query(query)}
, m_order{(order_by.empty()) ? std::string{}
                             : fmt::format(" ORDER BY {}", sanitize_query(order_by))}
, m_chunk_size{chunk_size}
, m_num_entries{tool::sql::extract_row_count(m_conn, sanitize_query(query))}
, m_num_chunks{(m_num_entries / m_chunk_size) + ((m_num_entries % m_chunk_size) > 0 ? 1 : 0)}
, m_archive{m_conn, fmt::format("{}{}", m_query, m_order), m_num_entries, m_chunk_size}
{
    base_type::resize(m_num_chunks);

    m_expected.resize(m_num_chunks, m_chunk_size);
    if(!m_expected.empty() && (m_num_entries % m_chunk_size) > 0)
        m_expected.back() = (m_num_entries % m_chunk_size);

    ROCP_TRACE << fmt::format("- Query   : {}", query);
    ROCP_TRACE << fmt::format("  Expected: {}",
                              fmt::join(m_expected.begin(), m_expected.end(), ", "));
}

template <typename Tp>
std::vector<Tp>
sql_generator<Tp>::get(size_t idx) const
{
    auto _data = std::vector<Tp>{};

    if(idx < static_cast<size_t>(m_num_chunks))
    {
        // auto _offset = idx * m_chunk_size;
        // auto _limit  = m_chunk_size;
        // auto _query  = fmt::format("{}{} LIMIT {} OFFSET {};", m_query, m_order, _limit,
        // _offset);

        // auto* conn = const_cast<sqlite3*>(m_conn);
        // auto  ar   = cereal::SQLite3InputArchive{conn, _query};

        auto& ar = const_cast<archive_t&>(m_archive);
        ar.set_chunk_index(idx);

        cereal::load(ar, _data);

        ROCP_FATAL_IF(_data.size() != m_expected.at(idx))
            << fmt::format("Unexpected SQL query result for group {}. Found {} rows. Expected {} "
                           "rows.\nQuery:\n\t{}\n# of entries: {}, chunk size: {}, # chunks: {}",
                           idx,
                           _data.size(),
                           m_expected.at(idx),
                           m_query,
                           m_num_entries,
                           m_chunk_size,
                           m_num_chunks);
    }

    return _data;
}
}  // namespace rocpd
