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

#include "lib/python/rocpd/source/serialization/sql.hpp"
#include "lib/output/sql/common.hpp"

namespace cereal
{
SQLite3InputArchive::SQLite3InputArchive(sqlite3*         conn,
                                         std::string_view query,
                                         int64_t          len,
                                         int64_t          chunk_len)
: InputArchive<SQLite3InputArchive>{this}
, m_conn{conn}
, m_row_count{(len > 0) ? len : getRowCount(conn, query)}
, m_chunk_size{chunk_len}
, m_query{query}
, m_iterator{this}
{
    ROCP_CI_LOG_IF(ERROR,
                   sqlite3_prepare_v2(m_conn, m_query.c_str(), -1, &m_stmt, nullptr) != SQLITE_OK)
        << "Error preparing select statement: " << sqlite3_errmsg(m_conn);

    ROCP_CI_LOG_IF(ERROR, m_stmt == nullptr) << "Error preparing statment: " << query;

    if(!m_stmt) return;

    auto col_count = sqlite3_column_count(m_stmt);

    ROCP_TRACE << " sql query: " << query;
    ROCP_TRACE << "  -  col_count: " << col_count;

    m_column_count = sqlite3_column_count(m_stmt);
    for(int64_t i = 0; i < m_column_count; ++i)
    {
        auto name = impl::extract_column_name(m_stmt, i);
        m_column_names.emplace(name, i);
        ROCP_TRACE << "  - column " << i << ": " << name;
    }

    if(m_chunk_size > 0)
    {
        auto _num_chunks   = (m_row_count / m_chunk_size);
        auto _chunk_modulo = (m_row_count % m_chunk_size);

        m_sizes.resize(_num_chunks, m_chunk_size);
        if(_chunk_modulo > 0) m_sizes.emplace_back(_chunk_modulo);
    }
}

void
SQLite3InputArchive::set_chunk_index(size_t idx)
{
    ROCP_FATAL_IF(idx >= m_sizes.size()) << fmt::format(
        "Invalid chunk index {} (>= {}) for query '{}'", idx, m_sizes.size(), m_query);

    ROCP_TRACE << fmt::format("Setting chunk index to {}. Current index is {}", idx, m_size_idx);
    if(idx != m_size_idx)
    {
        auto _status = sqlite3_reset(m_stmt);

        ROCP_FATAL_IF(_status != SQLITE_OK)
            << fmt::format("sqlite3_reset failed for statement '{}'", m_query);

        for(size_t i = 0; i < (idx * m_chunk_size); ++i)
            ++m_iterator;
        m_size_idx = idx;
    }
}

void
SQLite3InputArchive::loadBinaryValue(void* data, size_t size, const char* name)
{
    m_next_name = name;

    std::string encoded;
    loadValue(encoded);
    auto decoded = base64::decode(encoded);

    if(size != decoded.size())
        throw Exception("Decoded binary data size does not match specified size");

    std::memcpy(data, decoded.data(), decoded.size());
    m_next_name = nullptr;
};

void
SQLite3InputArchive::loadSize(size_type& size) const
{
    if(m_sizes.empty())
        size = m_row_count;
    else if(m_size_idx >= m_sizes.size())
        size = 0;
    else
        size = m_sizes.at(m_size_idx++);

    // ROCP_WARNING << fmt::format(
    //     "[SQLite3InputArchive] counted={}, chunk_size={}, size={}", m_counted, m_chunk_size,
    //     size);
}

int64_t
SQLite3InputArchive::search(std::string_view _col_name)
{
    auto itr = m_column_names.find(_col_name.data());
    if(itr == m_column_names.end())
    {
        auto _names = std::vector<std::string_view>{};
        for(const auto& citr : m_column_names)
            _names.emplace_back(citr.first);
        auto _msg = fmt::format("SQL query '{}' does not contain a column named '{}'. Columns: {}",
                                m_query,
                                m_next_name,
                                fmt::join(_names.begin(), _names.end(), ", "));
        ROCP_WARNING << _msg;
        throw Exception(_msg);
    }
    return itr->second;
}

int64_t
SQLite3InputArchive::search(std::string_view _col_name, std::nothrow_t)
{
    auto itr = m_column_names.find(_col_name.data());
    if(itr == m_column_names.end())
    {
        ROCP_WARNING << fmt::format(
            "SQL query '{}' does not contain a column named '{}'", m_query, m_next_name);
        return -1;
    }
    return itr->second;
}

int64_t
SQLite3InputArchive::search()
{
    auto idx    = search(m_next_name);
    m_next_name = nullptr;
    return idx;
}

int64_t
SQLite3InputArchive::getRowCount(sqlite3* conn, std::string_view query)
{
    return impl::extract_row_count(conn, query);
}
}  // namespace cereal
