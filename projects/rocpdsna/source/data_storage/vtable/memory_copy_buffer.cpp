// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "memory_copy_buffer.hpp"

#include "debug.hpp"

#include <sqlite3.h>

#include <array>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace rocpdsna::data_storage::vtable
{

namespace
{

constexpr std::array<const char*, memory_copy_buffer::k_total_columns> k_column_names = {
    "id",          "nid",     "pid",          "tid",         "start",
    "end",         "name_id", "dst_agent_id", "dst_address", "src_agent_id",
    "src_address", "size",    "queue_id",     "stream_id",   "region_name_id",
    "event_id",    "extdata"
};

std::mutex&
registry_mutex()
{
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, memory_copy_buffer*>&
registry()
{
    static std::unordered_map<std::string, memory_copy_buffer*> r;
    return r;
}

}  // namespace

memory_copy_buffer::memory_copy_buffer(std::string real_table_name, std::string db_path)
: m_real_table_name(std::move(real_table_name))
, m_db_path(std::move(db_path))
{
    std::string columns;
    std::string placeholders;
    for(size_t i = 0; i < k_column_names.size(); ++i)
    {
        if(i != 0)
        {
            columns += ',';
            placeholders += ',';
        }
        columns += '"';
        columns += k_column_names[i];
        columns += '"';
        placeholders += '?';
    }
    m_insert_sql = "INSERT INTO \"" + m_real_table_name + "\" (" + columns +
                   ") VALUES (" + placeholders + ")";
}

memory_copy_buffer::~memory_copy_buffer()
{
    flush();
    if(m_insert_stmt != nullptr) sqlite3_finalize(m_insert_stmt);
    if(m_writer_conn != nullptr) sqlite3_close(m_writer_conn);
}

void
memory_copy_buffer::push(int64_t                id,
                         int64_t                nid,
                         int64_t                pid,
                         std::optional<int64_t> tid,
                         int64_t                start,
                         int64_t                end,
                         int64_t                name_id,
                         std::optional<int64_t> dst_agent_id,
                         std::optional<int64_t> dst_address,
                         std::optional<int64_t> src_agent_id,
                         std::optional<int64_t> src_address,
                         int64_t                size,
                         std::optional<int64_t> queue_id,
                         std::optional<int64_t> stream_id,
                         std::optional<int64_t> region_name_id,
                         std::optional<int64_t> event_id,
                         std::string_view       extdata)
{
    auto push_int_at = [&](size_t col_idx, int64_t v) {
        m_int_cols[col_idx].values.push_back(v);
        m_int_cols[col_idx].is_null.push_back(0);
    };

    auto push_opt_at = [&](size_t col_idx, std::optional<int64_t> v) {
        if(v.has_value())
        {
            m_int_cols[col_idx].values.push_back(*v);
            m_int_cols[col_idx].is_null.push_back(0);
        }
        else
        {
            m_int_cols[col_idx].values.push_back(0);
            m_int_cols[col_idx].is_null.push_back(1);
        }
    };

    push_int_at(0, id);
    push_int_at(1, nid);
    push_int_at(2, pid);
    push_opt_at(3, tid);
    push_int_at(4, start);
    push_int_at(5, end);
    push_int_at(6, name_id);
    push_opt_at(7, dst_agent_id);
    push_opt_at(8, dst_address);
    push_opt_at(9, src_agent_id);
    push_opt_at(10, src_address);
    push_int_at(11, size);
    push_opt_at(12, queue_id);
    push_opt_at(13, stream_id);
    push_opt_at(14, region_name_id);
    push_opt_at(15, event_id);

    m_text_col.values.emplace_back(extdata);
    m_text_col.is_null.push_back(0);

    ++m_row_count;
}

void
memory_copy_buffer::push_from_values(sqlite3_value** argv)
{
    for(size_t c = 0; c < k_int_column_count; ++c)
    {
        sqlite3_value* v   = argv[2 + static_cast<int>(c)];
        auto&          col = m_int_cols[c];
        if(sqlite3_value_type(v) == SQLITE_NULL)
        {
            col.values.push_back(0);
            col.is_null.push_back(1);
        }
        else
        {
            col.values.push_back(sqlite3_value_int64(v));
            col.is_null.push_back(0);
        }
    }
    sqlite3_value* v = argv[2 + static_cast<int>(k_text_column_index)];
    if(sqlite3_value_type(v) == SQLITE_NULL)
    {
        m_text_col.values.emplace_back();
        m_text_col.is_null.push_back(1);
    }
    else
    {
        const auto* txt = reinterpret_cast<const char*>(sqlite3_value_text(v));
        const auto  n   = static_cast<size_t>(sqlite3_value_bytes(v));
        m_text_col.values.emplace_back(txt != nullptr ? std::string(txt, n)
                                                      : std::string{});
        m_text_col.is_null.push_back(0);
    }
    ++m_row_count;
}

int
memory_copy_buffer::prepare_insert_stmt()
{
    if(m_insert_stmt != nullptr) return SQLITE_OK;

    if(m_writer_conn == nullptr)
    {
        if(m_db_path.empty() || m_db_path == ":memory:")
        {
            LOG_ERROR("buffer: bulk writer requires on-disk db, got '{}'", m_db_path);
            return SQLITE_ERROR;
        }
        int rc = sqlite3_open(m_db_path.c_str(), &m_writer_conn);
        if(rc != SQLITE_OK)
        {
            LOG_ERROR("buffer: failed to open bulk writer conn for '{}': {}",
                      m_db_path,
                      sqlite3_errmsg(m_writer_conn));
            return rc;
        }
        sqlite3_exec(m_writer_conn, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr);
        sqlite3_exec(
            m_writer_conn, "PRAGMA synchronous=NORMAL", nullptr, nullptr, nullptr);
        sqlite3_exec(m_writer_conn, "PRAGMA foreign_keys=OFF", nullptr, nullptr, nullptr);
        sqlite3_exec(
            m_writer_conn, "PRAGMA cache_size=-65536", nullptr, nullptr, nullptr);
        sqlite3_exec(
            m_writer_conn, "PRAGMA temp_store=MEMORY", nullptr, nullptr, nullptr);
        sqlite3_busy_timeout(m_writer_conn, 5000);
    }

    int rc = sqlite3_prepare_v2(
        m_writer_conn, m_insert_sql.c_str(), -1, &m_insert_stmt, nullptr);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("buffer: failed to prepare insert: {} sql={}",
                  sqlite3_errmsg(m_writer_conn),
                  m_insert_sql);
        return rc;
    }
    return SQLITE_OK;
}

int
memory_copy_buffer::flush()
{
    if(m_row_count == 0) return SQLITE_OK;

    int rc = prepare_insert_stmt();
    if(rc != SQLITE_OK) return rc;

    char* err = nullptr;
    rc        = sqlite3_exec(m_writer_conn, "BEGIN", nullptr, nullptr, &err);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("buffer: BEGIN failed: {}", err != nullptr ? err : "?");
        sqlite3_free(err);
        return rc;
    }

    sqlite3_stmt* stmt = m_insert_stmt;
    for(size_t r = 0; r < m_row_count; ++r)
    {
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);

        for(size_t c = 0; c < k_int_column_count; ++c)
        {
            const int pos = static_cast<int>(c) + 1;
            if(m_int_cols[c].is_null[r] != 0u)
            {
                sqlite3_bind_null(stmt, pos);
            }
            else
            {
                sqlite3_bind_int64(stmt, pos, m_int_cols[c].values[r]);
            }
        }
        const int text_pos = static_cast<int>(k_text_column_index) + 1;
        if(m_text_col.is_null[r] != 0u)
        {
            sqlite3_bind_null(stmt, text_pos);
        }
        else
        {
            const std::string& s = m_text_col.values[r];
            sqlite3_bind_text(
                stmt, text_pos, s.data(), static_cast<int>(s.size()), SQLITE_TRANSIENT);
        }

        rc = sqlite3_step(stmt);
        if(rc != SQLITE_DONE)
        {
            LOG_ERROR(
                "buffer: step failed at row {}: {}", r, sqlite3_errmsg(m_writer_conn));
            sqlite3_exec(m_writer_conn, "ROLLBACK", nullptr, nullptr, nullptr);
            return rc;
        }
    }

    rc = sqlite3_exec(m_writer_conn, "COMMIT", nullptr, nullptr, &err);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("buffer: COMMIT failed: {}", err != nullptr ? err : "?");
        sqlite3_free(err);
        sqlite3_exec(m_writer_conn, "ROLLBACK", nullptr, nullptr, nullptr);
        return rc;
    }

    for(auto& col : m_int_cols)
    {
        col.values.clear();
        col.is_null.clear();
    }
    m_text_col.values.clear();
    m_text_col.is_null.clear();
    m_row_count = 0;
    return SQLITE_OK;
}

void
memory_copy_buffer::reserve(std::size_t expected_rows)
{
    for(auto& col : m_int_cols)
    {
        col.values.reserve(expected_rows);
        col.is_null.reserve(expected_rows);
    }
    m_text_col.values.reserve(expected_rows);
    m_text_col.is_null.reserve(expected_rows);
}

void
memory_copy_buffer::register_instance(const std::string&  real_table_name,
                                      memory_copy_buffer* buffer)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[real_table_name] = buffer;
}

void
memory_copy_buffer::unregister_instance(const std::string& real_table_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(real_table_name);
}

memory_copy_buffer*
memory_copy_buffer::get_active_instance(const std::string& real_table_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto                        it = registry().find(real_table_name);
    return (it != registry().end()) ? it->second : nullptr;
}

}  // namespace rocpdsna::data_storage::vtable
