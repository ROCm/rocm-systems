// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "sample_buffer.hpp"

#include "debug.hpp"

#include <sqlite3.h>

#include <array>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace profiler_hub::data_storage::vtable
{

namespace
{

// Column order for INSERT: id, track_id, timestamp, event_id, extdata.
// INT columns: id(0), track_id(1), timestamp(2), event_id(3).
// TEXT columns: extdata(4).
constexpr std::array<const char*, sample_buffer::k_total_columns>
    k_column_names = { "id", "track_id", "timestamp", "event_id", "extdata" };

std::mutex&
registry_mutex()
{
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, sample_buffer*>&
registry()
{
    static std::unordered_map<std::string, sample_buffer*> r;
    return r;
}

}  // namespace

sample_buffer::sample_buffer(std::string real_table_name, sqlite3* writer_conn)
: m_real_table_name(std::move(real_table_name))
, m_writer_conn(writer_conn)
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

sample_buffer::~sample_buffer()
{
    flush();
    if(m_insert_stmt != nullptr) sqlite3_finalize(m_insert_stmt);
    // m_writer_conn is owned by sqlite_backend; do not close it here.
}

void
sample_buffer::push(const sample_row& row)
{
    // id
    m_int_cols[0].values.push_back(row.id);
    m_int_cols[0].is_null.push_back(0);

    // track_id
    m_int_cols[1].values.push_back(row.track_id);
    m_int_cols[1].is_null.push_back(0);

    // timestamp
    m_int_cols[2].values.push_back(row.timestamp);
    m_int_cols[2].is_null.push_back(0);

    // event_id - nullable
    if(row.event_id.has_value())
    {
        m_int_cols[3].values.push_back(*row.event_id);
        m_int_cols[3].is_null.push_back(0);
    }
    else
    {
        m_int_cols[3].values.push_back(0);
        m_int_cols[3].is_null.push_back(1);
    }

    // extdata - not null
    m_text_cols[0].values.emplace_back(row.extdata);
    m_text_cols[0].is_null.push_back(0);

    ++m_row_count;
}

void
sample_buffer::push_from_values(sqlite3_value** argv)
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
    for(size_t c = 0; c < k_text_column_count; ++c)
    {
        sqlite3_value* v = argv[2 + static_cast<int>(k_int_column_count + c)];
        if(sqlite3_value_type(v) == SQLITE_NULL)
        {
            m_text_cols[c].values.emplace_back();
            m_text_cols[c].is_null.push_back(1);
        }
        else
        {
            const auto* txt = reinterpret_cast<const char*>(sqlite3_value_text(v));
            const auto  n   = static_cast<size_t>(sqlite3_value_bytes(v));
            m_text_cols[c].values.emplace_back(txt != nullptr ? std::string(txt, n)
                                                              : std::string{});
            m_text_cols[c].is_null.push_back(0);
        }
    }
    ++m_row_count;
}

int
sample_buffer::prepare_insert_stmt()
{
    if(m_insert_stmt != nullptr) return SQLITE_OK;

    if(m_writer_conn == nullptr)
    {
        LOG_ERROR("buffer: writer connection not provided");
        return SQLITE_ERROR;
    }

    int rc = sqlite3_prepare_v3(m_writer_conn,
                                m_insert_sql.c_str(),
                                -1,
                                SQLITE_PREPARE_PERSISTENT,
                                &m_insert_stmt,
                                nullptr);
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
sample_buffer::flush()
{
    if(m_row_count == 0) return SQLITE_OK;

    int rc = prepare_insert_stmt();
    if(rc != SQLITE_OK) return rc;

    auto clear_state = [this]() {
        for(auto& col : m_int_cols)
        {
            col.values.clear();
            col.is_null.clear();
        }
        for(auto& col : m_text_cols)
        {
            col.values.clear();
            col.is_null.clear();
        }
        m_row_count = 0;
    };

    char* err = nullptr;
    rc        = sqlite3_exec(m_writer_conn, "BEGIN IMMEDIATE", nullptr, nullptr, &err);
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
        for(size_t c = 0; c < k_text_column_count; ++c)
        {
            const int pos = static_cast<int>(k_int_column_count + c) + 1;
            if(m_text_cols[c].is_null[r] != 0u)
            {
                sqlite3_bind_null(stmt, pos);
            }
            else
            {
                const std::string& s = m_text_cols[c].values[r];
                sqlite3_bind_text(
                    stmt, pos, s.data(), static_cast<int>(s.size()), SQLITE_STATIC);
            }
        }

        rc = sqlite3_step(stmt);
        if(rc != SQLITE_DONE)
        {
            LOG_ERROR(
                "buffer: step failed at row {}: {}", r, sqlite3_errmsg(m_writer_conn));
            sqlite3_exec(m_writer_conn, "ROLLBACK", nullptr, nullptr, nullptr);
            clear_state();
            return rc;
        }
    }

    rc = sqlite3_exec(m_writer_conn, "COMMIT", nullptr, nullptr, &err);
    if(rc != SQLITE_OK)
    {
        LOG_ERROR("buffer: COMMIT failed: {}", err != nullptr ? err : "?");
        sqlite3_free(err);
        sqlite3_exec(m_writer_conn, "ROLLBACK", nullptr, nullptr, nullptr);
        clear_state();
        return rc;
    }

    clear_state();
    return SQLITE_OK;
}

void
sample_buffer::reserve(std::size_t expected_rows)
{
    for(auto& col : m_int_cols)
    {
        col.values.reserve(expected_rows);
        col.is_null.reserve(expected_rows);
    }
    for(auto& col : m_text_cols)
    {
        col.values.reserve(expected_rows);
        col.is_null.reserve(expected_rows);
    }
}

void
sample_buffer::register_instance(const std::string& real_table_name,
                                 sample_buffer*     buffer)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[real_table_name] = buffer;
}

void
sample_buffer::unregister_instance(const std::string& real_table_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(real_table_name);
}

sample_buffer*
sample_buffer::get_active_instance(const std::string& real_table_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto                        it = registry().find(real_table_name);
    return (it != registry().end()) ? it->second : nullptr;
}

}  // namespace profiler_hub::data_storage::vtable
