// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "kernel_dispatch_buffer.hpp"

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

constexpr std::array<const char*, kernel_dispatch_buffer::k_total_columns>
    k_column_names = { "id",
                       "nid",
                       "pid",
                       "tid",
                       "agent_id",
                       "kernel_id",
                       "dispatch_id",
                       "queue_id",
                       "stream_id",
                       "start",
                       "end",
                       "private_segment_size",
                       "group_segment_size",
                       "workgroup_size_x",
                       "workgroup_size_y",
                       "workgroup_size_z",
                       "grid_size_x",
                       "grid_size_y",
                       "grid_size_z",
                       "region_name_id",
                       "event_id",
                       "extdata" };

std::mutex&
registry_mutex()
{
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, kernel_dispatch_buffer*>&
registry()
{
    static std::unordered_map<std::string, kernel_dispatch_buffer*> r;
    return r;
}

}  // namespace

kernel_dispatch_buffer::kernel_dispatch_buffer(std::string real_table_name,
                                               sqlite3*    writer_conn)
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

kernel_dispatch_buffer::~kernel_dispatch_buffer()
{
    flush();
    if(m_insert_stmt != nullptr) sqlite3_finalize(m_insert_stmt);
    // m_writer_conn is owned by sqlite_backend; do not close it here.
}

void
kernel_dispatch_buffer::push(const kernel_dispatch_row& row)
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

    push_int_at(0, row.id);
    push_int_at(1, row.nid);
    push_int_at(2, row.pid);
    push_opt_at(3, row.tid);
    push_int_at(4, row.agent_id);
    push_int_at(5, row.kernel_id);
    push_int_at(6, row.dispatch_id);
    push_int_at(7, row.queue_id);
    push_int_at(8, row.stream_id);
    push_int_at(9, row.start);
    push_int_at(10, row.end);
    push_opt_at(11, row.private_segment_size);
    push_opt_at(12, row.group_segment_size);
    push_int_at(13, row.workgroup_size_x);
    push_int_at(14, row.workgroup_size_y);
    push_int_at(15, row.workgroup_size_z);
    push_int_at(16, row.grid_size_x);
    push_int_at(17, row.grid_size_y);
    push_int_at(18, row.grid_size_z);
    push_opt_at(19, row.region_name_id);
    push_opt_at(20, row.event_id);

    m_text_col.values.emplace_back(row.extdata);
    m_text_col.is_null.push_back(0);

    ++m_row_count;
}

void
kernel_dispatch_buffer::push_from_values(sqlite3_value** argv)
{
    // argv layout matches xUpdate for INSERT: argv[0]=NULL rowid, argv[1]=user
    // rowid, argv[2..] are the column values in declared order.
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
kernel_dispatch_buffer::prepare_insert_stmt()
{
    if(m_insert_stmt != nullptr) return SQLITE_OK;

    if(m_writer_conn == nullptr)
    {
        LOG_ERROR("buffer: writer connection not provided");
        return SQLITE_ERROR;
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
kernel_dispatch_buffer::flush()
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
kernel_dispatch_buffer::reserve(std::size_t expected_rows)
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
kernel_dispatch_buffer::register_instance(const std::string&      real_table_name,
                                          kernel_dispatch_buffer* buffer)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[real_table_name] = buffer;
}

void
kernel_dispatch_buffer::unregister_instance(const std::string& real_table_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(real_table_name);
}

kernel_dispatch_buffer*
kernel_dispatch_buffer::get_active_instance(const std::string& real_table_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto                        it = registry().find(real_table_name);
    return (it != registry().end()) ? it->second : nullptr;
}

}  // namespace rocpdsna::data_storage::vtable
