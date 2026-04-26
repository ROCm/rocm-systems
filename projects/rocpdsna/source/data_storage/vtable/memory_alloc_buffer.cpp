// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "memory_alloc_buffer.hpp"

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

constexpr std::array<const char*, memory_alloc_buffer::k_total_columns> k_column_names = {
    "id",  "nid",     "pid",  "tid",      "agent_id",  "type",     "level",  "start",
    "end", "address", "size", "queue_id", "stream_id", "event_id", "extdata"
};

// kind_t encodes per-SQL-column whether to bind from an int column or a text
// column, plus the index into the corresponding per-column array.
enum class column_kind_t : uint8_t
{
    integer,
    text
};

struct bind_slot_t
{
    column_kind_t kind;
    uint8_t       index;  // index into m_int_cols or m_text_cols
};

// Maps SQL column position -> (kind, per-kind index).
constexpr std::array<bind_slot_t, memory_alloc_buffer::k_total_columns> k_plan = { {
    { column_kind_t::integer, 0 },   // id
    { column_kind_t::integer, 1 },   // nid
    { column_kind_t::integer, 2 },   // pid
    { column_kind_t::integer, 3 },   // tid
    { column_kind_t::integer, 4 },   // agent_id
    { column_kind_t::text, 0 },      // type
    { column_kind_t::text, 1 },      // level
    { column_kind_t::integer, 5 },   // start
    { column_kind_t::integer, 6 },   // end
    { column_kind_t::integer, 7 },   // address
    { column_kind_t::integer, 8 },   // size
    { column_kind_t::integer, 9 },   // queue_id
    { column_kind_t::integer, 10 },  // stream_id
    { column_kind_t::integer, 11 },  // event_id
    { column_kind_t::text, 2 },      // extdata
} };

std::mutex&
registry_mutex()
{
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, memory_alloc_buffer*>&
registry()
{
    static std::unordered_map<std::string, memory_alloc_buffer*> r;
    return r;
}

}  // namespace

memory_alloc_buffer::memory_alloc_buffer(std::string real_table_name,
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

memory_alloc_buffer::~memory_alloc_buffer()
{
    flush();
    if(m_insert_stmt != nullptr) sqlite3_finalize(m_insert_stmt);
    // m_writer_conn is owned by sqlite_backend; do not close it here.
}

void
memory_alloc_buffer::push(const memory_alloc_row& row)
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

    auto push_opt_text_at = [&](size_t col_idx, std::optional<std::string_view> v) {
        if(v.has_value())
        {
            m_text_cols[col_idx].values.emplace_back(*v);
            m_text_cols[col_idx].is_null.push_back(0);
        }
        else
        {
            m_text_cols[col_idx].values.emplace_back();
            m_text_cols[col_idx].is_null.push_back(1);
        }
    };

    push_int_at(0, row.id);
    push_int_at(1, row.nid);
    push_int_at(2, row.pid);
    push_opt_at(3, row.tid);
    push_opt_at(4, row.agent_id);
    push_opt_text_at(0, row.type);
    push_opt_text_at(1, row.level);
    push_int_at(5, row.start);
    push_int_at(6, row.end);
    push_opt_at(7, row.address);
    push_int_at(8, row.size);
    push_opt_at(9, row.queue_id);
    push_opt_at(10, row.stream_id);
    push_opt_at(11, row.event_id);

    m_text_cols[2].values.emplace_back(row.extdata);
    m_text_cols[2].is_null.push_back(0);

    ++m_row_count;
}

void
memory_alloc_buffer::push_from_values(sqlite3_value** argv)
{
    for(size_t c = 0; c < k_total_columns; ++c)
    {
        sqlite3_value* v    = argv[2 + static_cast<int>(c)];
        const auto&    slot = k_plan[c];
        if(slot.kind == column_kind_t::integer)
        {
            auto& col = m_int_cols[slot.index];
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
        else
        {
            auto& col = m_text_cols[slot.index];
            if(sqlite3_value_type(v) == SQLITE_NULL)
            {
                col.values.emplace_back();
                col.is_null.push_back(1);
            }
            else
            {
                const auto* txt = reinterpret_cast<const char*>(sqlite3_value_text(v));
                const auto  n   = static_cast<size_t>(sqlite3_value_bytes(v));
                col.values.emplace_back(txt != nullptr ? std::string(txt, n)
                                                       : std::string{});
                col.is_null.push_back(0);
            }
        }
    }
    ++m_row_count;
}

int
memory_alloc_buffer::prepare_insert_stmt()
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
memory_alloc_buffer::flush()
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

        for(size_t c = 0; c < k_total_columns; ++c)
        {
            const int   pos  = static_cast<int>(c) + 1;
            const auto& slot = k_plan[c];
            if(slot.kind == column_kind_t::integer)
            {
                const auto& col = m_int_cols[slot.index];
                if(col.is_null[r] != 0u)
                {
                    sqlite3_bind_null(stmt, pos);
                }
                else
                {
                    sqlite3_bind_int64(stmt, pos, col.values[r]);
                }
            }
            else
            {
                const auto& col = m_text_cols[slot.index];
                if(col.is_null[r] != 0u)
                {
                    sqlite3_bind_null(stmt, pos);
                }
                else
                {
                    const std::string& s = col.values[r];
                    sqlite3_bind_text(
                        stmt, pos, s.data(), static_cast<int>(s.size()), SQLITE_STATIC);
                }
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
memory_alloc_buffer::reserve(std::size_t expected_rows)
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
memory_alloc_buffer::register_instance(const std::string&   real_table_name,
                                       memory_alloc_buffer* buffer)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry()[real_table_name] = buffer;
}

void
memory_alloc_buffer::unregister_instance(const std::string& real_table_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    registry().erase(real_table_name);
}

memory_alloc_buffer*
memory_alloc_buffer::get_active_instance(const std::string& real_table_name)
{
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto                        it = registry().find(real_table_name);
    return (it != registry().end()) ? it->second : nullptr;
}

}  // namespace rocpdsna::data_storage::vtable
