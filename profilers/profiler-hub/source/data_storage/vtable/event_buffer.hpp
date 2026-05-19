// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Per-table buffer for the event writer.

#pragma once

#include <sqlite3.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace profiler_hub::data_storage::vtable
{

// Named-field row for event_buffer::push. Field order matches the SQL
// column list: id, category_id, stack_id, parent_stack_id, correlation_id,
// call_stack, line_info, extdata.
//
// call_stack and line_info are owned strings: the call-site serializes them
// from structured data, so there is no stable string to view into.
// extdata is a view into a string owned by the caller.
struct event_row
{
    int64_t                id;
    std::optional<int64_t> category_id;
    std::optional<int64_t> stack_id;
    std::optional<int64_t> parent_stack_id;
    std::optional<int64_t> correlation_id;
    std::string            call_stack;
    std::string            line_info;
    std::string_view       extdata;
};

class event_buffer
{
public:
    static constexpr size_t k_int_column_count  = 5;
    static constexpr size_t k_text_column_count = 3;
    static constexpr size_t k_total_columns     = 8;

    // The writer_conn is owned by sqlite_backend and outlives this buffer.
    event_buffer(std::string real_table_name, sqlite3* writer_conn);
    ~event_buffer();

    event_buffer(const event_buffer&)            = delete;
    event_buffer& operator=(const event_buffer&) = delete;
    event_buffer(event_buffer&&)                 = delete;
    event_buffer& operator=(event_buffer&&)      = delete;

    // Push a row and return the locally-assigned id (same as row.id).
    // Takes ownership so call_stack/line_info strings are moved into the buffer.
    int64_t push(event_row row);

    // Remove the last pushed row from the buffer. Used to undo a push when
    // a subsequent operation fails and the in-flight write must not persist.
    // Must only be called if row_count() > 0 and exactly one push has occurred
    // since the most recent flush (or construction).
    void pop_last_row() noexcept;

    void push_from_values(sqlite3_value** argv);

    int flush();

    void reserve(std::size_t expected_rows);

    [[nodiscard]] size_t row_count() const noexcept { return m_row_count; }

    [[nodiscard]] sqlite3* writer_connection() const noexcept { return m_writer_conn; }

    static void register_instance(const std::string& real_table_name,
                                  event_buffer*      buffer);
    static void unregister_instance(const std::string& real_table_name);
    [[nodiscard]] static event_buffer* get_active_instance(
        const std::string& real_table_name);

private:
    int prepare_insert_stmt();

    struct int_column_t
    {
        std::vector<int64_t> values;
        std::vector<uint8_t> is_null;
    };

    struct text_column_t
    {
        std::vector<std::string> values;
        std::vector<uint8_t>     is_null;
    };

    std::string                                    m_real_table_name;
    std::string                                    m_insert_sql;
    sqlite3*                                       m_writer_conn = nullptr;
    sqlite3_stmt*                                  m_insert_stmt = nullptr;
    std::array<int_column_t, k_int_column_count>   m_int_cols{};
    std::array<text_column_t, k_text_column_count> m_text_cols{};
    size_t                                         m_row_count = 0;
};

}  // namespace profiler_hub::data_storage::vtable
