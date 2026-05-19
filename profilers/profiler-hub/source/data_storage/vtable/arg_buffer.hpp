// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Per-table buffer for the arg writer.

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

// Named-field row for arg_buffer::push. Field order matches the SQL
// column list: id, event_id, position, type, name, value, extdata.
struct arg_row
{
    int64_t                         id;
    int64_t                         event_id;
    int64_t                         position;
    std::string_view                type;
    std::string_view                name;
    std::optional<std::string_view> value;
    std::string_view                extdata;
};

class arg_buffer
{
public:
    static constexpr size_t k_int_column_count  = 3;
    static constexpr size_t k_text_column_count = 4;
    static constexpr size_t k_total_columns     = 7;

    // The writer_conn is owned by sqlite_backend and outlives this buffer.
    arg_buffer(std::string real_table_name, sqlite3* writer_conn);
    ~arg_buffer();

    arg_buffer(const arg_buffer&)            = delete;
    arg_buffer& operator=(const arg_buffer&) = delete;
    arg_buffer(arg_buffer&&)                 = delete;
    arg_buffer& operator=(arg_buffer&&)      = delete;

    void push(const arg_row& row);

    void push_from_values(sqlite3_value** argv);

    int flush();

    void reserve(std::size_t expected_rows);

    [[nodiscard]] size_t row_count() const noexcept { return m_row_count; }

    [[nodiscard]] sqlite3* writer_connection() const noexcept { return m_writer_conn; }

    static void register_instance(const std::string& real_table_name, arg_buffer* buffer);
    static void unregister_instance(const std::string& real_table_name);
    [[nodiscard]] static arg_buffer* get_active_instance(
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
