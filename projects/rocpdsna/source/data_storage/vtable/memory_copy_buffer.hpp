// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Per-table buffer for the memory_copy writer. Same shape as
// kernel_dispatch_buffer; column count and null-policy are table-specific.

#pragma once

#include <sqlite3.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocpdsna::data_storage::vtable
{

// Named-field row for memory_copy_buffer::push. Replaces the old
// 17-positional-argument push() so callers cannot silently swap any of the
// addresses or agent IDs.
struct memory_copy_row
{
    int64_t                id;
    int64_t                nid;
    int64_t                pid;
    std::optional<int64_t> tid;
    int64_t                start;
    int64_t                end;
    int64_t                name_id;
    std::optional<int64_t> dst_agent_id;
    std::optional<int64_t> dst_address;
    std::optional<int64_t> src_agent_id;
    std::optional<int64_t> src_address;
    int64_t                size;
    std::optional<int64_t> queue_id;
    std::optional<int64_t> stream_id;
    std::optional<int64_t> region_name_id;
    std::optional<int64_t> event_id;
    std::string_view       extdata;
};

class memory_copy_buffer
{
public:
    static constexpr size_t k_int_column_count  = 16;
    static constexpr size_t k_text_column_index = 16;
    static constexpr size_t k_total_columns     = 17;

    // The writer_conn is owned by sqlite_backend and outlives this buffer.
    memory_copy_buffer(std::string real_table_name, sqlite3* writer_conn);
    ~memory_copy_buffer();

    memory_copy_buffer(const memory_copy_buffer&)            = delete;
    memory_copy_buffer& operator=(const memory_copy_buffer&) = delete;
    memory_copy_buffer(memory_copy_buffer&&)                 = delete;
    memory_copy_buffer& operator=(memory_copy_buffer&&)      = delete;

    // Direct push - no SQLite trampoline. Field order in memory_copy_row
    // matches the SQL column list: id, nid, pid, tid, start, end, name_id,
    // dst_agent_id, dst_address, src_agent_id, src_address, size, queue_id,
    // stream_id, region_name_id, event_id, extdata.
    void push(const memory_copy_row& row);

    void push_from_values(sqlite3_value** argv);

    int flush();

    void reserve(std::size_t expected_rows);

    [[nodiscard]] size_t row_count() const noexcept { return m_row_count; }

    [[nodiscard]] sqlite3* writer_connection() const noexcept { return m_writer_conn; }

    static void register_instance(const std::string&  real_table_name,
                                  memory_copy_buffer* buffer);
    static void unregister_instance(const std::string& real_table_name);
    [[nodiscard]] static memory_copy_buffer* get_active_instance(
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

    std::string                                  m_real_table_name;
    std::string                                  m_insert_sql;
    sqlite3*                                     m_writer_conn = nullptr;
    sqlite3_stmt*                                m_insert_stmt = nullptr;
    std::array<int_column_t, k_int_column_count> m_int_cols{};
    text_column_t                                m_text_col{};
    size_t                                       m_row_count = 0;
};

}  // namespace rocpdsna::data_storage::vtable
