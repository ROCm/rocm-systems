// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Bypass-the-trampoline POC: per-table buffer for the memory_alloc writer.
// Schema mixes optional text columns (type, level, extdata) with integers,
// so the buffer carries an array of text columns indexed by position.

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

// Named-field row for memory_alloc_buffer::push. Field order matches the
// SQL column list: id, nid, pid, tid, agent_id, type, level, start, end,
// address, size, queue_id, stream_id, event_id, extdata.
struct memory_alloc_row
{
    int64_t                         id;
    int64_t                         nid;
    int64_t                         pid;
    std::optional<int64_t>          tid;
    std::optional<int64_t>          agent_id;
    std::optional<std::string_view> type;
    std::optional<std::string_view> level;
    int64_t                         start;
    int64_t                         end;
    std::optional<int64_t>          address;
    int64_t                         size;
    std::optional<int64_t>          queue_id;
    std::optional<int64_t>          stream_id;
    std::optional<int64_t>          event_id;
    std::string_view                extdata;
};

class memory_alloc_buffer
{
public:
    // Layout: 11 int columns interleaved with 3 text columns.
    // int  positions: 0=id, 1=nid, 2=pid, 3=tid, 4=agent_id,
    //                 5=start, 6=end, 7=address, 8=size,
    //                 9=queue_id, 10=stream_id, 11=event_id (12 total ints)
    // text positions: 0=type, 1=level, 2=extdata
    static constexpr size_t k_int_column_count  = 12;
    static constexpr size_t k_text_column_count = 3;
    // Column order in the SQL statement (0=int, 1=text):
    //   id, nid, pid, tid, agent_id, type, level, start, end, address,
    //   size, queue_id, stream_id, event_id, extdata
    static constexpr size_t k_total_columns = 15;

    // The writer_conn is owned by sqlite_backend and outlives this buffer.
    memory_alloc_buffer(std::string real_table_name, sqlite3* writer_conn);
    ~memory_alloc_buffer();

    memory_alloc_buffer(const memory_alloc_buffer&)            = delete;
    memory_alloc_buffer& operator=(const memory_alloc_buffer&) = delete;
    memory_alloc_buffer(memory_alloc_buffer&&)                 = delete;
    memory_alloc_buffer& operator=(memory_alloc_buffer&&)      = delete;

    void push(const memory_alloc_row& row);

    void push_from_values(sqlite3_value** argv);

    int flush();

    void reserve(std::size_t expected_rows);

    [[nodiscard]] size_t row_count() const noexcept { return m_row_count; }

    [[nodiscard]] sqlite3* writer_connection() const noexcept { return m_writer_conn; }

    static void register_instance(const std::string&   real_table_name,
                                  memory_alloc_buffer* buffer);
    static void unregister_instance(const std::string& real_table_name);
    [[nodiscard]] static memory_alloc_buffer* get_active_instance(
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

}  // namespace rocpdsna::data_storage::vtable
