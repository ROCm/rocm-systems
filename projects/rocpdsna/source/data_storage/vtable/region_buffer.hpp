// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Bypass-the-trampoline POC: per-table buffer for the region writer.

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

// Named-field row for region_buffer::push. Field order matches the SQL
// column list: id, nid, pid, tid, start, end, name_id, event_id, extdata.
struct region_row
{
    int64_t                id;
    int64_t                nid;
    int64_t                pid;
    int64_t                tid;
    int64_t                start;
    int64_t                end;
    int64_t                name_id;
    std::optional<int64_t> event_id;
    std::string_view       extdata;
};

class region_buffer
{
public:
    static constexpr size_t k_int_column_count  = 8;
    static constexpr size_t k_text_column_index = 8;
    static constexpr size_t k_total_columns     = 9;

    region_buffer(std::string real_table_name, std::string db_path);
    ~region_buffer();

    region_buffer(const region_buffer&)            = delete;
    region_buffer& operator=(const region_buffer&) = delete;
    region_buffer(region_buffer&&)                 = delete;
    region_buffer& operator=(region_buffer&&)      = delete;

    void push(const region_row& row);

    void push_from_values(sqlite3_value** argv);

    int flush();

    void reserve(std::size_t expected_rows);

    [[nodiscard]] size_t row_count() const noexcept { return m_row_count; }

    static void register_instance(const std::string& real_table_name,
                                  region_buffer*     buffer);
    static void unregister_instance(const std::string& real_table_name);
    [[nodiscard]] static region_buffer* get_active_instance(
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
    std::string                                  m_db_path;
    std::string                                  m_insert_sql;
    sqlite3*                                     m_writer_conn = nullptr;
    sqlite3_stmt*                                m_insert_stmt = nullptr;
    std::array<int_column_t, k_int_column_count> m_int_cols{};
    text_column_t                                m_text_col{};
    size_t                                       m_row_count = 0;
};

}  // namespace rocpdsna::data_storage::vtable
