// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Bypass-the-trampoline POC: kernel_dispatch_buffer holds the per-column
// vectors that previously lived inside the SQLite virtual table xUpdate
// path. Writers can push rows directly via push() without going through
// sqlite3_step / xUpdate / sqlite3_value unpacking.
//
// The vtable wrapper still exists for SELECT compatibility and for the
// flush trigger reachable via storage_t. Its xUpdate path is now
// vestigial: the writer no longer drives it.

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

class kernel_dispatch_buffer
{
public:
    static constexpr size_t k_int_column_count  = 21;
    static constexpr size_t k_text_column_index = 21;
    static constexpr size_t k_total_columns     = 22;

    kernel_dispatch_buffer(std::string real_table_name, std::string db_path);
    ~kernel_dispatch_buffer();

    kernel_dispatch_buffer(const kernel_dispatch_buffer&)            = delete;
    kernel_dispatch_buffer& operator=(const kernel_dispatch_buffer&) = delete;
    kernel_dispatch_buffer(kernel_dispatch_buffer&&)                 = delete;
    kernel_dispatch_buffer& operator=(kernel_dispatch_buffer&&)      = delete;

    // Direct push - no SQLite trampoline.
    // Argument order matches the kernel_dispatch column list:
    //   id, nid, pid, tid, agent_id, kernel_id, dispatch_id, queue_id,
    //   stream_id, start, end, private_segment_size, group_segment_size,
    //   workgroup_size_x, workgroup_size_y, workgroup_size_z,
    //   grid_size_x, grid_size_y, grid_size_z, region_name_id, event_id,
    //   extdata.
    void push(int64_t                id,
              int64_t                nid,
              int64_t                pid,
              std::optional<int64_t> tid,
              int64_t                agent_id,
              int64_t                kernel_id,
              int64_t                dispatch_id,
              int64_t                queue_id,
              int64_t                stream_id,
              int64_t                start,
              int64_t                end,
              std::optional<int64_t> private_segment_size,
              std::optional<int64_t> group_segment_size,
              int64_t                workgroup_size_x,
              int64_t                workgroup_size_y,
              int64_t                workgroup_size_z,
              int64_t                grid_size_x,
              int64_t                grid_size_y,
              int64_t                grid_size_z,
              std::optional<int64_t> region_name_id,
              std::optional<int64_t> event_id,
              std::string_view       extdata);

    // Existing vtable trampoline path (kept for completeness, used by xUpdate).
    void push_from_values(sqlite3_value** argv);

    // Bulk-write the buffered rows to the real table.
    int flush();

    // Pre-size all per-column vectors to avoid reallocation thrashing on the
    // hot insert path. Bench-driven hint; safe to call before any push().
    void reserve(std::size_t expected_rows);

    [[nodiscard]] size_t row_count() const noexcept { return m_row_count; }

    // Static registry: lets the writer reach the active buffer instance for
    // a given real table without plumbing through storage_t.
    static void register_instance(const std::string&      real_table_name,
                                  kernel_dispatch_buffer* buffer);
    static void unregister_instance(const std::string& real_table_name);
    [[nodiscard]] static kernel_dispatch_buffer* get_active_instance(
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
