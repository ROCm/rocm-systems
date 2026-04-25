// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// Bypass-the-trampoline POC: per-table buffer for the pmc_event writer.
// Schema mixes integers with one REAL (value) column and one TEXT (extdata)
// column.

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

class pmc_event_buffer
{
public:
    // Column order in SQL: id, event_id, pmc_id, value, extdata.
    // 3 ints (id, event_id, pmc_id), 1 real (value), 1 text (extdata).
    static constexpr size_t k_int_column_count  = 3;
    static constexpr size_t k_real_column_index = 3;
    static constexpr size_t k_text_column_index = 4;
    static constexpr size_t k_total_columns     = 5;

    pmc_event_buffer(std::string real_table_name, std::string db_path);
    ~pmc_event_buffer();

    pmc_event_buffer(const pmc_event_buffer&)            = delete;
    pmc_event_buffer& operator=(const pmc_event_buffer&) = delete;
    pmc_event_buffer(pmc_event_buffer&&)                 = delete;
    pmc_event_buffer& operator=(pmc_event_buffer&&)      = delete;

    void push(int64_t                id,
              std::optional<int64_t> event_id,
              int64_t                pmc_id,
              double                 value,
              std::string_view       extdata);

    void push_from_values(sqlite3_value** argv);

    int flush();

    void reserve(std::size_t expected_rows);

    [[nodiscard]] size_t row_count() const noexcept { return m_row_count; }

    static void register_instance(const std::string& real_table_name,
                                  pmc_event_buffer*  buffer);
    static void unregister_instance(const std::string& real_table_name);
    [[nodiscard]] static pmc_event_buffer* get_active_instance(
        const std::string& real_table_name);

private:
    int prepare_insert_stmt();

    struct int_column_t
    {
        std::vector<int64_t> values;
        std::vector<uint8_t> is_null;
    };

    struct real_column_t
    {
        std::vector<double>  values;
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
    real_column_t                                m_real_col{};
    text_column_t                                m_text_col{};
    size_t                                       m_row_count = 0;
};

}  // namespace rocpdsna::data_storage::vtable
