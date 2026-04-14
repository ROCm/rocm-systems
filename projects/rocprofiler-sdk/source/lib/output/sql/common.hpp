// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "lib/output/sql/extract_data_type.hpp"

#include "lib/common/container/ring_buffer.hpp"
#include "lib/common/mpl.hpp"
#include "lib/common/units.hpp"

#include <fmt/format.h>
#include <sqlite3.h>

#include <chrono>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace rocprofiler
{
namespace tool
{
namespace sql
{
using exec_callback_t = int (*)(void* user_data, int ncols, char** coltext, char** colnames);

template <typename Tp>
using ring_buffer_t = rocprofiler::common::container::ring_buffer<Tp>;

int
exec_callback(void* user_data, int ncols, char** coltext, char** colnames);

void
check(std::string_view function, int status, std::string_view stmt);

int
busy_handler(void* data, int count);

int64_t
execute_raw_sql_statements_impl(sqlite3* conn, std::string_view stmts, int line);

int64_t
execute_raw_sql_statements_impl(sqlite3*         conn,
                                std::string_view stmts,
                                exec_callback_t  callback,
                                void*            data,
                                int              line);

std::string
extract_column_name(sqlite3_stmt* stmt, int32_t col);

int64_t
extract_row_count(sqlite3* conn, std::string_view query);

std::string_view
get_sql_data_type(int val);

template <typename Tp>
auto
extract_column(sqlite3_stmt* stmt, int32_t col);

template <typename Tp, int ExpectedV>
bool
column_data_is_null(sqlite3_stmt* stmt, int32_t col);

//
//
//  Template function definitions
//
//
template <typename Tp>
auto
extract_column(sqlite3_stmt* stmt, int32_t col)
{
    return extract_data_type<Tp>{}(stmt, col);
}

template <typename Tp, int ExpectedV>
bool
column_data_is_null(sqlite3_stmt* stmt, int32_t col)
{
    auto coltype = sqlite3_column_type(stmt, col);
    if(coltype == SQLITE_NULL) return true;

    std::string column_name = extract_column_name(stmt, col);
    const char* sql_text    = sqlite3_sql(stmt);

    ROCP_CI_LOG_IF(WARNING, coltype != ExpectedV) << fmt::format(
        "Data in SQL column {} ('{}') is neither NULL nor the expected data type ({} == {}). "
        "Column data type is: ({} == {}), SQL: {}",
        col,
        column_name,
        ExpectedV,
        get_sql_data_type(ExpectedV),
        coltype,
        get_sql_data_type(coltype),
        sql_text);

    return false;
}
}  // namespace sql
}  // namespace tool
}  // namespace rocprofiler

#define execute_raw_sql_statements(...)                                                            \
    ::rocprofiler::tool::sql::execute_raw_sql_statements_impl(__VA_ARGS__, __LINE__)

#define SQLITE3_CHECK(RESULT)                                                                      \
    ::rocprofiler::tool::sql::check(__FUNCTION__, (RESULT), std::string_view{#RESULT})
