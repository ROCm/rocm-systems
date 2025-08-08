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

#include "lib/output/sql/common.hpp"
#include "lib/output/kernel_symbol_info.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"

#include <rocprofiler-sdk/cxx/details/tokenize.hpp>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/format.h>
#include <sqlite3.h>

#include <iomanip>
#include <sstream>
#include <thread>

namespace rocprofiler
{
namespace tool
{
namespace sql
{
namespace sdk = ::rocprofiler::sdk;

void
check(std::string_view function, int status, std::string_view stmt)
{
    if(status != SQLITE_OK)
    {
        ROCP_FATAL << "[" << function << "] " << stmt << " failed with error code " << status;
    }
}

int
busy_handler(void* /*data*/, int count)
{
    count = (count < 9) ? count : 8;
    std::this_thread::sleep_for(std::chrono::microseconds{1000 * (0x1 << count)});
    return 1;
}

// invoked during SELECT operation; unused but kept for reference
int
exec_callback(void* user_data, int ncols, char** coltext, char** colnames)
{
    ROCP_INFO << "SQL callback invoked with " << ncols << " columns";

    if(!coltext || !colnames) return SQLITE_OK;

    auto header  = std::stringstream{};
    auto div     = std::stringstream{};
    auto content = std::stringstream{};

    header << fmt::format(
        "| {} |", fmt::join(std::vector<std::string_view>(colnames, colnames + ncols), " | "));
    div << fmt::format("|-{}-|", std::string(header.str().size() - 4, '-'));
    content << fmt::format(
        "| {} |", fmt::join(std::vector<std::string_view>(coltext, coltext + ncols), " | "));

    ROCP_WARNING << "SQL callback for " << ncols << " columns contents: "
                 << "\n\t" << header.str() << "\n\t" << div.str() << "\n\t" << content.str();

    return SQLITE_OK;

    (void) user_data;
}

int64_t
execute_raw_sql_statements_impl(sqlite3*         conn,
                                std::string_view stmts,
                                exec_callback_t  callback,
                                void*            data,
                                int              line)
{
    int64_t row_id = -1;
    // NOLINTNEXTLINE(performance-for-range-copy)
    for(auto stmt : sdk::parse::tokenize(stmts, ";"))
    {
        stmt = fmt::format("{};", std::move(stmt));  // make sure ends with semi-colon

        ROCP_TRACE << "Executing SQLite3 statement: " << stmt;

        char* msg = nullptr;
        auto  ret = sqlite3_exec(conn, stmt.c_str(), callback, data, &msg);
        if(ret != SQLITE_OK)
        {
            // ensure no memory leak
            auto dtor = common::scope_destructor{[msg]() {
                if(msg) sqlite3_free(msg);
            }};

            static constexpr auto full_file = std::string_view{__FILE__};
            static constexpr auto base_file = full_file.substr(full_file.find_last_of('/') + 1);
            ROCP_FATAL << "SQLite3 error " << ret << ": " << ((msg) ? msg : "unknown error")
                       << "\n\tSQLite3 error: " << base_file << ":" << line
                       << "\n\tSQL Statement: " << stmt;
        }
        else
        {
            if(stmt.find("INSERT") != std::string::npos) row_id = sqlite3_last_insert_rowid(conn);
        }
    }
    return row_id;
}

int64_t
execute_raw_sql_statements_impl(sqlite3* conn, std::string_view stmts, int line)
{
    return execute_raw_sql_statements_impl(conn, stmts, exec_callback, nullptr, line);
}

std::string
extract_column_name(sqlite3_stmt* stmt, int32_t col)
{
    return std::string{sqlite3_column_name(stmt, col)};
}

int64_t
extract_row_count(sqlite3* conn, std::string_view query)
{
    auto _pos         = query.find(';');
    auto _query       = (_pos == std::string_view::npos) ? query : query.substr(0, _pos);
    auto _count_query = fmt::format("SELECT COUNT(*) AS count FROM ({}) x;", _query);

    sqlite3_stmt* _stmt = nullptr;
    if(sqlite3_prepare_v2(conn, _count_query.c_str(), -1, &_stmt, nullptr) != SQLITE_OK)
    {
        ROCP_CI_LOG(ERROR) << fmt::format(
            "Error preparing select statement for '{}': {}", _count_query, sqlite3_errmsg(conn));
        return 0;
    }

    ROCP_CI_LOG_IF(ERROR, _stmt == nullptr) << "Error preparing statment: " << query;

    int64_t nrows = 0;
    if(_stmt && sqlite3_column_count(_stmt) == 1 && sqlite3_step(_stmt) == SQLITE_ROW)
        nrows = extract_column<int64_t>(_stmt, 0).value_or(0);

    sqlite3_finalize(_stmt);  // Finalize statement

    ROCP_INFO << fmt::format("SQL query '{}' contains {} rows", query, nrows);
    return nrows;
}

namespace
{
template <int Idx>
struct sql_data_type_info;

#define SPECIALIZE_SQL_DATA_TYPE_INFO(VALUE)                                                       \
    template <>                                                                                    \
    struct sql_data_type_info<VALUE>                                                               \
    {                                                                                              \
        static constexpr auto value = VALUE;                                                       \
        static constexpr auto name  = #VALUE;                                                      \
    };

SPECIALIZE_SQL_DATA_TYPE_INFO(SQLITE_INTEGER)
SPECIALIZE_SQL_DATA_TYPE_INFO(SQLITE_FLOAT)
SPECIALIZE_SQL_DATA_TYPE_INFO(SQLITE_BLOB)
SPECIALIZE_SQL_DATA_TYPE_INFO(SQLITE_NULL)
SPECIALIZE_SQL_DATA_TYPE_INFO(SQLITE_TEXT)

template <size_t Idx, size_t... Tail>
std::string_view
get_sql_data_type(int val, std::index_sequence<Idx, Tail...>)
{
    using info_type = sql_data_type_info<Idx + 1>;
    if(val == info_type::value) return std::string_view{info_type::name};
    if constexpr(sizeof...(Tail) > 0) return get_sql_data_type(val, std::index_sequence<Tail...>{});
    return std::string_view{"SQLITE_UNKNOWN"};
}
}  // namespace

std::string_view
get_sql_data_type(int val)
{
    return get_sql_data_type(val, std::make_index_sequence<SQLITE_NULL>{});
}
}  // namespace sql
}  // namespace tool
}  // namespace rocprofiler
