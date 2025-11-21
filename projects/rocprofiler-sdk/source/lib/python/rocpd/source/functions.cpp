// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "lib/python/rocpd/source/functions.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/serialization.hpp>

#include <fmt/format.h>
#include <sqlite3.h>
#include <cereal/cereal.hpp>

#include <cstdint>

namespace rocpd
{
namespace functions
{
namespace
{
// Custom SQL function: rocpd_get_string(common_string_id, unique_string_id, nid, pid)
void
rocpd_get_string(sqlite3_context* context, int argc, sqlite3_value** argv)
{
    if(argc != 4)
    {
        ROCP_WARNING << "rocpd_get_string requires exactly 4 arguments (common_string_id, "
                        "unique_string_id, nid, pid)";
        sqlite3_result_null(context);
        return;
    }

    auto* db = static_cast<sqlite3*>(sqlite3_user_data(context));

    // common and unique name ids passed in
    auto c_name_id = sqlite3_value_int64(argv[0]);
    auto u_name_id = sqlite3_value_int64(argv[1]);

    auto execute_query = [&](std::string_view _query, std::initializer_list<int64_t>&& _args) {
        // char query[256];
        // snprintf(query, sizeof(query), "SELECT value FROM %s WHERE id = ?", table);

        sqlite3_stmt* stmt = nullptr;

        if(int rc = sqlite3_prepare_v2(db, _query.data(), -1, &stmt, nullptr); rc != SQLITE_OK)
        {
            ROCP_WARNING << fmt::format("SQL prepare failed: {}", sqlite3_errmsg(db));
            sqlite3_result_error(context, "SQL prepare failed", -1);
            return;
        }

        int64_t idx = 1;
        for(auto itr : _args)
            sqlite3_bind_int64(stmt, idx++, itr);

        if(auto rc = sqlite3_step(stmt); rc == SQLITE_ROW)
        {
            const unsigned char* result = sqlite3_column_text(stmt, 0);
            sqlite3_result_text(
                context, reinterpret_cast<const char*>(result), -1, SQLITE_TRANSIENT);
        }
        else if(rc == SQLITE_DONE)
        {
            ROCP_WARNING << fmt::format("No row found for query '{}'", _query);
            sqlite3_result_null(context);
        }
        else
        {
            ROCP_WARNING << fmt::format("SQL step failed: {}", sqlite3_errmsg(db));
            sqlite3_result_error(context, "SQL step failed", -1);
        }

        sqlite3_finalize(stmt);
    };

    if(c_name_id != 0)
    {
        execute_query("SELECT string FROM rocpd_common_string WHERE id == ?",
                      std::initializer_list<int64_t>{c_name_id});
    }
    else if(u_name_id != 0)
    {
        auto u_nid = sqlite3_value_int64(argv[2]);
        auto u_pid = sqlite3_value_int64(argv[3]);

        execute_query(
            "SELECT string FROM rocpd_unique_string WHERE id == ? AND nid = ? AND pid = ?",
            std::initializer_list<int64_t>{u_name_id, u_nid, u_pid});
    }
    else
    {
        sqlite3_result_null(context);
    }
}
}  // namespace

void
define_for_database(sqlite3* conn)
{
    if(false)
    {
        sqlite3_create_function_v2(conn,
                                   "rocpd_get_string",
                                   4,
                                   SQLITE_UTF8,
                                   conn,
                                   rocpd_get_string,
                                   nullptr,
                                   nullptr,
                                   nullptr);
    }

    rocprofiler::common::consume_args(conn);
}
}  // namespace functions
}  // namespace rocpd
