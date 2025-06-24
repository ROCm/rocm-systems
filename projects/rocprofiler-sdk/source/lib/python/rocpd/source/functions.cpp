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
#include <string_view>

namespace rocpd
{
namespace functions
{
namespace
{
// Custom SQL function: rocpd_get_string(string_id, guid)
void
rocpd_get_string(sqlite3_context* context, int argc, sqlite3_value** argv)
{
    if(argc != 4)
    {
        ROCP_WARNING << "rocpd_get_string requires exactly 2 arguments (string_id, guid)";
        sqlite3_result_null(context);
        return;
    }

    auto* db = static_cast<sqlite3*>(sqlite3_user_data(context));

    // common and unique name ids passed in
    auto        _name_id = sqlite3_value_int64(argv[0]);
    const auto* _guid    = reinterpret_cast<const char*>(sqlite3_value_text(argv[1]));

    auto execute_query = [&](std::string_view _query) {
        sqlite3_stmt* stmt = nullptr;

        if(int rc = sqlite3_prepare_v2(db, _query.data(), -1, &stmt, nullptr); rc != SQLITE_OK)
        {
            ROCP_WARNING << fmt::format("SQL prepare failed: {}", sqlite3_errmsg(db));
            sqlite3_result_error(context, "SQL prepare failed", -1);
            return;
        }

        sqlite3_bind_int64(stmt, 1, _name_id);
        sqlite3_bind_text(stmt, 1, _guid, std::string_view{_guid}.length(), nullptr);

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

    if(_name_id != 0)
    {
        execute_query("SELECT string FROM rocpd_string WHERE id == ? AND guid = '?'");
    }
    else
    {
        sqlite3_result_null(context);
    }
}

// --- 1) Define the aggregation context ---
struct stddev_context
{
    sqlite3_int64 nsamp    = 0;    // count of values
    double        mean     = 0.0;  // running mean
    double        diff_sqr = 0.0;  // running sum of squares of differences
};

// --- 2) step function: called once per row ---
void
stddev_step(sqlite3_context* ctx, int argc, sqlite3_value** argv)
{
    if(argc == 0) return;

    // We expect a single REAL or INT argument
    if(sqlite3_value_type(argv[0]) == SQLITE_NULL) return;

    auto val = sqlite3_value_double(argv[0]);

    // Allocate or fetch our context struct
    auto* p = static_cast<stddev_context*>(sqlite3_aggregate_context(ctx, sizeof(stddev_context)));
    if(!p) return;  // OOM

    // Initialize on first call
    if(p->nsamp == 0)
    {
        p->nsamp    = 0;
        p->mean     = 0.0;
        p->diff_sqr = 0.0;
    }

    // Welford’s algorithm
    ++p->nsamp;
    auto delta = (val - p->mean);
    p->mean += (delta / p->nsamp);
    auto delta2 = (val - p->mean);
    p->diff_sqr += (delta * delta2);
}

// --- 3) finalize function: called after all rows are processed ---
void
stddev_finalize(sqlite3_context* ctx)
{
    auto* p = static_cast<stddev_context*>(sqlite3_aggregate_context(ctx, 0));
    if(!p || p->nsamp < 2)
    {
        // Not enough data to form a sample stddev
        sqlite3_result_null(ctx);
    }
    else
    {
        auto variance = p->diff_sqr / (p->nsamp - 1);
        sqlite3_result_double(ctx, std::sqrt(variance));
    }
}
}  // namespace

void
define_for_database(sqlite3* conn)
{
    // name = "STDDEV_SAMP", 1 arg, UTF-8, no user data,
    // no scalar function, but these aggregate callbacks:
    sqlite3_create_function_v2(conn,
                               "STDDEV_SAMP",  // SQL name
                               1,              // number of args
                               SQLITE_UTF8,
                               nullptr,          // user data pointer
                               nullptr,          // xFunc (for scalar) — null for aggregates
                               stddev_step,      // xStep
                               stddev_finalize,  // xFinal
                               nullptr           // destructor for user data
    );

    sqlite3_create_function_v2(conn,
                               "rocpd_get_string",
                               2,
                               SQLITE_UTF8,
                               conn,
                               rocpd_get_string,
                               nullptr,
                               nullptr,
                               nullptr);
}
}  // namespace functions
}  // namespace rocpd
