// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "common/traits.hpp"
#include <memory>
#include <mutex>
#include <sqlite3.h>
#include <sstream>
#include <stdexcept>

namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
static std::mutex _mutex;
class database
{
public:
    static database& get_instance();

    database(database&)            = delete;
    database& operator=(database&) = delete;

    void flush();

    ~database();

private:
    database();

    template <typename... Args>
    inline void validate_sqlite3_result(int sqlite3_error_code, const char* query,
                                        Args&&... args)
    {
        std::stringstream ss;
        ss << "\n===========================================================\n";
        ss << "Database Error\n";
        ((ss << args << " "), ...);
        ss << "\nQuery: " << query << "\n";
        switch(sqlite3_error_code)
        {
            case SQLITE_OK:
            case SQLITE_DONE: return;
            case SQLITE_CONSTRAINT:
            {
                sqlite3_stmt* stmt;

                ss << "Constraint violation(s): " << "\n";

                sqlite3_exec(_sqlite3_db_temp, "PRAGMA foreign_keys = OFF;", nullptr,
                             nullptr, nullptr);
                sqlite3_exec(_sqlite3_db_temp, query, nullptr, nullptr, nullptr);
                sqlite3_exec(_sqlite3_db_temp, "PRAGMA foreign_keys = ON;", nullptr,
                             nullptr, nullptr);
                sqlite3_prepare_v2(_sqlite3_db_temp, "PRAGMA foreign_key_check", -1,
                                   &stmt, nullptr);
                int rc = 0;
                while((rc = sqlite3_step(stmt)) == SQLITE_ROW)
                {
                    const char* table  = (const char*) sqlite3_column_text(stmt, 0);
                    int         rowid  = sqlite3_column_int(stmt, 1);
                    const char* parent = (const char*) sqlite3_column_text(stmt, 2);
                    int         fkid   = sqlite3_column_int(stmt, 3);

                    ss << "  - " << "FK Violation - Table: " << (table ? table : "NULL")
                       << ", RowID: " << rowid
                       << ", Parent: " << (parent ? parent : "NULL") << ", FKID: " << fkid
                       << "\n";
                }

                sqlite3_finalize(stmt);
            }
            break;
            default:
            {
            }
            break;
        }
        ss << " [Sqlite3 error: " << sqlite3_errstr(sqlite3_error_code);
        ss << " (Extended error message: " << sqlite3_errmsg(_sqlite3_db_temp) << ")]";
        throw std::runtime_error(ss.str());
    }

    template <typename T, std::enable_if_t<!(common::traits::is_string_literal<T>() ||
                                             std::is_floating_point_v<std::decay_t<T>> ||
                                             std::is_same_v<std::decay_t<T>, int64_t> ||
                                             std::is_same_v<std::decay_t<T>, uint64_t> ||
                                             std::is_same_v<std::decay_t<T>, int32_t> ||
                                             std::is_same_v<std::decay_t<T>, uint32_t>),
                                           int> = 0>
    inline void bind_value([[maybe_unused]] sqlite3_stmt* stmt,
                           [[maybe_unused]] int position, [[maybe_unused]] T& _value,
                           [[maybe_unused]] const std::string& query)
    {
        throw std::runtime_error("Unsupported type for binding!");
    }

    template <typename T,
              std::enable_if_t<common::traits::is_string_literal<T>(), int> = 0>
    inline void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                           const std::string& query)
    {
        validate_sqlite3_result(
            sqlite3_bind_text(stmt, position, _value, -1, SQLITE_STATIC), query.c_str(),
            "Failed to bind text! Position: ", position, ", Value: ", _value);
    }

    template <typename T,
              std::enable_if_t<std::is_floating_point_v<std::decay_t<T>>, int> = 0>
    inline void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                           const std::string& query)
    {
        validate_sqlite3_result(
            sqlite3_bind_double(stmt, position, _value), query.c_str(),
            "Failed to bind double! Position: ", position, ", Value: ", _value);
    }

    template <typename T, std::enable_if_t<std::is_same_v<std::decay_t<T>, int64_t> ||
                                               std::is_same_v<std::decay_t<T>, uint64_t>,
                                           int> = 0>
    inline void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                           const std::string& query)
    {
        validate_sqlite3_result(sqlite3_bind_int64(stmt, position, _value), query.c_str(),
                                "Failed to bind int64_t/uint64_t! Position: ", position,
                                ", Value: ", _value);
    }

    template <typename T, std::enable_if_t<std::is_same_v<std::decay_t<T>, int32_t> ||
                                               std::is_same_v<std::decay_t<T>, uint32_t>,
                                           int> = 0>
    inline void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                           const std::string& query)
    {
        validate_sqlite3_result(sqlite3_bind_int(stmt, position, _value), query.c_str(),
                                "Failed to bind int32_t/uint32_t! Position: ", position,
                                ", Value: ", _value);
    }

public:
    void initialize_schema();

    void execute_query(const std::string& query);

    size_t get_last_insert_id() const;

    /**
     * This function prepares an SQLite statement based on the provided SQL query and
     * returns a lambda that can execute the prepared statement, binding the provided
     * values to the respective placeholders in the query.
     */
    template <typename... Values>
    auto create_statement_executor(const std::string& query)
    {
        sqlite3_stmt* p_stmt;
        validate_sqlite3_result(
            sqlite3_prepare_v2(_sqlite3_db_temp, query.c_str(), -1, &p_stmt, nullptr),
            query.c_str(), "Failed to create statement!");
        std::shared_ptr<sqlite3_stmt> stmt{ p_stmt, sqlite3_finalize };

        return [stmt, query, this](Values... value) {
            std::lock_guard lock{ _mutex };
            int             position = 1;

            ((bind_value(stmt.get(), position++, value, query)), ...);

            validate_sqlite3_result(sqlite3_step(stmt.get()), query.c_str(),
                                    "Failed to execute step!\n", "Values: ", value...);
            sqlite3_reset(stmt.get());
        };
    }

    static std::string get_upid();

private:
    sqlite3* _sqlite3_db{ nullptr };
    sqlite3* _sqlite3_db_temp{ nullptr };
};

}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
