// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include "common/traits.hpp"
#include "logger/debug.hpp"
#include <cstdint>

#include <memory>
#include <mutex>
#include <sqlite3.h>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace rocprofsys
{
namespace rocpd
{
namespace data_storage
{
class database
{
public:
    explicit database(int pid, int ppid, std::string output_path);
    database()                      = delete;
    database(database&)             = delete;
    database& operator=(database&)  = delete;
    database(database&&)            = delete;
    database& operator=(database&&) = delete;

    void flush();

    ~database();

private:
    template <typename... Args>
    void validate_sqlite3_result(int sqlite3_error_code, const char* query,
                                 Args&&... args)
    {
        std::stringstream ss;
        ss << "\n===========================================================\n";
        ss << "Database Error\n";
        ((ss << args << " "), ...);
        ss << "\nQuery: " << query << "\n";
        // Fetch error message of last sqlite3_* call
        const auto* error_message = sqlite3_errstr(sqlite3_error_code);
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
        ss << " [Sqlite3 error: " << error_message;
        ss << " (Extended error message: " << sqlite3_errmsg(_sqlite3_db_temp) << ")]";
        throw std::runtime_error(ss.str());
    }

    template <typename T>
    static constexpr bool sql_int64_v = std::is_same_v<std::decay_t<T>, std::int64_t> ||
                                        std::is_same_v<std::decay_t<T>, std::uint64_t>;

    template <typename T>
    static constexpr bool sql_int32_v = std::is_same_v<std::decay_t<T>, std::int32_t> ||
                                        std::is_same_v<std::decay_t<T>, std::uint32_t>;

    template <typename T>
    static constexpr bool sql_supported_v =
        common::traits::is_string_literal<T>() ||
        std::is_floating_point_v<std::decay_t<T>> || sql_int64_v<T> || sql_int32_v<T>;

    template <typename T>
        requires(!sql_supported_v<T>)
    void bind_value([[maybe_unused]] sqlite3_stmt* stmt, [[maybe_unused]] int position,
                    [[maybe_unused]] T& _value, [[maybe_unused]] const std::string& query)
    {
        throw std::runtime_error("Unsupported type for binding!");
    }

    template <typename T>
        requires(common::traits::is_string_literal<T>())
    void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                    const std::string& query)
    {
        validate_sqlite3_result(
            sqlite3_bind_text(stmt, position, _value, -1, SQLITE_STATIC), query.c_str(),
            "Failed to bind text! Position: ", position, ", Value: ", _value);
    }

    template <typename T>
        requires std::is_floating_point_v<std::decay_t<T>>
    void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                    const std::string& query)
    {
        validate_sqlite3_result(
            sqlite3_bind_double(stmt, position, _value), query.c_str(),
            "Failed to bind double! Position: ", position, ", Value: ", _value);
    }

    template <typename T>
        requires sql_int64_v<T>
    void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                    const std::string& query)
    {
        validate_sqlite3_result(sqlite3_bind_int64(stmt, position, _value), query.c_str(),
                                "Failed to bind std::int64_t/std::uint64_t! Position: ",
                                position, ", Value: ", _value);
    }

    template <typename T>
        requires sql_int32_v<T>
    void bind_value(sqlite3_stmt* stmt, int position, T&& _value,
                    const std::string& query)
    {
        validate_sqlite3_result(sqlite3_bind_int(stmt, position, _value), query.c_str(),
                                "Failed to bind std::int32_t/std::uint32_t! Position: ",
                                position, ", Value: ", _value);
    }

public:
    void initialize_schema();

    void execute_query(const std::string& query);

    size_t get_last_insert_id() const;

    /**
     * This function prepares an SQLite statement based on the provided SQL query and
     * returns a lambda that can execute the prepared statement, binding the provided
     * values to the respective placeholders in the query.
     *
     * @param db_ref A shared_ptr to this database instance. The statement's deleter
     *               captures it to guarantee the connection stays open until all
     *               statements are finalized.
     */
    template <typename... Values>
    static auto create_statement_executor(const std::string&        query,
                                          std::shared_ptr<database> db_ref)
    {
        if(db_ref == nullptr)
        {
            throw std::runtime_error("Database cannot be nullptr!");
        }

        sqlite3_stmt* p_stmt;
        db_ref->validate_sqlite3_result(sqlite3_prepare_v2(db_ref->_sqlite3_db_temp,
                                                           query.c_str(), -1, &p_stmt,
                                                           nullptr),
                                        query.c_str(), "Failed to create statement!");

        std::shared_ptr<sqlite3_stmt> stmt{ p_stmt, [db = db_ref](sqlite3_stmt* s) {
                                               if(db == nullptr)
                                               {
                                                   return;
                                               }
                                               sqlite3_finalize(s);
                                           } };

        return [stmt, query, db_ref](Values... value) {
            if(db_ref == nullptr)
            {
                return;
            }

            int position = 1;

            ((db_ref->bind_value(stmt.get(), position++, value, query)), ...);

            db_ref->validate_sqlite3_result(sqlite3_step(stmt.get()), query.c_str(),
                                            "Failed to execute step!\n",
                                            "Values: ", value...);
            sqlite3_reset(stmt.get());
        };
    }

    std::string get_upid();

private:
    static std::string generate_upid(const int pid, const int ppid);

private:
    sqlite3*    _sqlite3_db{ nullptr };
    sqlite3*    _sqlite3_db_temp{ nullptr };
    std::string m_tag;
    std::string m_upid;
    std::string m_path;
};

}  // namespace data_storage
}  // namespace rocpd
}  // namespace rocprofsys
