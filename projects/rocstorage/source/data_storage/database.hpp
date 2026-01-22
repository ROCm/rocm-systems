// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "traits.hpp"

#include <sqlite3.h>

#include <memory>
#include <sstream>
#include <stdexcept>

namespace rocstorage
{
namespace data_storage
{
class database
{
public:
    explicit database(std::string abs_db_path, std::string uuid);
    database()                      = delete;
    database(database&)             = delete;
    database& operator=(database&)  = delete;
    database(database&&)            = delete;
    database& operator=(database&&) = delete;
    ~database();

    void        flush();
    void        initialize_schema();
    void        execute_query(const std::string& query);
    size_t      get_last_insert_id() const;
    std::string get_uuid() const;

    /**
     * This function prepares an SQLite statement based on the provided SQL query
     * and returns a lambda that can execute the prepared statement, binding the
     * provided values to the respective placeholders in the query.
     */
    template <typename... Values>
    auto create_statement_executor(const std::string& query)
    {
        sqlite3_stmt* p_stmt;
        validate_sqlite3_result(
            sqlite3_prepare_v2(m_sqlite3_inmemory, query.c_str(), -1, &p_stmt, nullptr),
            query.c_str(),
            "Failed to create statement!");
        std::shared_ptr<sqlite3_stmt> stmt{ p_stmt, sqlite3_finalize };

        return [stmt, query, this](Values... value) {
            int position = 1;

            ((bind_value(stmt.get(), position++, value, query)), ...);

            validate_sqlite3_result(sqlite3_step(stmt.get()),
                                    query.c_str(),
                                    "Failed to execute step!\n",
                                    "Values: ",
                                    value...);
            sqlite3_reset(stmt.get());
        };
    }

private:
    template <typename... Args>
    void validate_sqlite3_result(int         sqlite3_error_code,
                                 const char* query,
                                 Args&&... args)
    {
        // if(sqlite3_error_code == SQLITE_OK || sqlite3_error_code == SQLITE_DONE)
        // {
        //     return;
        // }

        // std::stringstream ss;
        // ss << "\n===========================================================\n";
        // ss << "Database Error\n";
        // ((ss << args << " "), ...);
        // ss << "\nQuery: " << query << "\n";
        // const auto* error_message = sqlite3_errstr(sqlite3_error_code);
        // switch(sqlite3_error_code)
        // {
        //     case SQLITE_CONSTRAINT:
        //     {
        //         sqlite3_stmt* stmt;

        //         ss << "Constraint violation(s): " << "\n";

        //         sqlite3_exec(m_sqlite3_inmemory,
        //                      "PRAGMA foreign_keys = OFF;",
        //                      nullptr,
        //                      nullptr,
        //                      nullptr);
        //         sqlite3_exec(m_sqlite3_inmemory, query, nullptr, nullptr, nullptr);
        //         sqlite3_exec(m_sqlite3_inmemory,
        //                      "PRAGMA foreign_keys = ON;",
        //                      nullptr,
        //                      nullptr,
        //                      nullptr);
        //         sqlite3_prepare_v2(
        //             m_sqlite3_inmemory, "PRAGMA foreign_key_check", -1, &stmt,
        //             nullptr);
        //         int rc = 0;
        //         while((rc = sqlite3_step(stmt)) == SQLITE_ROW)
        //         {
        //             const char* table  = (const char*) sqlite3_column_text(stmt, 0);
        //             int         rowid  = sqlite3_column_int(stmt, 1);
        //             const char* parent = (const char*) sqlite3_column_text(stmt, 2);
        //             int         fkid   = sqlite3_column_int(stmt, 3);

        //             ss << "  - " << "FK Violation - Table: " << (table ? table :
        //             "NULL")
        //                << ", RowID: " << rowid
        //                << ", Parent: " << (parent ? parent : "NULL") << ", FKID: " <<
        //                fkid
        //                << "\n";
        //         }

        //         sqlite3_finalize(stmt);
        //     }
        //     break;
        //     default:
        //     {
        //     }
        //     break;
        // }
        // ss << " [Sqlite3 error: " << error_message;
        // ss << " (Extended error message: " << sqlite3_errmsg(m_sqlite3_inmemory) <<
        // ")]"; throw std::runtime_error(ss.str());
    }

    template <typename T>
    static constexpr bool is_int64_bindable_v =
        std::is_same_v<common::traits::unwrap_optional_t<T>, int64_t> ||
        std::is_same_v<common::traits::unwrap_optional_t<T>, uint64_t> ||
        std::is_same_v<common::traits::unwrap_optional_t<T>, size_t>;

    template <typename T>
    static constexpr bool is_int32_bindable_v =
        std::is_same_v<common::traits::unwrap_optional_t<T>, int32_t> ||
        std::is_same_v<common::traits::unwrap_optional_t<T>, uint32_t>;

    template <typename T>
    static constexpr bool is_text_bindable_v =
        std::is_same_v<common::traits::unwrap_optional_t<T>, const char*>;

    template <typename T>
    static constexpr bool is_double_bindable_v =
        std::is_floating_point_v<common::traits::unwrap_optional_t<T>>;

    template <typename T>
    static constexpr bool is_supported_bind_type_v =
        is_int64_bindable_v<T> || is_int32_bindable_v<T> || is_text_bindable_v<T> ||
        is_double_bindable_v<T>;

    template <typename T,
              std::enable_if_t<!is_supported_bind_type_v<std::decay_t<T>>, int> = 0>
    void bind_value([[maybe_unused]] sqlite3_stmt*      stmt,
                    [[maybe_unused]] int                position,
                    [[maybe_unused]] T&                 _value,
                    [[maybe_unused]] const std::string& query)
    {
        throw std::runtime_error("Unsupported type for binding!");
    }

    template <typename T, std::enable_if_t<is_text_bindable_v<std::decay_t<T>>, int> = 0>
    void bind_value(sqlite3_stmt*      stmt,
                    int                position,
                    T&&                _value,
                    const std::string& query)
    {
        if constexpr(common::traits::is_optional_v<std::decay_t<T>>)
        {
            if(common::traits::is_null_value(_value))
            {
                validate_sqlite3_result(sqlite3_bind_null(stmt, position),
                                        query.c_str(),
                                        "Failed to bind NULL! Position: ",
                                        position);
            }
        }
        else
        {
            auto val = common::traits::get_value(std::forward<T>(_value));
            validate_sqlite3_result(
                sqlite3_bind_text(stmt, position, val, -1, SQLITE_STATIC),
                query.c_str(),
                "Failed to bind text! Position: ",
                position,
                ", Value: ",
                val);
        }
    }

    template <typename T,
              std::enable_if_t<is_double_bindable_v<std::decay_t<T>>, int> = 0>
    void bind_value(sqlite3_stmt*      stmt,
                    int                position,
                    T&&                _value,
                    const std::string& query)
    {
        if constexpr(common::traits::is_optional_v<std::decay_t<T>>)
        {
            if(common::traits::is_null_value(_value))
            {
                validate_sqlite3_result(sqlite3_bind_null(stmt, position),
                                        query.c_str(),
                                        "Failed to bind NULL! Position: ",
                                        position);
            }
        }
        else
        {
            auto val = common::traits::get_value(std::forward<T>(_value));
            validate_sqlite3_result(sqlite3_bind_double(stmt, position, val),
                                    query.c_str(),
                                    "Failed to bind double! Position: ",
                                    position,
                                    ", Value: ",
                                    val);
        }
    }

    template <typename T, std::enable_if_t<is_int64_bindable_v<std::decay_t<T>>, int> = 0>
    void bind_value(sqlite3_stmt*      stmt,
                    int                position,
                    T&&                _value,
                    const std::string& query)
    {
        if constexpr(common::traits::is_optional_v<std::decay_t<T>>)
        {
            if(common::traits::is_null_value(_value))
            {
                validate_sqlite3_result(sqlite3_bind_null(stmt, position),
                                        query.c_str(),
                                        "Failed to bind NULL! Position: ",
                                        position);
            }
        }
        else
        {
            auto val = common::traits::get_value(std::forward<T>(_value));
            validate_sqlite3_result(
                sqlite3_bind_int64(stmt, position, static_cast<int64_t>(val)),
                query.c_str(),
                "Failed to bind int64! Position: ",
                position,
                ", Value: ",
                val);
        }
    }

    template <typename T, std::enable_if_t<is_int32_bindable_v<std::decay_t<T>>, int> = 0>
    void bind_value(sqlite3_stmt*      stmt,
                    int                position,
                    T&&                _value,
                    const std::string& query)
    {
        if constexpr(common::traits::is_optional_v<std::decay_t<T>>)
        {
            if(common::traits::is_null_value(_value))
            {
                validate_sqlite3_result(sqlite3_bind_null(stmt, position),
                                        query.c_str(),
                                        "Failed to bind NULL! Position: ",
                                        position);
            }
        }
        else
        {
            auto val = common::traits::get_value(std::forward<T>(_value));
            validate_sqlite3_result(
                sqlite3_bind_int(stmt, position, static_cast<int>(val)),
                query.c_str(),
                "Failed to bind int32! Position: ",
                position,
                ", Value: ",
                val);
        }
    }

private:
    sqlite3*    m_sqlite3_inmemory{ nullptr };
    std::string m_db_path{};
    std::string m_uuid{};
    bool        m_initialized{ false };
    bool        m_flushed{ false };
};

}  // namespace data_storage
}  // namespace rocstorage
