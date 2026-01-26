// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "debug.hpp"
#include "traits.hpp"
#include "transaction.hpp"

#include <sqlite3.h>

#include <memory>
#include <stdexcept>
#include <string_view>

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
            "Failed to create statement");
        std::shared_ptr<sqlite3_stmt> stmt{ p_stmt, sqlite3_finalize };

        return [stmt, query, this](Values... value) {
            int position = 1;

            ((bind_value(stmt.get(), position++, value, query)), ...);

            const auto expanded_sql = sqlite3_expanded_sql(stmt.get());
            LOG_TRACE("Executing statement: {}", expanded_sql);

            validate_sqlite3_result(
                sqlite3_step(stmt.get()), expanded_sql, "Failed to execute step");
            sqlite3_reset(stmt.get());
        };
    }

    [[nodiscard]] transaction_block create_transaction_block() const noexcept
    {
        return transaction_block(m_sqlite3_inmemory);
    }

private:
    void validate_sqlite3_result(int              sqlite3_error_code,
                                 const char*      query,
                                 std::string_view context = {})
    {
        if(sqlite3_error_code == SQLITE_OK || sqlite3_error_code == SQLITE_DONE)
        {
            return;
        }

        auto message =
            fmt::format("\n===========================================================\n"
                        "Database Error: {}\n"
                        "Error code: {} ({})\n"
                        "Query: {}\n"
                        "{}"
                        "===========================================================",
                        sqlite3_errmsg(m_sqlite3_inmemory),
                        sqlite3_error_code,
                        sqlite3_errstr(sqlite3_error_code),
                        query,
                        context.empty() ? "" : fmt::format("Context: {}\n", context));

        throw std::runtime_error(message);
    }

    template <typename T>
    static constexpr bool is_int64_bindable_v =
        std::is_same_v<T, int64_t> || std::is_same_v<T, uint64_t> ||
        std::is_same_v<T, size_t>;

    template <typename T>
    static constexpr bool is_int32_bindable_v =
        std::is_same_v<T, int32_t> || std::is_same_v<T, uint32_t>;

    template <typename T>
    static constexpr bool is_text_bindable_v = std::is_same_v<T, const char*>;

    template <typename T>
    static constexpr bool is_double_bindable_v = std::is_floating_point_v<T>;

    template <typename T>
    static constexpr bool is_supported_bind_type_v =
        is_int64_bindable_v<T> || is_int32_bindable_v<T> || is_text_bindable_v<T> ||
        is_double_bindable_v<T>;

    void bind_null(sqlite3_stmt* stmt, int position, const std::string& query)
    {
        LOG_TRACE("bind_null: position={}", position);
        validate_sqlite3_result(
            sqlite3_bind_null(stmt, position),
            query.c_str(),
            fmt::format("Failed to bind NULL at position {}", position));
    }

    void bind_text(sqlite3_stmt*      stmt,
                   int                position,
                   const char*        val,
                   const std::string& query)
    {
        LOG_TRACE("bind_text: position={}, value={}", position, val ? val : "(null)");
        validate_sqlite3_result(
            sqlite3_bind_text(stmt, position, val, -1, SQLITE_STATIC),
            query.c_str(),
            fmt::format("Failed to bind text at position {}, value: {}",
                        position,
                        val ? val : "(null)"));
    }

    void bind_double(sqlite3_stmt*      stmt,
                     int                position,
                     double             val,
                     const std::string& query)
    {
        LOG_TRACE("bind_double: position={}, value={}", position, val);
        validate_sqlite3_result(
            sqlite3_bind_double(stmt, position, val),
            query.c_str(),
            fmt::format(
                "Failed to bind double at position {}, value: {}", position, val));
    }

    void bind_int64(sqlite3_stmt*      stmt,
                    int                position,
                    int64_t            val,
                    const std::string& query)
    {
        LOG_TRACE("bind_int64: position={}, value={}", position, val);
        validate_sqlite3_result(
            sqlite3_bind_int64(stmt, position, val),
            query.c_str(),
            fmt::format("Failed to bind int64 at position {}, value: {}", position, val));
    }

    void bind_int32(sqlite3_stmt*      stmt,
                    int                position,
                    int32_t            val,
                    const std::string& query)
    {
        LOG_TRACE("bind_int32: position={}, value={}", position, val);
        validate_sqlite3_result(
            sqlite3_bind_int(stmt, position, val),
            query.c_str(),
            fmt::format("Failed to bind int32 at position {}, value: {}", position, val));
    }

    template <typename T>
    void bind_value(sqlite3_stmt* stmt, int position, T&& value, const std::string& query)
    {
        using decayed_t = std::decay_t<T>;

        if constexpr(common::traits::is_optional_v<decayed_t>)
        {
            if(!value.has_value())
            {
                bind_null(stmt, position, query);
            }
            else
            {
                bind_value(stmt, position, *std::forward<T>(value), query);
            }
        }
        else if constexpr(is_text_bindable_v<decayed_t>)
        {
            if(value == nullptr)
            {
                bind_null(stmt, position, query);
            }
            else
            {
                bind_text(stmt, position, value, query);
            }
        }
        else if constexpr(is_double_bindable_v<decayed_t>)
        {
            bind_double(stmt, position, static_cast<double>(value), query);
        }
        else if constexpr(is_int64_bindable_v<decayed_t>)
        {
            bind_int64(stmt, position, static_cast<int64_t>(value), query);
        }
        else if constexpr(is_int32_bindable_v<decayed_t>)
        {
            bind_int32(stmt, position, static_cast<int32_t>(value), query);
        }
        else
        {
            static_assert(!std::is_same_v<decayed_t, decayed_t>,
                          "Unsupported type for binding");
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
