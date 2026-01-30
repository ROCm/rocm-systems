// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "debug.hpp"
#include "spdlog/fmt/bundled/core.h"
#include "statement_result.hpp"
#include "traits.hpp"
#include "transaction.hpp"

#include <sqlite3.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>

namespace rocstorage::data_storage
{

template <typename... Ts>
struct bind_types
{};

class database
{
public:
    enum class database_type_t
    {
        auto_detect = 0,
        in_memory   = 1,
        on_disk     = 2,
        mmap        = 3,
    };

    explicit database(std::string     abs_db_path,
                      std::string     uuid,
                      database_type_t database_type = database_type_t::in_memory);
    database()                      = delete;
    database(database&)             = delete;
    database& operator=(database&)  = delete;
    database(database&&)            = delete;
    database& operator=(database&&) = delete;
    ~database();

    void                      flush();
    void                      initialize_schema();
    void                      execute_query(const std::string& query);
    [[nodiscard]] size_t      get_last_insert_id() const;
    [[nodiscard]] std::string get_uuid() const;

    /**
     * This function prepares an SQLite statement based on the provided SQL query
     * and returns a lambda that can execute the prepared statement, binding the
     * provided values to the respective placeholders in the query.
     */
    template <typename... Values>
    auto create_write_statement_executor(const std::string& query)
    {
        sqlite3_stmt* p_stmt = nullptr;
        validate_sqlite3_result(
            sqlite3_prepare_v2(m_sqlite3, query.c_str(), -1, &p_stmt, nullptr),
            query.c_str(),
            "Failed to create statement");
        std::shared_ptr<sqlite3_stmt> const stmt{ p_stmt, sqlite3_finalize };

        return [stmt, query, this](Values... value) {
            int position = 1;

            ((bind_value(stmt.get(), position++, value, query)), ...);

            // TODO: Add as optional
            auto expanded_sql = std::unique_ptr<char, decltype(&sqlite3_free)>(
                sqlite3_expanded_sql(stmt.get()), sqlite3_free);
            LOG_TRACE("Executing statement: {}", expanded_sql);

            validate_sqlite3_result(
                sqlite3_step(stmt.get()), expanded_sql.get(), "Failed to execute step");
            sqlite3_reset(stmt.get());
        };
    }

    template <typename T, typename... BindTypes, typename... Members>
    auto create_read_statement_executor_impl(bind_types<BindTypes...> /*tag*/,
                                             const std::string& query,
                                             Members            T::*... members)
    {
        sqlite3_stmt* p_stmt = nullptr;
        validate_sqlite3_result(
            sqlite3_prepare_v2(m_sqlite3, query.c_str(), -1, &p_stmt, nullptr),
            query.c_str(),
            "Failed to create query reader statement");

        std::shared_ptr<sqlite3_stmt> stmt{ p_stmt, sqlite3_finalize };

        return [stmt, members..., query, this](
                   BindTypes... bind_values) -> statement_result<T> {
            sqlite3_reset(stmt.get());
            sqlite3_clear_bindings(stmt.get());

            int position = 1;
            ((this->bind_value(stmt.get(), position++, bind_values, query)), ...);

            return statement_result<T>(stmt, members...);
        };
    }

    /// Create a reusable query reader that maps results to struct T.
    /// For queries with bind parameters, specify them wrapped in bind_types<>:
    ///   create_read_statement_executor<MyStruct, bind_types<int, bool>>(query,
    ///   &MyStruct::field)
    /// For queries without bind parameters:
    ///   create_read_statement_executor<MyStruct>(query, &MyStruct::field)
    template <typename T, typename BindTypesPack = bind_types<>, typename... Members>
    auto create_read_statement_executor(const std::string& query, Members T::*... members)
    {
        return create_read_statement_executor_impl<T>(BindTypesPack{}, query, members...);
    }

    [[nodiscard]] transaction_block create_transaction_block() const noexcept
    {
        return transaction_block(m_sqlite3);
    }

private:
    [[nodiscard]] std::vector<std::string> discover_uuids();

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
                        sqlite3_errmsg(m_sqlite3),
                        sqlite3_error_code,
                        sqlite3_errstr(sqlite3_error_code),
                        query,
                        context.empty() ? "" : fmt::format("Context: {}\n", context));

        throw std::runtime_error(message);
    }

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
                        (val != nullptr) ? val : "(null)"));
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
        else if constexpr(common::traits::is_text_bindable_v<decayed_t>)
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
        else if constexpr(common::traits::is_double_bindable_v<decayed_t>)
        {
            bind_double(stmt, position, static_cast<double>(value), query);
        }
        else if constexpr(common::traits::is_int64_bindable_v<decayed_t>)
        {
            bind_int64(stmt, position, static_cast<int64_t>(value), query);
        }
        else if constexpr(common::traits::is_int32_bindable_v<decayed_t>)
        {
            bind_int32(stmt, position, static_cast<int32_t>(value), query);
        }
        else
        {
            static_assert(!std::is_same_v<decayed_t, decayed_t>,
                          "Unsupported type for binding");
        }
    }

    sqlite3*        m_sqlite3{ nullptr };
    std::string     m_db_path;
    std::string     m_uuid;
    database_type_t m_database_type;
    bool            m_initialized{ false };
    bool            m_flushed{ false };
};

}  // namespace rocstorage::data_storage
