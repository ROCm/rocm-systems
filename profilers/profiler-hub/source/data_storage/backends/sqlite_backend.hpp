// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/traits.hpp"
#include "debug.hpp"
#include "spdlog/fmt/bundled/core.h"

#include <sqlite3.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace profiler_hub::data_storage
{

template <typename... Ts>
struct bind_types
{};

class sqlite_backend : public std::enable_shared_from_this<sqlite_backend>
{
public:
    // =========================================================================
    // Opaque statement handle
    //
    // The deleter captures shared_ptr<sqlite_backend> so the connection stays
    // alive until every statement referencing it is destroyed.
    // =========================================================================
    class statement_handle
    {
        friend class sqlite_backend;

    public:
        statement_handle() = default;
        explicit operator bool() const noexcept { return m_stmt != nullptr; }

    private:
        statement_handle(sqlite3_stmt* raw, std::shared_ptr<sqlite_backend> owner)
        : m_stmt{ raw, [db = std::move(owner)](sqlite3_stmt* s) {
                     if(db) sqlite3_finalize(s);
                 } }
        {}

        [[nodiscard]] sqlite3_stmt*   raw() const noexcept { return m_stmt.get(); }
        std::shared_ptr<sqlite3_stmt> m_stmt;
    };

    // =========================================================================
    // Transaction guard (RAII)
    // =========================================================================
    class transaction_guard
    {
    public:
        explicit transaction_guard(std::shared_ptr<sqlite_backend> backend) noexcept
        : m_backend{ std::move(backend) }
        , m_uncaught_on_entry{ std::uncaught_exceptions() }
        {
            run_prepared(m_backend->m_begin_stmt);
        }

        ~transaction_guard()
        {
            if(std::uncaught_exceptions() > m_uncaught_on_entry)
            {
                run_prepared(m_backend->m_rollback_stmt);
            }
            else
            {
                run_prepared(m_backend->m_commit_stmt);
            }
        }

        transaction_guard(const transaction_guard&)            = delete;
        transaction_guard& operator=(const transaction_guard&) = delete;
        transaction_guard(transaction_guard&&)                 = delete;
        transaction_guard& operator=(transaction_guard&&)      = delete;

    private:
        static void run_prepared(sqlite3_stmt* stmt) noexcept
        {
            if(stmt == nullptr) return;
            int rc = sqlite3_step(stmt);
            if(rc != SQLITE_DONE && rc != SQLITE_OK)
            {
                LOG_ERROR("transaction_guard: sqlite3_step failed ({}): {}",
                          rc,
                          sqlite3_errstr(rc));
            }
            sqlite3_reset(stmt);
        }

        std::shared_ptr<sqlite_backend> m_backend;
        int                             m_uncaught_on_entry;
    };

    // =========================================================================
    // Prepared insert statement
    //
    // Concrete callable type that replaces std::function for write executors.
    // Eliminates heap allocation and indirect call overhead of std::function.
    // =========================================================================
    template <typename... Values>
    class prepared_insert_statement
    {
        friend class sqlite_backend;

    public:
        prepared_insert_statement() = default;

        void operator()(Values... value) const
        {
            auto* raw      = m_stmt.raw();
            int   position = 1;

            ((m_backend->bind_value(
                 raw, position++, std::forward<Values>(value), m_query)),
             ...);

            m_backend->validate_sqlite3_result(
                sqlite3_step(raw), m_query.c_str(), "Failed to execute step");
            sqlite3_reset(raw);
        }

    private:
        prepared_insert_statement(std::shared_ptr<sqlite_backend> backend,
                                  statement_handle                stmt,
                                  std::string                     query)
        : m_backend(std::move(backend))
        , m_stmt(std::move(stmt))
        , m_query(std::move(query))
        {}

        std::shared_ptr<sqlite_backend> m_backend;
        statement_handle                m_stmt;
        std::string                     m_query;
    };

    // =========================================================================
    // Result set -- replaces statement_result<T>
    //
    // Captures shared_ptr<sqlite_backend> for lifetime safety.
    // Uses backend's extract_column() for type-safe reading.
    // =========================================================================
    template <typename T>
    class result_set
    {
    public:
        template <typename... Members>
        result_set(std::shared_ptr<sqlite_backend> backend,
                   statement_handle                stmt,
                   Members                         T::*... members)
        : m_backend{ std::move(backend) }
        , m_stmt{ std::move(stmt) }
        , m_extractor{ [members...](sqlite_backend& be, sqlite3_stmt* raw, T& obj) {
            int pos = 0;
            ((be.extract_column(raw, pos++, obj.*members)), ...);
        } }
        {}

        std::vector<T> to_vector()
        {
            std::vector<T> results;
            auto*          raw = m_stmt.raw();
            while(sqlite3_step(raw) == SQLITE_ROW)
            {
                results.emplace_back();
                m_extractor(*m_backend, raw, results.back());
            }
            return results;
        }

    private:
        std::shared_ptr<sqlite_backend>                         m_backend;
        statement_handle                                        m_stmt;
        std::function<void(sqlite_backend&, sqlite3_stmt*, T&)> m_extractor;
    };

    // =========================================================================
    // Public types
    // =========================================================================
    using statement_handle_t  = statement_handle;
    using transaction_guard_t = transaction_guard;

    enum class storage_mode_t
    {
        in_memory = 0,
        on_disk   = 1,
    };

    // =========================================================================
    // Factory -- enforces shared_ptr creation for enable_shared_from_this
    // =========================================================================
    static std::shared_ptr<sqlite_backend> create(
        std::string    db_path,
        std::string    uuid,
        storage_mode_t mode = storage_mode_t::in_memory);

    ~sqlite_backend();

    sqlite_backend()                                 = delete;
    sqlite_backend(const sqlite_backend&)            = delete;
    sqlite_backend& operator=(const sqlite_backend&) = delete;
    sqlite_backend(sqlite_backend&&)                 = delete;
    sqlite_backend& operator=(sqlite_backend&&)      = delete;

    // =========================================================================
    // Schema & admin
    // =========================================================================
    void                      initialize_schema();
    void                      execute(const std::string& query);
    void                      flush();
    [[nodiscard]] std::string get_uuid() const;

    // =========================================================================
    // Write statement executor
    //
    // The returned lambda captures shared_ptr<sqlite_backend> + statement_handle.
    // Connection stays alive as long as any statement does.
    // =========================================================================
    template <typename... Values>
    prepared_insert_statement<Values...> create_write_statement_executor(
        const std::string& query)
    {
        return prepared_insert_statement<Values...>{ shared_from_this(),
                                                     prepare(query),
                                                     query };
    }

    // =========================================================================
    // Read statement executor
    //
    // Same lifetime guarantee. Returns a lambda that produces a result_set<T>.
    // =========================================================================
    template <typename T, typename BindTypesPack = bind_types<>, typename... Members>
    auto create_read_statement_executor(const std::string& query, Members T::*... members)
    {
        return create_read_statement_executor_impl<T>(BindTypesPack{}, query, members...);
    }

    // =========================================================================
    // Transaction
    // =========================================================================
    [[nodiscard]] transaction_guard_t begin_transaction()
    {
        return transaction_guard_t{ shared_from_this() };
    }

private:
    // =========================================================================
    // Private constructor -- use create() factory
    // =========================================================================
    sqlite_backend(std::string db_path, std::string uuid, storage_mode_t mode);

    // =========================================================================
    // Statement preparation
    // =========================================================================
    [[nodiscard]] statement_handle_t prepare(const std::string& query)
    {
        sqlite3_stmt* raw = nullptr;
        validate_sqlite3_result(
            sqlite3_prepare_v2(m_sqlite3, query.c_str(), -1, &raw, nullptr),
            query.c_str(),
            "Failed to create statement");
        return statement_handle_t{ raw, shared_from_this() };
    }

    // =========================================================================
    // Read executor implementation
    // =========================================================================
    template <typename T, typename... BindTypes, typename... Members>
    auto create_read_statement_executor_impl(bind_types<BindTypes...> /*tag*/,
                                             const std::string& query,
                                             Members            T::*... members)
    {
        auto self = shared_from_this();
        auto stmt = prepare(query);

        return
            [self, stmt, members..., query](BindTypes... bind_values) -> result_set<T> {
                auto* raw = stmt.raw();
                sqlite3_reset(raw);
                sqlite3_clear_bindings(raw);

                int position = 1;
                ((self->bind_value(raw, position++, bind_values, query)), ...);

                return result_set<T>{ self, stmt, members... };
            };
    }

    // =========================================================================
    // UUID discovery
    // =========================================================================
    [[nodiscard]] std::vector<std::string> discover_uuids();

    // =========================================================================
    // Value binding
    // =========================================================================
    void bind_null(sqlite3_stmt* stmt, int position, const std::string& query)
    {
        LOG_TRACE("bind_null: position={}", position);
        validate_sqlite3_result(
            sqlite3_bind_null(stmt, position), query.c_str(), [position] {
                return fmt::format("Failed to bind NULL at position {}", position);
            });
    }

    void bind_text(sqlite3_stmt*      stmt,
                   int                position,
                   std::string_view   val,
                   const std::string& query)
    {
        LOG_TRACE("bind_text: position={}, value={}", position, val);
        validate_sqlite3_result(
            sqlite3_bind_text(
                stmt, position, val.data(), static_cast<int>(val.size()), SQLITE_STATIC),
            query.c_str(),
            [position, val] {
                return fmt::format(
                    "Failed to bind text at position {}, value: {}", position, val);
            });
    }

    void bind_double(sqlite3_stmt*      stmt,
                     int                position,
                     double             val,
                     const std::string& query)
    {
        LOG_TRACE("bind_double: position={}, value={}", position, val);
        validate_sqlite3_result(
            sqlite3_bind_double(stmt, position, val), query.c_str(), [position, val] {
                return fmt::format(
                    "Failed to bind double at position {}, value: {}", position, val);
            });
    }

    void bind_int64(sqlite3_stmt*      stmt,
                    int                position,
                    int64_t            val,
                    const std::string& query)
    {
        LOG_TRACE("bind_int64: position={}, value={}", position, val);
        validate_sqlite3_result(
            sqlite3_bind_int64(stmt, position, val), query.c_str(), [position, val] {
                return fmt::format(
                    "Failed to bind int64 at position {}, value: {}", position, val);
            });
    }

    void bind_int32(sqlite3_stmt*      stmt,
                    int                position,
                    int32_t            val,
                    const std::string& query)
    {
        LOG_TRACE("bind_int32: position={}, value={}", position, val);
        validate_sqlite3_result(
            sqlite3_bind_int(stmt, position, val), query.c_str(), [position, val] {
                return fmt::format(
                    "Failed to bind int32 at position {}, value: {}", position, val);
            });
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
        else if constexpr(std::is_same_v<decayed_t, const char*>)
        {
            if(value == nullptr)
            {
                bind_null(stmt, position, query);
            }
            else
            {
                bind_text(stmt, position, std::string_view(value), query);
            }
        }
        else if constexpr(std::is_same_v<decayed_t, std::string_view>)
        {
            bind_text(stmt, position, value, query);
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

    // =========================================================================
    // Column extraction (for result_set)
    // =========================================================================
    template <typename T>
    void extract_column(sqlite3_stmt* stmt, int position, T& value)
    {
        using decayed_t = std::decay_t<T>;

        if constexpr(common::traits::is_optional_v<decayed_t>)
        {
            if(sqlite3_column_type(stmt, position) == SQLITE_NULL)
            {
                value = std::nullopt;
            }
            else
            {
                using inner_type_t = typename decayed_t::value_type;
                inner_type_t inner_value;
                extract_column(stmt, position, inner_value);
                value = inner_value;
            }
        }
        else if constexpr(std::is_same_v<decayed_t, std::string>)
        {
            const unsigned char* text = sqlite3_column_text(stmt, position);
            if(text != nullptr)
            {
                value = std::string{ reinterpret_cast<const char*>(text) };
            }
            else
            {
                value = std::string{};
            }
        }
        else if constexpr(common::traits::is_double_bindable_v<decayed_t>)
        {
            value = sqlite3_column_double(stmt, position);
        }
        else if constexpr(common::traits::is_int64_bindable_v<decayed_t>)
        {
            if constexpr(std::is_same_v<decayed_t, size_t>)
            {
                value = static_cast<size_t>(sqlite3_column_int64(stmt, position));
            }
            else
            {
                value = sqlite3_column_int64(stmt, position);
            }
        }
        else if constexpr(common::traits::is_int32_bindable_v<decayed_t>)
        {
            value = sqlite3_column_int(stmt, position);
        }
        else
        {
            static_assert(!std::is_same_v<decayed_t, decayed_t>,
                          "Unsupported type for column value");
        }
    }

    // =========================================================================
    // Error handling
    // =========================================================================
    void validate_sqlite3_result(int              sqlite3_error_code,
                                 const char*      query,
                                 std::string_view context = {})
    {
        if(sqlite3_error_code == SQLITE_OK || sqlite3_error_code == SQLITE_DONE)
        {
            return;
        }
        throw_sqlite_error(sqlite3_error_code, query, context);
    }

    /**
     * Overload that builds the context message lazily.
     *
     * The callable is invoked only on failure, so per-call fmt::format and
     * the string allocation it requires are skipped on the SQLITE_OK fast
     * path (the dominant case for bulk inserts).
     */
    template <
        typename ContextFn,
        std::enable_if_t<!std::is_convertible_v<ContextFn, std::string_view>, int> = 0>
    void validate_sqlite3_result(int         sqlite3_error_code,
                                 const char* query,
                                 ContextFn&& context_fn)
    {
        if(sqlite3_error_code == SQLITE_OK || sqlite3_error_code == SQLITE_DONE)
        {
            return;
        }
        throw_sqlite_error(
            sqlite3_error_code, query, std::forward<ContextFn>(context_fn)());
    }

    [[noreturn]] void throw_sqlite_error(int              sqlite3_error_code,
                                         const char*      query,
                                         std::string_view context)
    {
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

    sqlite3*       m_sqlite3{ nullptr };
    sqlite3_stmt*  m_begin_stmt{ nullptr };
    sqlite3_stmt*  m_commit_stmt{ nullptr };
    sqlite3_stmt*  m_rollback_stmt{ nullptr };
    std::string    m_db_path;
    std::string    m_uuid;
    storage_mode_t m_mode;
    bool           m_initialized{ false };
    bool           m_flushed{ false };
};

}  // namespace profiler_hub::data_storage
