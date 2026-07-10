// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gmock/gmock.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace profiler_hub::data_storage::mocks
{

// Opaque fake handles. Distinct types so the type system keeps a connection
// handle and a statement handle from being confused, and so the mock can hand
// back non-null sentinels without touching real sqlite3 state.
struct mock_connection
{};

struct mock_statement
{};

/**
 * GMock recorder for the SQLite policy operations.
 *
 * A test instantiates one of these, sets EXPECT_CALL expectations on it, and
 * makes it the active recorder for the current thread with
 * mock_sqlite3::scoped_bind. The mock_sqlite3 policy forwards every call to the
 * bound recorder, so the backend under test drives these methods instead of the
 * real sqlite3 C API.
 *
 * The gmock-generated methods are an internal test-only detail; the production
 * seam (mock_sqlite3) is a plain compile-time policy with no virtual dispatch.
 */
class sqlite3_recorder
{
public:
    // Connection lifecycle
    MOCK_METHOD(int, open, (const char* path, mock_connection** out_db));
    MOCK_METHOD(int, close, (mock_connection * db));
    MOCK_METHOD(int, exec, (mock_connection * db, const char* sql));
    MOCK_METHOD(int,
                prepare,
                (mock_connection * db, const char* sql, mock_statement** out_stmt));
    MOCK_METHOD(std::string, errmsg, (mock_connection * db));
    MOCK_METHOD(std::string, errstr, (int result_code));

    // Persist an in-memory database to a file (wraps the sqlite3_backup_* dance).
    MOCK_METHOD(int,
                backup_to_file,
                (mock_connection * src, const char* dst_path, std::string& out_errmsg));

    // Statement lifecycle
    MOCK_METHOD(int, step, (mock_statement * stmt));
    MOCK_METHOD(int, reset, (mock_statement * stmt));
    MOCK_METHOD(int, clear_bindings, (mock_statement * stmt));
    MOCK_METHOD(int, finalize, (mock_statement * stmt));

    // Parameter binding
    MOCK_METHOD(int, bind_null, (mock_statement * stmt, int position));
    MOCK_METHOD(int, bind_int, (mock_statement * stmt, int position, int value));
    MOCK_METHOD(int,
                bind_int64,
                (mock_statement * stmt, int position, std::int64_t value));
    MOCK_METHOD(int, bind_double, (mock_statement * stmt, int position, double value));
    MOCK_METHOD(int,
                bind_text,
                (mock_statement * stmt, int position, std::string_view value));

    // Column extraction
    MOCK_METHOD(int, column_type, (mock_statement * stmt, int position));
    MOCK_METHOD(int, column_int, (mock_statement * stmt, int position));
    MOCK_METHOD(std::int64_t, column_int64, (mock_statement * stmt, int position));
    MOCK_METHOD(double, column_double, (mock_statement * stmt, int position));
    MOCK_METHOD(std::string, column_text, (mock_statement * stmt, int position));
};

/**
 * Compile-time SQLite policy backed by a sqlite3_recorder.
 *
 * Intended to be plugged into the templated backend in tests
 * (database_backend<mock_sqlite3>) in place of the production sqlite3
 * policy. Every operation forwards to the recorder bound by the active
 * scoped_bind on this thread; calling an operation with no bound recorder is a
 * test bug and fails fast.
 *
 * This type is the canonical statement of the SQLite policy contract: the
 * production sqlite_api_policy must expose the same member names, signatures,
 * associated types, and status constants.
 */
struct mock_sqlite3
{
    using database_t  = mock_connection*;
    using statement_t = mock_statement*;

    // Status constants the backend compares results against. Values mirror the
    // real sqlite3 macros so any accidental hard-coded comparison behaves the
    // same; the backend should reference these names, never the raw macros.
    static constexpr int result_ok   = 0;
    static constexpr int result_row  = 100;
    static constexpr int result_done = 101;
    static constexpr int column_null = 5;

    /**
     * RAII guard that makes a recorder the active mock for the current thread
     * for the duration of its lifetime.
     */
    class scoped_bind
    {
    public:
        explicit scoped_bind(sqlite3_recorder& recorder) noexcept
        {
            active_slot() = &recorder;
        }

        ~scoped_bind() { active_slot() = nullptr; }

        scoped_bind(const scoped_bind&)            = delete;
        scoped_bind& operator=(const scoped_bind&) = delete;
        scoped_bind(scoped_bind&&)                 = delete;
        scoped_bind& operator=(scoped_bind&&)      = delete;
    };

    static int open(const char* path, database_t* out_db)
    {
        return active().open(path, out_db);
    }
    static int close(database_t db) { return active().close(db); }
    static int exec(database_t db, const char* sql) { return active().exec(db, sql); }
    static int prepare(database_t db, const char* sql, statement_t* out_stmt)
    {
        return active().prepare(db, sql, out_stmt);
    }
    static std::string errmsg(database_t db) { return active().errmsg(db); }
    static std::string errstr(int result_code) { return active().errstr(result_code); }

    static int backup_to_file(database_t   src,
                              const char*  dst_path,
                              std::string& out_errmsg)
    {
        return active().backup_to_file(src, dst_path, out_errmsg);
    }

    static int step(statement_t stmt) { return active().step(stmt); }
    static int reset(statement_t stmt) { return active().reset(stmt); }
    static int clear_bindings(statement_t stmt) { return active().clear_bindings(stmt); }
    static int finalize(statement_t stmt) { return active().finalize(stmt); }

    static int bind_null(statement_t stmt, int position)
    {
        return active().bind_null(stmt, position);
    }
    static int bind_int(statement_t stmt, int position, int value)
    {
        return active().bind_int(stmt, position, value);
    }
    static int bind_int64(statement_t stmt, int position, std::int64_t value)
    {
        return active().bind_int64(stmt, position, value);
    }
    static int bind_double(statement_t stmt, int position, double value)
    {
        return active().bind_double(stmt, position, value);
    }
    static int bind_text(statement_t stmt, int position, std::string_view value)
    {
        return active().bind_text(stmt, position, value);
    }

    static int column_type(statement_t stmt, int position)
    {
        return active().column_type(stmt, position);
    }
    static int column_int(statement_t stmt, int position)
    {
        return active().column_int(stmt, position);
    }
    static std::int64_t column_int64(statement_t stmt, int position)
    {
        return active().column_int64(stmt, position);
    }
    static double column_double(statement_t stmt, int position)
    {
        return active().column_double(stmt, position);
    }
    static std::string column_text(statement_t stmt, int position)
    {
        return active().column_text(stmt, position);
    }

private:
    static sqlite3_recorder*& active_slot() noexcept
    {
        static thread_local sqlite3_recorder* recorder = nullptr;
        return recorder;
    }

    static sqlite3_recorder& active()
    {
        auto* recorder = active_slot();
        if(recorder == nullptr)
        {
            ADD_FAILURE()
                << "mock_sqlite3 used without an active sqlite3_recorder scoped_bind";
            throw std::logic_error(
                "mock_sqlite3 used without an active sqlite3_recorder scoped_bind");
        }
        return *recorder;
    }
};

}  // namespace profiler_hub::data_storage::mocks
