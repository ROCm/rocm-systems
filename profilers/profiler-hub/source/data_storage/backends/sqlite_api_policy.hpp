// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sqlite3.h>

#include <cstdint>
#include <string>
#include <string_view>

namespace profiler_hub::data_storage
{

/**
 * Production SQLite policy: forwards every operation to the sqlite3 C API.
 *
 * This struct is the compile-time seam that makes the SQLite C API swappable in
 * database_backend. It is the ONLY production translation unit that includes
 * <sqlite3.h>. Its member names, signatures, associated types, and status
 * constants mirror mocks::mock_sqlite3 exactly so the two are interchangeable as
 * the SqlitePolicy template argument.
 *
 * All operations are static; there is no virtual dispatch on this seam.
 */
struct sqlite_api_policy
{
    using database_t  = sqlite3*;
    using statement_t = sqlite3_stmt*;

    static constexpr int result_ok   = SQLITE_OK;
    static constexpr int result_row  = SQLITE_ROW;
    static constexpr int result_done = SQLITE_DONE;
    static constexpr int column_null = SQLITE_NULL;

    static int open(const char* path, database_t* out_db) noexcept
    {
        return sqlite3_open(path, out_db);
    }

    static int close(database_t db) noexcept { return sqlite3_close(db); }

    static int exec(database_t db, const char* sql) noexcept
    {
        return sqlite3_exec(db, sql, nullptr, nullptr, nullptr);
    }

    static int prepare(database_t db, const char* sql, statement_t* out_stmt) noexcept
    {
        return sqlite3_prepare_v2(db, sql, -1, out_stmt, nullptr);
    }

    static std::string errmsg(database_t db) { return std::string{ sqlite3_errmsg(db) }; }

    static std::string errstr(int result_code)
    {
        return std::string{ sqlite3_errstr(result_code) };
    }

    /**
     * Persist an in-memory database to a file via the sqlite3_backup_* dance.
     * @param src Source (in-memory) connection to copy from.
     * @param dst_path Destination file path to create/overwrite.
     * @return result_ok on full success, otherwise a non-ok sqlite3 result code.
     */
    static int backup_to_file(database_t   src,
                              const char*  dst_path,
                              std::string& out_errmsg)
    {
        database_t out_db = nullptr;
        const int  rc     = sqlite3_open(dst_path, &out_db);
        if(rc != SQLITE_OK)
        {
            out_errmsg = sqlite3_errmsg(out_db);
            sqlite3_close(out_db);
            return rc;
        }

        sqlite3_backup* backup = sqlite3_backup_init(out_db, "main", src, "main");
        if(backup == nullptr)
        {
            out_errmsg = sqlite3_errmsg(out_db);
            sqlite3_close(out_db);
            return SQLITE_ERROR;
        }

        const int step_rc   = sqlite3_backup_step(backup, -1);
        const int finish_rc = sqlite3_backup_finish(backup);

        if(step_rc != SQLITE_DONE || finish_rc != SQLITE_OK)
        {
            out_errmsg = sqlite3_errmsg(out_db);
        }

        sqlite3_close(out_db);

        if(step_rc != SQLITE_DONE) return step_rc;
        if(finish_rc != SQLITE_OK) return finish_rc;
        return SQLITE_OK;
    }

    static int step(statement_t stmt) noexcept { return sqlite3_step(stmt); }
    static int reset(statement_t stmt) noexcept { return sqlite3_reset(stmt); }
    static int clear_bindings(statement_t stmt) noexcept
    {
        return sqlite3_clear_bindings(stmt);
    }
    static int finalize(statement_t stmt) noexcept { return sqlite3_finalize(stmt); }

    static int bind_null(statement_t stmt, int position) noexcept
    {
        return sqlite3_bind_null(stmt, position);
    }
    static int bind_int(statement_t stmt, int position, int value) noexcept
    {
        return sqlite3_bind_int(stmt, position, value);
    }
    static int bind_int64(statement_t stmt, int position, std::int64_t value) noexcept
    {
        return sqlite3_bind_int64(stmt, position, value);
    }
    static int bind_double(statement_t stmt, int position, double value) noexcept
    {
        return sqlite3_bind_double(stmt, position, value);
    }
    static int bind_text(statement_t stmt, int position, std::string_view value) noexcept
    {
        return sqlite3_bind_text(
            stmt, position, value.data(), static_cast<int>(value.size()), SQLITE_STATIC);
    }

    static int column_type(statement_t stmt, int position) noexcept
    {
        return sqlite3_column_type(stmt, position);
    }
    static int column_int(statement_t stmt, int position) noexcept
    {
        return sqlite3_column_int(stmt, position);
    }
    static std::int64_t column_int64(statement_t stmt, int position) noexcept
    {
        return sqlite3_column_int64(stmt, position);
    }
    static double column_double(statement_t stmt, int position) noexcept
    {
        return sqlite3_column_double(stmt, position);
    }
    static std::string column_text(statement_t stmt, int position)
    {
        const auto* text =
            reinterpret_cast<const char*>(sqlite3_column_text(stmt, position));
        return text != nullptr ? std::string{ text } : std::string{};
    }
};

}  // namespace profiler_hub::data_storage
