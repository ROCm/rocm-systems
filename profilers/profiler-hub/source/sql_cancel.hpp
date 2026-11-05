// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <exception>

struct ph_future;

namespace profiler_hub
{

/** Thrown when sqlite3_interrupt aborts the statement a call is running. */
class sql_interrupted : public std::exception
{
public:
    [[nodiscard]] const char* what() const noexcept override
    {
        return "SQLite query interrupted";
    }
};

/**
 * Makes SQL started on this thread interruptible by @p future.
 * A null future leaves the surrounding scope in place.
 */
class sql_cancel_scope
{
public:
    explicit sql_cancel_scope(ph_future* future) noexcept;
    ~sql_cancel_scope();

    sql_cancel_scope(const sql_cancel_scope&)            = delete;
    sql_cancel_scope& operator=(const sql_cancel_scope&) = delete;

private:
    ph_future* m_previous{ nullptr };
    bool       m_active{ false };
};

/**
 * Publishes the connection that is about to run SQL so ph_future_cancel can
 * call @p interrupt from the cancelling thread. @p interrupt must be safe to
 * call from another thread and must not throw.
 */
class running_sql
{
public:
    running_sql(void (*interrupt)(void*) noexcept, void* arg);
    ~running_sql();

    running_sql(const running_sql&)            = delete;
    running_sql& operator=(const running_sql&) = delete;

private:
    ph_future* m_future{ nullptr };
    void*      m_arg{ nullptr };
    bool       m_bound{ false };
};

[[nodiscard]] ph_future* current_sql_future() noexcept;

}  // namespace profiler_hub
