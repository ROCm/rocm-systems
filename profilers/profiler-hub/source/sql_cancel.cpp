// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "sql_cancel.hpp"

#include "ph_future.hpp"

namespace
{
thread_local ph_future* g_current_sql_future = nullptr;
}

namespace profiler_hub
{

sql_cancel_scope::sql_cancel_scope(ph_future* future) noexcept
: m_previous{ g_current_sql_future }
, m_active{ future != nullptr }
{
    if(m_active)
    {
        g_current_sql_future = future;
    }
}

sql_cancel_scope::~sql_cancel_scope()
{
    if(m_active)
    {
        g_current_sql_future = m_previous;
    }
}

running_sql::running_sql(void (*interrupt)(void*) noexcept, void* arg)
: m_future{ g_current_sql_future }
, m_arg{ arg }
{
    if(m_future == nullptr || interrupt == nullptr)
    {
        return;
    }
    m_future->bind_sql(interrupt, arg);
    m_bound = true;
}

running_sql::~running_sql()
{
    if(m_bound)
    {
        m_future->unbind_sql(m_arg);
    }
}

ph_future*
current_sql_future() noexcept
{
    return g_current_sql_future;
}

void
throw_if_cancelled()
{
    if(g_current_sql_future != nullptr &&
       g_current_sql_future->stop_token().stop_requested())
    {
        throw operation_cancelled{};
    }
}

}  // namespace profiler_hub
