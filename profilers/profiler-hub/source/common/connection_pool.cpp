// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "connection_pool.hpp"

#include "debug.hpp"

#include <utility>

namespace profiler_hub::common
{

connection_pool::lease::lease(lease&& other) noexcept
: m_pool{ std::exchange(other.m_pool, nullptr) }
, m_connection{ std::exchange(other.m_connection, nullptr) }
{}

connection_pool::lease&
connection_pool::lease::operator=(lease&& other) noexcept
{
    if(this != &other)
    {
        if(m_pool != nullptr)
        {
            m_pool->release(m_connection);
        }
        m_pool       = std::exchange(other.m_pool, nullptr);
        m_connection = std::exchange(other.m_connection, nullptr);
    }
    return *this;
}

connection_pool::lease::~lease()
{
    if(m_pool != nullptr)
    {
        m_pool->release(m_connection);
    }
}

connection_pool::connection_pool(std::string_view file_path, size_t num_connections)
{
    m_connections.reserve(num_connections);
    for(size_t i = 0; i < num_connections; ++i)
    {
        m_connections.push_back(std::make_unique<connection>(file_path));
        m_free.push_back(m_connections.back().get());
    }
}

connection_pool::lease
connection_pool::acquire()
{
    LOG_DEBUG("connection_pool: requesting a connection");

    std::unique_lock lock{ m_mutex };
    m_cv.wait(lock, [this] { return !m_free.empty(); });

    connection* conn = m_free.front();
    m_free.pop_front();

    LOG_DEBUG("connection_pool: acquired connection {} ({} left free)",
              fmt::ptr(conn),
              m_free.size());
    return lease{ *this, *conn };
}

void
connection_pool::release(connection* conn)
{
    if(conn == nullptr)
    {
        return;
    }
    {
        std::scoped_lock lock{ m_mutex };
        m_free.push_back(conn);
    }
    LOG_DEBUG("connection_pool: released connection {}", fmt::ptr(conn));
    m_cv.notify_one();
}

}  // namespace profiler_hub::common
