// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "connection_pool.hpp"

#include "debug.hpp"
#include "reader_catalog.hpp"

#include <array>
#include <thread>
#include <utility>
#include <vector>

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
    auto catalog = std::make_shared<reader_catalog_t>();

    m_connections.reserve(num_connections);
    for(size_t i = 0; i < num_connections; ++i)
    {
        m_connections.push_back(std::make_unique<connection>(file_path, catalog));
        m_free.push_back(m_connections.back().get());
    }

    build_catalog(*catalog);
}

void
connection_pool::build_catalog(reader_catalog_t& catalog)
{
    // Independent of each other once {string_list,nodes,processes,threads}
    // are done; kernel_symbols additionally depends on code_objects, so
    // it's run as a final serial step after this batch instead of
    // alongside it.
    const std::array<reader_t::catalog_category_t, 6> parallel_categories{
        reader_t::catalog_category_t::agents,
        reader_t::catalog_category_t::tracks,
        reader_t::catalog_category_t::code_objects,
        reader_t::catalog_category_t::streams,
        reader_t::catalog_category_t::queues,
        reader_t::catalog_category_t::pmc_infos,
    };

    // Sequential prefix -- each depends on the previous; cheap tables, not
    // worth parallelizing.
    run_sync([&](connection& conn) {
        conn.reader().build_catalog_category(reader_t::catalog_category_t::string_list,
                                             catalog);
        conn.reader().build_catalog_category(reader_t::catalog_category_t::nodes,
                                             catalog);
        conn.reader().build_catalog_category(reader_t::catalog_category_t::processes,
                                             catalog);
        conn.reader().build_catalog_category(reader_t::catalog_category_t::threads,
                                             catalog);
    });

    {
        // run_sync's blocking acquire() throttles actual concurrency to
        // the pool's connection count; a jthread per category is safe even
        // when there are more categories than connections.
        std::vector<std::jthread> workers;
        workers.reserve(parallel_categories.size());
        for(auto category : parallel_categories)
        {
            workers.emplace_back([this, category, &catalog] {
                run_sync([category, &catalog](connection& conn) {
                    conn.reader().build_catalog_category(category, catalog);
                });
            });
        }
    }  // jthreads joined here

    // Depends on code_objects, already built above.
    run_sync([&](connection& conn) {
        conn.reader().build_catalog_category(reader_t::catalog_category_t::kernel_symbols,
                                             catalog);
    });
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
