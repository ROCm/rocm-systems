// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "connection.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace profiler_hub::common
{

/**
 * @brief Fixed-size pool of connection instances (Object Pool pattern),
 *        independent of thread_pool -- acquiring a connection blocks the
 *        calling thread directly, it never submits work anywhere.
 */
class connection_pool
{
public:
    connection_pool(std::string_view file_path, size_t num_connections);
    ~connection_pool() = default;

    connection_pool(const connection_pool&)            = delete;
    connection_pool& operator=(const connection_pool&) = delete;
    connection_pool(connection_pool&&)                 = delete;
    connection_pool& operator=(connection_pool&&)      = delete;

    /** @brief RAII handle to a leased connection; releases it back to the pool on
     * destruction. */
    class lease
    {
    public:
        lease(lease&& other) noexcept;
        lease& operator=(lease&& other) noexcept;
        ~lease();

        lease(const lease&)            = delete;
        lease& operator=(const lease&) = delete;

        [[nodiscard]] connection& operator*() const noexcept { return *m_connection; }
        [[nodiscard]] connection* operator->() const noexcept { return m_connection; }

    private:
        friend class connection_pool;
        lease(connection_pool& pool, connection& conn) noexcept
        : m_pool{ &pool }
        , m_connection{ &conn }
        {}

        connection_pool* m_pool;
        connection*      m_connection;
    };

    /** @brief Blocks until a connection is free, then leases it to the caller. */
    [[nodiscard]] lease acquire();

    /**
     * @brief Acquires a connection and runs @p fn with it, on the calling
     *        thread. Deliberately does not touch thread_pool -- safe to
     *        call even from inside an already-running thread_pool task.
     */
    template <typename Fn>
    [[nodiscard]] auto run_sync(Fn&& fn) -> std::invoke_result_t<Fn&, connection&>
    {
        auto held = acquire();
        return std::forward<Fn>(fn)(*held);
    }

private:
    void release(connection* conn);

    // Populates the connections' shared catalog: a short sequential prefix
    // (string_list/nodes/processes/threads, each depending on the last),
    // then the remaining independent categories fanned out across the
    // pool's own connections via scoped jthreads -- deliberately not using
    // thread_pool (see class doc: connection_pool stays independent of it).
    void build_catalog(reader_catalog_t& catalog);

    std::vector<std::unique_ptr<connection>> m_connections;
    std::deque<connection*>                  m_free;
    std::mutex                               m_mutex;
    std::condition_variable                  m_cv;
};

}  // namespace profiler_hub::common
