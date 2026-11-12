// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "../sql_cancel.hpp"
#include "thread_pool.hpp"

#include <tuple>
#include <vector>

namespace profiler_hub::common
{

/**
 * Helpers that give a caller a hand with work it also does itself. The caller never
 * waits for a helper to start: on destruction helpers that did not start yet are
 * skipped and running ones are waited for, so using it inside a task of the same pool
 * cannot deadlock, whatever the size of the pool.
 */
class helper_group
{
public:
    explicit helper_group(thread_pool& workers) noexcept
    : m_workers{ workers }
    {}

    ~helper_group()
    {
        for(const auto& handle : m_handles)
        {
            std::ignore = handle.cancel();
            handle.wait();
        }
    }

    helper_group(const helper_group&)            = delete;
    helper_group& operator=(const helper_group&) = delete;
    helper_group(helper_group&&)                 = delete;
    helper_group& operator=(helper_group&&)      = delete;

    void add(thread_pool::task_fn helper)
    {
        ph_future* future = profiler_hub::current_sql_future();
        m_handles.push_back(m_workers.submit(
            [helper = std::move(helper), future](const std::stop_token& token) {
                const profiler_hub::sql_cancel_scope cancel{ future };
                helper(token);
            }));
    }

private:
    thread_pool&                          m_workers;
    std::vector<thread_pool::task_handle> m_handles;
};

}  // namespace profiler_hub::common
