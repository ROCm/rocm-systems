// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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
        m_handles.push_back(m_workers.submit(std::move(helper)));
    }

private:
    thread_pool&                          m_workers;
    std::vector<thread_pool::task_handle> m_handles;
};

}  // namespace profiler_hub::common
