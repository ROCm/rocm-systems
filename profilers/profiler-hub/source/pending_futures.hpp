// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <mutex>
#include <unordered_set>

struct ph_future;

namespace profiler_hub
{

/**
 * The futures whose operation was submitted but has not ended yet. A destroyed pool
 * drops queued tasks without running them, so whatever is still here when this is
 * destroyed, after the pool, ends with PH_FUTURE_CANCELLED.
 */
class pending_futures
{
public:
    pending_futures() = default;
    ~pending_futures();

    pending_futures(const pending_futures&)            = delete;
    pending_futures& operator=(const pending_futures&) = delete;
    pending_futures(pending_futures&&)                 = delete;
    pending_futures& operator=(pending_futures&&)      = delete;

    void add(std::shared_ptr<ph_future> future);
    void remove(const std::shared_ptr<ph_future>& future);

private:
    std::mutex                                     m_mutex;
    std::unordered_set<std::shared_ptr<ph_future>> m_futures;
};

}  // namespace profiler_hub
