// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "pending_futures.hpp"

#include "ph_future.hpp"

#include <utility>

namespace profiler_hub
{

pending_futures::~pending_futures()
{
    for(const auto& future : m_futures)
    {
        future->finish(PH_FUTURE_CANCELLED, PH_RESULT_CANCELLED);
    }
}

void
pending_futures::add(std::shared_ptr<ph_future> future)
{
    const std::scoped_lock lock{ m_mutex };
    m_futures.insert(std::move(future));
}

void
pending_futures::remove(const std::shared_ptr<ph_future>& future)
{
    const std::scoped_lock lock{ m_mutex };
    m_futures.erase(future);
}

}  // namespace profiler_hub
