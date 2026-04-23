// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "subscriber.hpp"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace rocprofsys::control
{
class session
{
public:
    explicit session(std::string_view trace_regions = {});
    ~session() = default;

    session(const session&)            = delete;
    session& operator=(const session&) = delete;
    session(session&&)                 = delete;
    session& operator=(session&&)      = delete;

    void shutdown();

    void subscribe(subscriber sub);

    [[nodiscard]] bool region_filter_active() const noexcept
    {
        return m_region_filter_active.load(std::memory_order_relaxed);
    }

    [[nodiscard]] bool is_active() const noexcept
    {
        return m_active.load(std::memory_order_relaxed);
    }

    void force_initial_pause();

    void handle_range_start(std::uint64_t range_id, const char* message);
    void handle_range_stop(std::uint64_t range_id);
    void handle_pause(std::uint64_t tid);
    void handle_resume(std::uint64_t tid);

private:
    std::set<std::string, std::less<>> m_trace_regions;
    std::unordered_set<std::uint64_t>  m_active_range_ids;
    std::atomic<bool>                  m_region_filter_active{ false };
    std::atomic<std::uint32_t>         m_active_region_count{ 0 };
    std::atomic<bool>                  m_user_paused{ false };
    std::atomic<bool>                  m_active{ true };

    std::vector<subscriber> m_subscribers;

    std::mutex m_region_mutex;
    std::mutex m_subscribers_mutex;

    void recompute_active();
    void notify_pause();
    void notify_resume();
};
}  // namespace rocprofsys::control
