// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/trigger.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_set>

namespace rocprofsys::control
{
class session;
}

namespace rocprofsys::control::triggers
{
/// ROCtx region-filter and user-pause trigger.
///
/// Lifetime contract: @p sess (a bare reference, not owned) must outlive
/// this object; the destructor calls session::detach() on it, so the
/// session itself must not have been destroyed first.
///
/// Thread-safety: on_range_start()/on_range_stop()/on_pause()/on_resume()
/// are called from arbitrary rocprofiler-sdk callback threads (not
/// necessarily the same thread across calls) and are safe to call
/// concurrently with each other - state is guarded by m_mutex plus a set
/// of atomics for the hot-path read side (should_write_markers()).
class roctx : public trigger
{
public:
    roctx(session& sess, std::string_view trace_regions);
    ~roctx() override;

    [[nodiscard]] std::string_view name() const noexcept override { return "roctx"; }
    [[nodiscard]] vote             initial_vote() const noexcept override;

    void on_range_start(std::uint64_t range_id, const char* message);
    void on_range_stop(std::uint64_t range_id);
    void on_pause();
    void on_resume();

    [[nodiscard]] bool filter_active() const noexcept { return !m_trace_regions.empty(); }

    /// Marker-write gate: paused always suppresses writes; otherwise write
    /// iff no filter is configured or a target region is currently open.
    [[nodiscard]] bool should_write_markers() const noexcept
    {
        return m_should_write.load(std::memory_order_relaxed);
    }

private:
    session&                           m_session;
    std::set<std::string, std::less<>> m_trace_regions;
    std::unordered_set<std::uint64_t>   m_active_range_ids;
    std::atomic<bool>                   m_in_region{ false };
    std::atomic<bool>                   m_user_paused{ false };
    std::atomic<bool>                   m_should_write{ true };
    std::mutex                          m_mutex;

    [[nodiscard]] vote compute_vote() const noexcept;
    [[nodiscard]] bool compute_should_write() const noexcept;
    void               refresh_state();
};
}  // namespace rocprofsys::control::triggers
