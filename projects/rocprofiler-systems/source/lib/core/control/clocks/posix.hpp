// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/clock.hpp"
#include "core/state.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <system_error>
#include <time.h>

namespace rocprofsys::control::clocks
{
/// Clock backed by a POSIX clock (clock_gettime / clock_nanosleep).
class posix
{
public:
    explicit posix(clockid_t clock_id = CLOCK_REALTIME) noexcept  // NOLINT
    : m_clock_id{ clock_id }
    {
        struct timespec specs = {};
        if(clock_gettime(m_clock_id, &specs) != 0)
        {
            m_usable = false;
            LOG_WARNING("unusable clock id {}: {}", m_clock_id,
                        std::system_category().message(errno));
        }
    }

    ~posix() = default;

    posix(const posix&)            = delete;
    posix& operator=(const posix&) = delete;
    posix(posix&&)                 = delete;
    posix& operator=(posix&&)      = delete;

    [[nodiscard]] clock_time_point now() const noexcept
    {
        struct timespec specs = {};
        if(clock_gettime(m_clock_id, &specs) != 0)
        {
            LOG_WARNING("clock_gettime failed for clock id {}: {}", m_clock_id,
                        std::system_category().message(errno));
        }
        return clock_time_point{ clock_duration{ to_nanoseconds(specs) } };
    }

    [[nodiscard]] bool sleep_until(clock_time_point deadline)
    {
        if(!m_usable)
        {
            return false;
        }

        for(auto current = now(); current < deadline; current = now())
        {
            if(is_interrupted())
            {
                return false;
            }

            const auto remaining_ns = (deadline - current).count();
            if(remaining_ns <= 0)
            {
                break;
            }

            sleep_one_chunk(current, remaining_ns);
        }

        return !is_interrupted();
    }

    void interrupt() noexcept
    {
        const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
        const std::scoped_lock notify_lk{ m_mutex };
        m_interrupted = true;
    }

    void reset() noexcept
    {
        const auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
        const std::scoped_lock notify_lk{ m_mutex };
        m_interrupted = false;
    }

private:
    static constexpr std::int64_t nsec_per_sec = 1'000'000'000;
    static constexpr std::int64_t chunk_ns     = 1'000'000;  // 1 ms

    [[nodiscard]] bool is_interrupted()
    {
        const std::scoped_lock notify_lk{ m_mutex };
        return m_interrupted;
    }

    void sleep_one_chunk(clock_time_point current, std::int64_t remaining_ns)
    {
        const auto this_chunk_ns    = std::min(remaining_ns, chunk_ns);
        const auto next_ns          = current.time_since_epoch().count() + this_chunk_ns;
        const struct timespec specs = to_timespec(next_ns);
        if(const auto sleep_status =
               clock_nanosleep(m_clock_id, TIMER_ABSTIME, &specs, nullptr);  // NOLINT
           sleep_status != 0 && sleep_status != EINTR)
        {
            LOG_WARNING("clock_nanosleep failed for clock id {}: {}", m_clock_id,
                        std::system_category().message(sleep_status));
        }
    }

    [[nodiscard]] static std::int64_t to_nanoseconds(
        const struct timespec& specs) noexcept
    {
        return (static_cast<std::int64_t>(specs.tv_sec) * nsec_per_sec) + specs.tv_nsec;
    }

    [[nodiscard]] static struct timespec to_timespec(std::int64_t nanoseconds) noexcept
    {
        struct timespec specs = {};
        specs.tv_sec          = static_cast<time_t>(nanoseconds / nsec_per_sec);
        specs.tv_nsec         = static_cast<long>(nanoseconds % nsec_per_sec);
        return specs;
    }

    clockid_t  m_clock_id;
    bool       m_usable = true;
    std::mutex m_mutex;
    bool       m_interrupted{ false };
};
}  // namespace rocprofsys::control::clocks
