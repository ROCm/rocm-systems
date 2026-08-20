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

namespace rocprofsys::control::clocks
{
/// Clock backed by a POSIX clock (clock_gettime / clock_nanosleep).
class posix
{
public:
    explicit posix(clockid_t clock_id = CLOCK_REALTIME) noexcept
    : m_clock_id{ clock_id }
    {
        struct timespec ts = {};
        if(clock_gettime(m_clock_id, &ts) != 0)
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
        struct timespec ts = {};
        if(clock_gettime(m_clock_id, &ts) != 0)
        {
            LOG_WARNING("clock_gettime failed for clock id {}: {}", m_clock_id,
                        std::system_category().message(errno));
        }
        return clock_time_point{ clock_duration{ to_nanoseconds(ts) } };
    }

    [[nodiscard]] bool sleep_until(clock_time_point deadline)
    {
        static constexpr std::int64_t chunk_ns = 1'000'000;  // 1 ms

        if(!m_usable) return false;

        for(auto current = now(); current < deadline; current = now())
        {
            {
                const std::scoped_lock lk{ m_mutex };
                if(m_interrupted) return false;
            }

            const auto remaining_ns = (deadline - current).count();
            if(remaining_ns <= 0) break;

            const auto this_chunk_ns = std::min(remaining_ns, chunk_ns);

            const auto next_ns       = current.time_since_epoch().count() + this_chunk_ns;
            const struct timespec ts = to_timespec(next_ns);
            if(const auto rc = clock_nanosleep(m_clock_id, TIMER_ABSTIME, &ts, nullptr);
               rc != 0 && rc != EINTR)
            {
                LOG_WARNING("clock_nanosleep failed for clock id {}: {}", m_clock_id,
                            std::system_category().message(rc));
            }
        }

        const std::scoped_lock lk{ m_mutex };
        return !m_interrupted;
    }

    void interrupt()
    {
        auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
        const std::scoped_lock lk{ m_mutex };
        m_interrupted = true;
    }

    void reset()
    {
        auto _thread_state_guard = state::thread::scoped(state::thread::Internal);
        const std::scoped_lock lk{ m_mutex };
        m_interrupted = false;
    }

private:
    static constexpr std::int64_t nsec_per_sec = 1'000'000'000;

    [[nodiscard]] static std::int64_t to_nanoseconds(const struct timespec& ts) noexcept
    {
        return (static_cast<std::int64_t>(ts.tv_sec) * nsec_per_sec) + ts.tv_nsec;
    }

    [[nodiscard]] static struct timespec to_timespec(std::int64_t ns) noexcept
    {
        struct timespec ts = {};
        ts.tv_sec          = static_cast<time_t>(ns / nsec_per_sec);
        ts.tv_nsec         = static_cast<long>(ns % nsec_per_sec);
        return ts;
    }

    clockid_t  m_clock_id;
    bool       m_usable = true;
    std::mutex m_mutex;
    bool       m_interrupted{ false };
};
}  // namespace rocprofsys::control::clocks
