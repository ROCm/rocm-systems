// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/clock.hpp"

#include <cstdint>
#include <ctime>
#include <mutex>
#include <thread>

namespace rocprofsys::control::clocks
{
/// ClockPolicy backed by a POSIX clock (clock_gettime / clock_nanosleep).
///
/// sleep_until() sleeps in 1 ms chunks against the configured POSIX clock.
/// On Linux, each chunk uses clock_nanosleep(TIMER_ABSTIME) so the sleep is
/// measured against the clock (e.g. CLOCK_PROCESS_CPUTIME_ID means 1 ms of
/// process CPU time per chunk). On other platforms a wall-clock
/// std::this_thread::sleep_for(1ms) is used with clock_gettime for the
/// deadline check — adequate for delay/duration values in the 0.05-10 s range.
///
/// interrupt() sets a flag that the chunk loop checks between sleeps;
/// worst-case interrupt latency is one chunk interval (1 ms).
///
/// Satisfies the ClockPolicy concept (see core/control/clock.hpp).
class posix
{
public:
    explicit posix(clockid_t clock_id = CLOCK_REALTIME) noexcept
    : m_clock_id{ clock_id }
    {}

    ~posix() = default;

    posix(const posix&)            = delete;
    posix& operator=(const posix&) = delete;
    posix(posix&&)                 = delete;
    posix& operator=(posix&&)      = delete;

    [[nodiscard]] clock_time_point now() const noexcept
    {
        struct timespec ts = {};
        clock_gettime(m_clock_id, &ts);
        const auto ns =
            static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
        return clock_time_point{ clock_duration{ ns } };
    }

    [[nodiscard]] bool sleep_until(clock_time_point deadline)
    {
        static constexpr std::int64_t CHUNK_NS = 1'000'000;  // 1 ms

        while(now() < deadline)
        {
            {
                std::scoped_lock const lk{ m_mutex };
                if(m_interrupted) return false;
            }

            const auto remaining_ns = (deadline - now()).count();
            if(remaining_ns <= 0) break;

            const auto chunk_ns = std::min(remaining_ns, CHUNK_NS);

#ifdef __linux__
            const auto      next_ns = now().time_since_epoch().count() + chunk_ns;
            struct timespec ts      = { static_cast<time_t>(next_ns / 1'000'000'000LL),
                                        static_cast<long>(next_ns % 1'000'000'000LL) };
            clock_nanosleep(m_clock_id, TIMER_ABSTIME, &ts, nullptr);
#else
            std::this_thread::sleep_for(std::chrono::nanoseconds{ chunk_ns });
#endif
        }

        std::scoped_lock const lk{ m_mutex };
        return !m_interrupted;
    }

    void interrupt() noexcept
    {
        std::scoped_lock const lk{ m_mutex };
        m_interrupted = true;
    }

    void reset() noexcept
    {
        std::scoped_lock const lk{ m_mutex };
        m_interrupted = false;
    }

    [[nodiscard]] clockid_t clock_id() const noexcept { return m_clock_id; }

private:
    clockid_t  m_clock_id;
    std::mutex m_mutex;
    bool       m_interrupted{ false };
};
}  // namespace rocprofsys::control::clocks
