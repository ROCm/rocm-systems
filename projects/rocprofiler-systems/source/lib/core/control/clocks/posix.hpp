// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/control/clock.hpp"

#include "logger/debug.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <mutex>
#include <thread>

namespace rocprofsys::control::clocks
{
/// Clock backed by a POSIX clock (clock_gettime / clock_nanosleep).
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
        if(clock_gettime(m_clock_id, &ts) != 0)
        {
            LOG_WARNING("clock_gettime failed for clock id {}: {}", m_clock_id,
                        std::strerror(errno));
        }
        const auto ns =
            static_cast<std::int64_t>(ts.tv_sec) * 1'000'000'000LL + ts.tv_nsec;
        return clock_time_point{ clock_duration{ ns } };
    }

    [[nodiscard]] bool sleep_until(clock_time_point deadline)
    {
        static constexpr std::int64_t chunk_ns = 1'000'000;  // 1 ms

        while(now() < deadline)
        {
            {
                std::scoped_lock const lk{ m_mutex };
                if(m_interrupted) return false;
            }

            const auto remaining_ns = (deadline - now()).count();
            if(remaining_ns <= 0) break;

            const auto this_chunk_ns = std::min(remaining_ns, chunk_ns);

#ifdef __linux__
            const auto      next_ns = now().time_since_epoch().count() + this_chunk_ns;
            struct timespec ts      = { static_cast<time_t>(next_ns / 1'000'000'000LL),
                                        static_cast<long>(next_ns % 1'000'000'000LL) };
            if(const auto rc = clock_nanosleep(m_clock_id, TIMER_ABSTIME, &ts, nullptr);
               rc != 0 && rc != EINTR)
            {
                LOG_WARNING("clock_nanosleep failed for clock id {}: {}", m_clock_id,
                            std::strerror(rc));
            }
#else
            std::this_thread::sleep_for(std::chrono::nanoseconds{ this_chunk_ns });
#endif
        }

        std::scoped_lock const lk{ m_mutex };
        return !m_interrupted;
    }

    void interrupt()
    {
        std::scoped_lock const lk{ m_mutex };
        m_interrupted = true;
    }

    void reset()
    {
        std::scoped_lock const lk{ m_mutex };
        m_interrupted = false;
    }

private:
    clockid_t  m_clock_id;
    std::mutex m_mutex;
    bool       m_interrupted{ false };
};
}  // namespace rocprofsys::control::clocks
