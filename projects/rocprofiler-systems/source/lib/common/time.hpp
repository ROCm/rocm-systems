// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <ratio>
#include <time.h>

namespace rocprofsys
{
inline namespace common
{
namespace time
{
// All timestamps placed on a rocprofiler-systems trace timeline use the same
// CLOCK_BOOTTIME domain as rocprofiler-sdk and Perfetto on Linux.
struct timeline_clock
{
    using rep        = std::int64_t;
    using period     = std::nano;
    using duration   = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<timeline_clock>;

    static constexpr bool is_steady = true;

    static time_point now() noexcept
    {
        timespec ts{};
        auto     ret = ::clock_gettime(CLOCK_BOOTTIME, &ts);
        if(ret != 0) std::abort();
        return time_point{
            duration{ static_cast<rep>(ts.tv_sec) * std::nano::den + ts.tv_nsec }
        };
    }
};

template <typename Tp = std::uint64_t>
inline Tp
timeline_ns() noexcept
{
    return static_cast<Tp>(timeline_clock::now().time_since_epoch().count());
}
}  // namespace time
}  // namespace common
}  // namespace rocprofsys
