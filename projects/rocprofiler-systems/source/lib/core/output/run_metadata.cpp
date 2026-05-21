// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/run_metadata.hpp"

#include <cstddef>
#include <ctime>

namespace rocprofsys::output
{

namespace
{
// "YYYY-MM-DDTHH:MM:SSZ" is 20 bytes plus the null terminator; 32 is
// a comfortable round-up for the strftime destination.
inline constexpr std::size_t ISO_8601_BUFFER_BYTES = 32;
}  // namespace

run_metadata
run_metadata::capture(std::chrono::steady_clock::time_point load_baseline)
{
    run_metadata meta{};

    const auto now = std::chrono::system_clock::now();
    const auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm    utc{};
    if(::gmtime_r(&tt, &utc) != nullptr)
    {
        char buf[ISO_8601_BUFFER_BYTES];
        if(std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc) > 0)
            meta.run_label = buf;
    }

    meta.duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - load_baseline);

    return meta;
}

}  // namespace rocprofsys::output
