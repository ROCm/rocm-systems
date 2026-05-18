// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "output/run_metadata.hpp"

#include <ctime>

namespace rocprofsys::output
{

run_metadata
run_metadata::capture(std::chrono::steady_clock::time_point load_baseline)
{
    run_metadata meta{};

    const auto now = std::chrono::system_clock::now();
    const auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm    utc{};
    if(::gmtime_r(&tt, &utc) != nullptr)
    {
        char buf[32];
        if(std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &utc) > 0)
            meta.run_label = buf;
    }

    meta.duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - load_baseline);

    return meta;
}

}  // namespace rocprofsys::output
