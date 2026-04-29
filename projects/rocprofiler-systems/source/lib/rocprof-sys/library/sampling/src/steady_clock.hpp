// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// steady_clock: wraps std::chrono::steady_clock for the clock policy concept.
// No main-lib deps — POSIX/std only.
//
// NFR-PORT-3: lives under src/linux/ — not under include/sampling/.

#include <chrono>
#include <cstdint>

namespace rocprofsys::sampling
{

class steady_clock
{
public:
    [[nodiscard]] uint64_t now_ns() const noexcept
    {
        auto tp = std::chrono::steady_clock::now();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch())
                .count());
    }

    [[nodiscard]] std::chrono::steady_clock::time_point now_steady() const noexcept
    {
        return std::chrono::steady_clock::now();
    }
};

}  // namespace rocprofsys::sampling
