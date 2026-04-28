// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstdint>

namespace rocprofsys::sampling::test
{

struct fake_clock
{
    static void advance_ns(uint64_t delta) noexcept { m_current_ns += delta; }
    static void reset(uint64_t start_ns = 0) noexcept { m_current_ns = start_ns; }

    [[nodiscard]] uint64_t now_ns() const noexcept { return m_current_ns; }

    [[nodiscard]] std::chrono::steady_clock::time_point now_steady() const noexcept
    {
        return std::chrono::steady_clock::time_point{ std::chrono::nanoseconds{
            m_current_ns } };
    }

    static uint64_t m_current_ns;
};

inline uint64_t fake_clock::m_current_ns = 0;

}  // namespace rocprofsys::sampling::test
