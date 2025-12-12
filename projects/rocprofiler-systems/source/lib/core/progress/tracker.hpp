// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/progress/callback.hpp"

#include <cstdint>
#include <functional>
#include <string>

namespace rocprofsys::progress
{

class tracker
{
public:
    using factory_t = std::function<progress_callback(std::string, std::uint64_t)>;

    explicit tracker(factory_t make) noexcept;
    ~tracker() = default;

    tracker(const tracker&)            = delete;
    tracker& operator=(const tracker&) = delete;
    tracker(tracker&&)                 = delete;
    tracker& operator=(tracker&&)      = delete;

    [[nodiscard]] progress_callback begin(std::string label, std::uint64_t total_bytes);

private:
    factory_t m_make;
};

}  // namespace rocprofsys::progress
