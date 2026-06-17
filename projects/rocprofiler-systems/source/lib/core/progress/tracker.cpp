// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/progress/tracker.hpp"

#include <cstdint>
#include <utility>

namespace rocprofsys::progress
{

tracker::tracker(factory_t make) noexcept
: m_make(std::move(make))
{}

progress_callback
tracker::begin(std::string label, std::uint64_t total_bytes)
{
    if(!m_make) return {};
    return m_make(std::move(label), total_bytes);
}

}  // namespace rocprofsys::progress
