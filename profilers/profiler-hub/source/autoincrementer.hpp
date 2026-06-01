// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>

namespace profiler_hub
{

template <typename PrimaryKey = size_t>
struct autoincrementer
{
    explicit autoincrementer() = default;

    auto get_primary_key_value() noexcept { return m_primary_key_value.fetch_add(1); }

private:
    std::atomic<PrimaryKey> m_primary_key_value{};
};
}  // namespace profiler_hub
