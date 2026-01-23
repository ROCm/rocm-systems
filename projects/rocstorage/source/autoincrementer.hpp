// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <atomic>
#include <cstddef>

namespace rocstorage
{

template <typename PrimaryKey = size_t>
struct autoincrementer
{
    explicit autoincrementer(const char* label)
    : m_label(label)
    {}

    auto get_primary_key_value() noexcept { return m_primary_key_value.fetch_add(1); }

private:
    std::atomic<PrimaryKey> m_primary_key_value{};
    const char*             m_label;
};
}  // namespace rocstorage
