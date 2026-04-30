// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "sampling/thread_info_data.hpp"

#include <cstdint>
#include <map>
#include <optional>

namespace rocprofsys::sampling::test
{

struct mock_thread_info_resolver
{
    mock_thread_info_resolver() = default;

    template <class Fn>
    explicit mock_thread_info_resolver(Fn&&)
    {}

    std::map<int64_t, thread_info_data> data;

    std::optional<thread_info_data> resolve(int64_t tid) const
    {
        auto it = data.find(tid);
        return it != data.end() ? std::optional{ it->second } : std::nullopt;
    }
};

}  // namespace rocprofsys::sampling::test
