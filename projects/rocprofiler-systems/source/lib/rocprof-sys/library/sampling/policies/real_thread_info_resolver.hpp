// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Production ThreadInfoResolverT: wraps thread_info::get() singleton.
// Only included from default_policies.hpp (production TU).

#include "library/thread_info.hpp"
#include "sampling/thread_info_data.hpp"

#include <cstdint>
#include <optional>

namespace rocprofsys::sampling
{

class real_thread_info_resolver
{
public:
    [[nodiscard]] std::optional<thread_info_data> resolve(int64_t tid) const
    {
        auto const& info = thread_info::get(tid, SequentTID);
        if(!info)
        {
            auto const& alt = thread_info::get(tid, InternalTID);
            if(!alt) return std::nullopt;
            return thread_info_data{
                static_cast<std::size_t>(alt->index_data->system_value),
                static_cast<std::size_t>(alt->index_data->sequent_value),
                static_cast<uint64_t>(alt->get_start()),
                static_cast<uint64_t>(alt->get_stop())
            };
        }
        return thread_info_data{ static_cast<std::size_t>(info->index_data->system_value),
                                 static_cast<std::size_t>(
                                     info->index_data->sequent_value),
                                 static_cast<uint64_t>(info->get_start()),
                                 static_cast<uint64_t>(info->get_stop()) };
    }
};

}  // namespace rocprofsys::sampling
