// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Production ThreadInfoResolverT: wraps a thread_info resolution callback.
// The callback is injected at construction time — in production it wraps
// thread_info::get(); in tests it can be replaced with a fixed-data provider.

#include "sampling/thread_info_data.hpp"

#include <cstdint>
#include <functional>
#include <optional>

namespace rocprofsys::sampling
{

class real_thread_info_resolver
{
public:
    using resolve_fn = std::function<std::optional<thread_info_data>(int64_t)>;

    real_thread_info_resolver() = default;

    explicit real_thread_info_resolver(resolve_fn fn)
    : resolve_fn_(std::move(fn))
    {}

    [[nodiscard]] std::optional<thread_info_data> resolve(int64_t tid) const
    {
        if(resolve_fn_) return resolve_fn_(tid);
        return std::nullopt;
    }

private:
    resolve_fn resolve_fn_;
};

}  // namespace rocprofsys::sampling
