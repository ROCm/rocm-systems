// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Type-erased callbacks injected into sampling_service at construction.
// Separated from sampling_config (POD values) so the config struct stays
// lightweight and copyable without std::function overhead.

#include "sampling/thread_info_data.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <set>
#include <string>

namespace rocprofsys::sampling
{

struct sampling_callbacks
{
    std::function<std::set<int>(int64_t)> resolve_signals = [](int64_t) {
        return std::set<int>{};
    };

    std::function<bool(int64_t)> is_thread_eligible = [](int64_t) { return true; };

    std::function<int64_t()> get_sys_tid = []() -> int64_t { return 0; };

    std::function<std::optional<thread_info_data>(int64_t)> resolve_thread_info =
        [](int64_t) -> std::optional<thread_info_data> { return std::nullopt; };

    std::function<void()>                         register_sampling_categories = []() {};
    std::function<void(int, int, std::size_t)>    register_thread_info = [](int, int,
                                                                         std::size_t) {};
    std::function<void(std::string, std::size_t)> register_track       = [](std::string,
                                                                      std::size_t) {};

    std::function<void(void*, std::string const&, double)> configure_overflow_pe_attr =
        [](void*, std::string const&, double) {};

    std::function<void()> postfork_parent_reinit = []() {};
    std::function<void()> postfork_child_cleanup = []() {};
};

}  // namespace rocprofsys::sampling
