// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/cpp/reader_types.hpp"

#include <optional>
#include <string_view>

namespace profiler_hub::pytorch
{
struct category_class_t
{
    bool gpu{};           // device work: goes on a stream track
    bool in_host_tree{};  // takes part in the per-thread call tree
    bool launch_site{};   // a runtime or driver call that can launch device work

    std::optional<reader_types::event_type_t> event_type{};
};

// What a trace's category string means; unknown categories are plain host events.
category_class_t
classify_category(std::string_view category);
}  // namespace profiler_hub::pytorch
