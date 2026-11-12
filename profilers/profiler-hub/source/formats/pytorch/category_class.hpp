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
    bool gpu{};          // device work: goes on a stream track
    bool launch_site{};  // a runtime or driver call that can launch device work

    std::optional<reader_types::event_type_t> event_type;

    // One event name in this category that has a type of its own.
    std::string_view                          named_event{};
    std::optional<reader_types::event_type_t> named_type{};
};

// What a trace's category string means; unknown categories are plain host events.
category_class_t
classify_category(std::string_view category);

// The event type of an event of @p cls called @p name; a category with no type of its own
// is served as regions.
reader_types::event_type_t
event_type_of(const category_class_t& cls, std::string_view name);
}  // namespace profiler_hub::pytorch
