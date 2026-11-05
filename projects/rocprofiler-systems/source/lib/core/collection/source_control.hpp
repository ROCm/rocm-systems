// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/record.hpp"
#include "core/collection/source_interface.hpp"

#include <memory>
#include <mutex>
#include <utility>

namespace rocprofsys::collection
{
enum class source_state
{
    created,
    running,
    paused,
    stopped,
    failed,
};

struct source_control
{
    source_control(std::unique_ptr<source_interface> source, source_id src_id)
    : src(std::move(source))
    , id(src_id)
    {}

    std::unique_ptr<source_interface> src;
    source_id                         id;
    source_state                      state     = source_state::created;
    bool                              described = false;
    std::mutex                        mutex;
};
}  // namespace rocprofsys::collection
