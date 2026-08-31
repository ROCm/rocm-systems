// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/track_registry.hpp"

namespace rocprofsys
{
namespace
{
thread_local track_registry* t_active_registry = nullptr;
}  // namespace

void
set_active_track_registry(track_registry* registry) noexcept
{
    t_active_registry = registry;
}

track_registry*
get_active_track_registry() noexcept
{
    return t_active_registry;
}
}  // namespace rocprofsys
