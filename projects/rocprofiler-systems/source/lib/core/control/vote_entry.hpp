// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "trigger.hpp"

#include <string_view>

namespace rocprofsys::control
{
/// Value snapshot of a trigger's contribution to the resolved state.
/// Captured at attach time so the session never holds a pointer to the
/// trigger itself; only the data needed for resolution.
struct vote_entry
{
    std::string_view name;
    vote             current_vote;
};
}  // namespace rocprofsys::control
