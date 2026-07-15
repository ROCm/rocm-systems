// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "trigger.hpp"

#include <string>

namespace rocprofsys::control
{
/// Value snapshot of a trigger's contribution to the resolved state.
/// Captured at attach time so the session never holds a pointer to the
/// trigger itself; only the data needed for resolution. Owns its name
/// (rather than a view into trigger::name()'s result) so a vote_entry
/// can safely outlive the trigger that produced it.
struct vote_entry
{
    std::string name;
    vote        current_vote;
};
}  // namespace rocprofsys::control
