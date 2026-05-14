// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "trigger.hpp"

#include <functional>
#include <string>
#include <vector>

namespace rocprofsys::control
{
/// A consumer of pause/resume notifications. The `scopes` field selects which
/// trigger scopes this subscriber listens to. Default = {global}: respond
/// only to global-scope triggers (TRACE_DELAY/DURATION, roctx, etc.).
/// Subsystems affected by narrower scopes (e.g. sampling for SAMPLING_DURATION)
/// list those scopes in addition to global.
struct subscriber
{
    std::function<void()> on_pause;
    std::function<void()> on_resume;
    std::string           name;
    std::vector<scope>    scopes = { scope::global };

    // Net resolved paused state for this subscriber: true iff at least one of
    // `scopes` is currently paused. Mutable so session::dispatch_for_scope()
    // can update it under the subscribers mutex while iterating by const-ref.
    // Reset by session::shutdown().
    mutable bool paused = false;
};
}  // namespace rocprofsys::control
