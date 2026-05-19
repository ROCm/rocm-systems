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
    // `scopes` is currently paused. Owned by `session` — only read or written
    // while holding `session::m_subscribers_mutex`. Mutable so
    // dispatch_for_scope() can update it while iterating subscribers by
    // const-ref. Reset by session::shutdown().
    mutable bool paused = false;
};
}  // namespace rocprofsys::control
