// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <string_view>

namespace rocprofsys::control
{
enum class vote
{
    abstain,
    active,
    paused
};

/// Blast radius of a trigger / interest filter of a subscriber.
///
/// - global: trigger affects every subscriber; subscriber listens to global events
/// - sampling_only: trigger affects only sampling-tagged subscribers; sampling
///   subscribers list this in addition to global
///
/// Add new scopes by appending before count_; subscriber/trigger code that
/// switches on scope must be updated accordingly.
enum class scope : std::size_t
{
    global = 0,
    sampling_only,
    count_,  // sentinel: number of scopes
};

class trigger
{
public:
    virtual ~trigger() = default;

    trigger(const trigger&)            = delete;
    trigger& operator=(const trigger&) = delete;
    trigger(trigger&&)                 = delete;
    trigger& operator=(trigger&&)      = delete;

    [[nodiscard]] virtual std::string_view name() const noexcept         = 0;
    [[nodiscard]] virtual vote             initial_vote() const noexcept = 0;

    /// Trigger's blast radius. Default is global (affects all subscribers).
    /// Override to narrow the effect to a specific subscriber set.
    [[nodiscard]] virtual scope event_scope() const noexcept { return scope::global; }

protected:
    trigger() = default;
};
}  // namespace rocprofsys::control
