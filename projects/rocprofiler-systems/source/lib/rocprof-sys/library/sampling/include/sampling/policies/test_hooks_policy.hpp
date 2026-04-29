// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// TestHooksPolicy concept — injection point for test-only state overrides.
//
// Production builds default to noop_test_hooks (constexpr-false overrides).
// Tests instantiate sampling_service<..., recording_test_hooks> to flip
// override flags without exposing _for_test setters on the public surface.
//
// No virtual functions; no friend; no compile flag.

namespace rocprofsys::sampling
{

// noop_test_hooks: production default. All overrides are constexpr-false so
// the compiler folds the consulting branches at every call site.
struct noop_test_hooks
{
    static constexpr bool override_duration_disabled() noexcept { return false; }
    static constexpr bool override_causal_mode() noexcept { return false; }
    static constexpr bool override_child_process() noexcept { return false; }
};

// recording_test_hooks: test-only specialization. Holds mutable override flags
// + setters; sampling_service consults override_X() in addition to its own
// production state.
struct recording_test_hooks
{
    bool duration_disabled_override = false;
    bool causal_mode_override       = false;
    bool child_process_override     = false;

    bool override_duration_disabled() const noexcept
    {
        return duration_disabled_override;
    }
    bool override_causal_mode() const noexcept { return causal_mode_override; }
    bool override_child_process() const noexcept { return child_process_override; }

    void set_duration_disabled(bool v) noexcept { duration_disabled_override = v; }
    void set_causal_mode(bool v) noexcept { causal_mode_override = v; }
    void set_child_process(bool v) noexcept { child_process_override = v; }
};

}  // namespace rocprofsys::sampling
