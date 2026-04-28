// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Stub implementations of rocprofsys config functions for the standalone
// sampling unit-test binary. These return deterministic values sufficient
// for testing signal-set logic without the full rocprof-sys config system.

#include "sampling/src/sampling_config_fwd.hpp"

#include <csignal>

namespace rocprofsys
{
inline namespace config
{

int
get_sampling_realtime_signal()
{
    return SIGRTMIN + 1;
}

int
get_sampling_cputime_signal()
{
    return SIGRTMIN + 2;
}

std::set<int>
get_sampling_signals(int64_t)
{
    return { get_sampling_realtime_signal(), get_sampling_cputime_signal() };
}

bool
get_use_causal()
{
    return false;
}

}  // namespace config
}  // namespace rocprofsys
