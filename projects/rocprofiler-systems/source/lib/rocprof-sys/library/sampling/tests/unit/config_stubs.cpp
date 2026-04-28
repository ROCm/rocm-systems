// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Stub implementations of rocprofsys config functions for the standalone
// sampling unit-test binary. These return deterministic values sufficient
// for testing signal-set logic without the full rocprof-sys config system.

#include "sampling/src/sampling_config_fwd.hpp"

#include <csignal>
#include <set>

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

// Minimal stubs so the linker satisfies declarations in sampling_config_fwd.hpp.
// The real implementations live in core/config.cpp and read from the config system.
// Unit tests that need to verify TID-filter behaviour must link against the full
// library (integration tests), not this stub binary.
std::set<int64_t>
get_sampling_cputime_tids()
{
    return {};
}

std::set<int64_t>
get_sampling_realtime_tids()
{
    return {};
}

}  // namespace config
}  // namespace rocprofsys
