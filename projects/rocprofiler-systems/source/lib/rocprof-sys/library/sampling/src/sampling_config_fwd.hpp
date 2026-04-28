// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Forward declarations for rocprofsys configuration functions used by the
// sampling subsystem. Avoids pulling in the full core/config.hpp (which requires
// generated headers not available in standalone test builds).
//
// When core/config.hpp has already been included (sentinel defined), these
// declarations are skipped to avoid default-arg and return-type conflicts.

#include <cstdint>
#include <set>

#ifndef ROCPROFSYS_CORE_CONFIG_HPP_INCLUDED

namespace rocprofsys
{
inline namespace config
{

int
get_sampling_realtime_signal();
int
get_sampling_cputime_signal();
int
get_sampling_overflow_signal();
double
get_sampling_realtime_freq();
double
get_sampling_cputime_freq();
double
get_sampling_realtime_delay();
double
get_sampling_cputime_delay();
double
get_sampling_overflow_freq();
double
get_sampling_duration();
std::set<int>
get_sampling_signals(int64_t _tid = 0);
bool
get_use_causal();

// Per-signal-set per-TID filters (R-3 fix surface).
// Empty set means "all threads" (no filter).
std::set<int64_t>
get_sampling_cputime_tids();
std::set<int64_t>
get_sampling_realtime_tids();

}  // namespace config
}  // namespace rocprofsys

#endif  // ROCPROFSYS_CORE_CONFIG_HPP_INCLUDED
