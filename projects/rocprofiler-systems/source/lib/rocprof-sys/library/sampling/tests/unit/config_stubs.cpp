// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Stub implementations of rocprofsys config functions for the standalone
// sampling unit-test binary. These return deterministic values sufficient
// for testing signal-set logic without the full rocprof-sys config system.
//
// Per-TID filter stubs (g_cputime_tids, g_realtime_tids):
//   Empty set (default) = all threads receive the signal.
//   Non-empty set = only listed TIDs receive the signal.
//   Tests override these via the set_stub_cputime_tids() / set_stub_realtime_tids()
//   helpers declared at the bottom of this file.

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

// Per-TID filter state — controllable by unit tests.
// Protected by nothing (single-threaded test context assumed).
static std::set<int64_t> g_cputime_tids{};   // empty = all threads
static std::set<int64_t> g_realtime_tids{};  // empty = all threads

std::set<int64_t>
get_sampling_cputime_tids()
{
    return g_cputime_tids;
}

std::set<int64_t>
get_sampling_realtime_tids()
{
    return g_realtime_tids;
}

// get_sampling_signals() applies per-TID filters when the respective TID set is
// non-empty.  An empty TID set means "all threads" (no filter applied).
// This mirrors the FIXED production behaviour that code-writer will implement in
// core/config.cpp's get_sampling_signals(int64_t tid).
std::set<int>
get_sampling_signals(int64_t tid)
{
    std::set<int> sigs;

    // Cputime: include if no filter or tid is in the allowed set.
    if(g_cputime_tids.empty() || g_cputime_tids.count(tid) > 0)
        sigs.insert(get_sampling_cputime_signal());

    // Realtime: include if no filter or tid is in the allowed set.
    if(g_realtime_tids.empty() || g_realtime_tids.count(tid) > 0)
        sigs.insert(get_sampling_realtime_signal());

    return sigs;
}

bool
get_use_causal()
{
    return false;
}

}  // namespace config
}  // namespace rocprofsys

// Test-only helpers — set the per-TID filter for the stub.
// Declared in a separate anonymous namespace-free scope so test TUs can include
// "unit/config_stubs_test_api.hpp" without conflicting with the config namespace.
void
test_stub_set_cputime_tids(std::set<int64_t> tids)
{
    rocprofsys::config::g_cputime_tids = std::move(tids);
}

void
test_stub_set_realtime_tids(std::set<int64_t> tids)
{
    rocprofsys::config::g_realtime_tids = std::move(tids);
}

void
test_stub_clear_tid_filters()
{
    rocprofsys::config::g_cputime_tids.clear();
    rocprofsys::config::g_realtime_tids.clear();
}
