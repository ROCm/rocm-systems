// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Test-only API for controlling config_stubs.cpp per-TID filter state.
// Include this from test TUs that need to override g_cputime_tids / g_realtime_tids.

#pragma once

#include <cstdint>
#include <set>

// Set the cputime TID filter. Empty = all threads (no filter).
void
test_stub_set_cputime_tids(std::set<int64_t> tids);

// Set the realtime TID filter. Empty = all threads (no filter).
void
test_stub_set_realtime_tids(std::set<int64_t> tids);

// Reset both filters to empty (all threads).
void
test_stub_clear_tid_filters();
