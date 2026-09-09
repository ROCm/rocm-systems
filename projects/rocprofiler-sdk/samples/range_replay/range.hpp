// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

// The contract between the application and the tools: what range the application opens, and what
// its own execution of that range produces. Shared so a client's expectations cannot drift from
// what main.cpp actually does.
//
// Included by main.cpp, which is compiled as HIP and does not link the SDK, so this header must
// stay free of both HIP and rocprofiler dependencies.

#pragma once

#include <cstdint>

// The id main.cpp passes to rocprofiler_range_replay_begin. Echoed back in every callback for the
// range, so a client can tell its own range from anyone else's.
constexpr uint64_t kRangeId = 0xABCD01;

// Dispatches main.cpp submits inside the range, on the range's own queue.
constexpr uint64_t kRangeDispatches = 3;

// What main.cpp's chain of dispatches produces from a zeroed buffer: 0 -> 1 -> 6 -> 21. Each
// dispatch reads what its predecessor wrote, so this pins the recording's order as well as its
// contents.
constexpr int kExpectedResult = 21;
