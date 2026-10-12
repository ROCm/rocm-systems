// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// The contract between main.cpp, whose second thread launches graphs, and client.cpp, which replays
// the first thread's range and tells the second thread when the replay window is open.

constexpr uint64_t kRangeId = 0xC0C0A7;

// Total passes the tool asks for, the application's own included.
constexpr uint64_t kPasses = 6;

// Dispatches the first thread submits inside its range.
constexpr uint64_t kRangeDispatches = 3;

// acc = acc*3 + add for add = 1, 2, 3 from a zeroed buffer: 0 -> 1 -> 5 -> 18.
constexpr int kExpectedResult = 18;

// Device memory the first thread holds across its range. The replay snapshot covers every tracked
// allocation on the agent, so this makes each restore take long enough for the second thread's
// launches to arrive while the window is still open.
constexpr size_t kBallastBytes = size_t{64} << 20;

// Graph launches the second thread makes once the window opens. Each launch increments its counter,
// which also took one warm-up launch before the range opened.
constexpr int kGraphLaunches   = 64;
constexpr int kExpectedCounter = 1 + kGraphLaunches;

// Set by the application before it exits without opening the range, so the tool reports a skip
// rather than a missing CLOSE. Its value is the reason.
constexpr const char* kSkipEnv = "RR_CONCURRENCY_SKIPPED";

// Handshake between the tool and the application's graph thread, exported by the tool and found by
// the application through dlsym(RTLD_DEFAULT, "rr_coord_get").
struct rr_coord_t
{
    std::atomic<int> window_open{0};       // tool -> graph thread: the first replayed pass began
    std::atomic<int> range_closed{0};      // tool -> graph thread: CLOSE was delivered
    std::atomic<int> launches_started{0};  // graph thread -> tool: its first launch is under way
};

extern "C" rr_coord_t*
rr_coord_get();
