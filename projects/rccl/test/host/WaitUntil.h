/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_WAIT_UNTIL_H_
#define RCCL_TEST_HOST_WAIT_UNTIL_H_

#include <chrono>
#include <thread>

// waitUntil -- bounded poll for something another thread will do
//
// A test that spawns a worker cannot assert on the worker's effect immediately,
// and must not sleep-and-hope either: a fixed sleep is both slower than it needs
// to be on a good run and still flaky on a loaded machine. waitUntil returns as
// soon as `pred()` holds, and returns false once `budget` has elapsed so the
// test fails with its own message instead of hanging until the suite's timeout:
//
//   ASSERT_TRUE(waitUntil([&] { return fake_.progressCalls.load() > 0; }))
//       << "the progress thread never called ginProgress";
//
// The default budget is deliberately generous: it is never waited for on a
// passing run (the predicate ends the wait), and it only has to be long enough
// that a heavily loaded CI machine does not trip it.
inline constexpr auto kWaitUntilBudget = std::chrono::milliseconds(2000);

template <typename Pred>
bool waitUntil(Pred pred, std::chrono::milliseconds budget = kWaitUntilBudget)
{
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while (!pred()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

#endif  // RCCL_TEST_HOST_WAIT_UNTIL_H_
