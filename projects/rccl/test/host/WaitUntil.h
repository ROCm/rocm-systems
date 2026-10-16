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

// staysFalse -- assert that another thread does NOT do something
//
// The mirror of waitUntil, for the cases it cannot express: "the worker never
// progressed this context", "the idle threads never called ginProgress". There is
// no predicate that becomes true to end the wait, so proving the negative costs a
// real window of wall time -- that is inherent, not a flaw in the helper.
//
// It exists so that cost is paid deliberately, in one place with a shared budget
// and an explanation, instead of each test growing a bare sleep_for that reads
// like the sleep-and-hope waitUntil above was written to replace:
//
//   ASSERT_TRUE(waitUntil([&] { return started.load(); }));   // positive control
//   EXPECT_TRUE(staysFalse([&] { return fake_.progressCalls.load() > 0; }))
//       << "a context that is not this thread's was progressed anyway";
//
// ALWAYS pair it with a positive control first. Without one, a worker that never
// started at all passes this trivially, and the test proves nothing.
//
// Returns true if `pred()` stayed false for the whole window. The window is short
// because, unlike waitUntil's budget, it IS paid on every passing run: these are the
// only waits in the file that make the suite slower, so the default is the smallest
// window that has proved stable rather than a generous one.
//
// It SLEEPS between polls where waitUntil yields. waitUntil's spin ends the moment the
// predicate flips, so spinning costs almost nothing; this one always runs the window out,
// and a yield loop would burn a core for its whole duration -- on a CI box running test
// binaries in parallel that is worse than the bare sleep_for this replaces.
//
// 30 ms is the smallest window these tests have run on, and length is not where the
// confidence comes from: the required positive control proves the worker is already
// inside its loop, and that loop iterates in microseconds, so a violation shows up in
// the first poll or not at all. A longer window would buy nothing but wall time.
inline constexpr auto kStaysFalseWindow = std::chrono::milliseconds(30);
inline constexpr auto kStaysFalsePollInterval = std::chrono::milliseconds(1);

template <typename Pred>
bool staysFalse(Pred pred, std::chrono::milliseconds window = kStaysFalseWindow)
{
    const auto deadline = std::chrono::steady_clock::now() + window;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return false;
        std::this_thread::sleep_for(kStaysFalsePollInterval);
    }
    return !pred();
}

#endif  // RCCL_TEST_HOST_WAIT_UNTIL_H_
