/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_WAIT_UNTIL_H_
#define RCCL_TEST_HOST_WAIT_UNTIL_H_

#include <chrono>
#include <thread>

// Bounded poll for something another thread will do: returns as soon as pred() holds,
// or false once budget elapses so the test fails with its own message rather than
// hanging. The default budget is generous because it is never waited for on a passing
// run, and only has to outlast a heavily loaded CI machine.
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
