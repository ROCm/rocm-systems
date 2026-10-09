/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_JOIN_THREADS_ON_EXIT_H_
#define RCCL_TEST_HOST_JOIN_THREADS_ON_EXIT_H_

#include <functional>
#include <thread>
#include <utility>
#include <vector>

// JoinThreadsOnExit -- spawn worker threads, stop and join them on scope exit
//
// Destroying a still-joinable std::thread calls std::terminate(), so a test that
// spawns a worker and then trips a fatal `ASSERT_*` would abort the whole binary
// rather than fail one test. Spawning through this type means the stop-and-join
// happens on every exit path, including that one.
//
// The loop a worker runs is usually ended by a flag the unit under test owns
// rather than by the thread returning on its own, so the stop action is a
// callback supplied by the caller -- this type knows nothing about what makes
// the workers finish, only that it must be done before joining them. Pass no
// callback for workers that finish by themselves.
//
// Usage:
//
//   JoinThreadsOnExit workers([gs] { gs->stopSignal.store(true); });
//   workers.spawn([gs] { ncclGinProgress(gs, 0); });
//   ASSERT_TRUE(waitUntil([&] { return fake_.progressCalls.load() > 0; }));
//   // workers' destructor raises the stop signal and joins, even on the ASSERT.
//
// Non-copyable and non-movable so the threads have exactly one owner; C++17
// guaranteed copy elision still allows a factory that binds the stop callback to
// return one by value (`auto workers = progressWorkers(gs);`).
class JoinThreadsOnExit {
public:
    explicit JoinThreadsOnExit(std::function<void()> stop = {}) : stop_(std::move(stop)) {}
    ~JoinThreadsOnExit() { stopAndJoin(); }

    JoinThreadsOnExit(const JoinThreadsOnExit&)            = delete;
    JoinThreadsOnExit& operator=(const JoinThreadsOnExit&) = delete;
    JoinThreadsOnExit(JoinThreadsOnExit&&)                 = delete;
    JoinThreadsOnExit& operator=(JoinThreadsOnExit&&)      = delete;

    template <class Fn>
    void spawn(Fn&& fn) { threads_.emplace_back(std::forward<Fn>(fn)); }

    // Also callable before the scope ends, for a test that asserts on what the
    // workers left behind once they have all stopped. Idempotent.
    void stopAndJoin()
    {
        if (threads_.empty()) return;
        if (stop_) stop_();
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
        threads_.clear();
    }

private:
    std::function<void()>    stop_;
    std::vector<std::thread> threads_;
};

#endif  // RCCL_TEST_HOST_JOIN_THREADS_ON_EXIT_H_
