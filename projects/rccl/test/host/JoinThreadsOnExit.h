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

// Spawn worker threads; stop and join them on every scope exit, including a fatal
// ASSERT_* (destroying a joinable std::thread would std::terminate the binary).
// The stop action is a caller-supplied callback because the worker's loop is ended
// by a flag the unit under test owns; omit it for workers that finish by themselves.
// Non-copyable and non-movable so the threads have exactly one owner.
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
