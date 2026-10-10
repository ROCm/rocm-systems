/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_SCOPE_EXIT_H_
#define RCCL_TEST_HOST_SCOPE_EXIT_H_

#include <utility>

// ScopeExit -- run this cleanup on every exit from the scope
//
// A fatal assertion (`ASSERT_*`) returns from the test body immediately, so
// cleanup written after the assertion simply does not run -- which in a host
// test usually means a leaked allocation, a still-joinable std::thread (that
// calls std::terminate from its destructor), or a fake left pointing at a dead
// stack local, so one failing test takes the rest of the binary with it. Put
// that cleanup in a ScopeExit at the point the resource is acquired and it runs
// on the fatal path too.
//
// Usage (CTAD deduces the callable's type):
//
//   std::thread worker([&] { ... });
//   ScopeExit joinWorker([&] { stop.store(true); worker.join(); });
//   ASSERT_TRUE(waitUntil([&] { return started.load(); }));  // safe to bail here
//
// Prefer a purpose-built RAII type where one already exists (ScopedHook for a
// std::function seam, JoinThreadsOnExit for a set of worker threads); ScopeExit
// is for the one-off cleanup that does not have one.
//
// Non-copyable and non-movable so the cleanup cannot run twice; C++17 guaranteed
// copy elision still allows a factory to return one by value.
template <typename Fn>
class ScopeExit {
public:
    explicit ScopeExit(Fn fn) : fn_(std::move(fn)) {}
    ~ScopeExit() { fn_(); }

    ScopeExit(const ScopeExit&)            = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ScopeExit(ScopeExit&&)                 = delete;
    ScopeExit& operator=(ScopeExit&&)      = delete;

private:
    Fn fn_;
};

template <typename Fn>
ScopeExit(Fn) -> ScopeExit<Fn>;

#endif  // RCCL_TEST_HOST_SCOPE_EXIT_H_
