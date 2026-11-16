/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_SCOPE_EXIT_H_
#define RCCL_TEST_HOST_SCOPE_EXIT_H_

#include <utility>

// Run this cleanup on every exit from the scope, including a fatal ASSERT_* (which
// returns before any cleanup written after it, leaking the resource or leaving a
// fake pointing at a dead stack local). Prefer a purpose-built RAII type where one
// exists (ScopedHook, JoinThreadsOnExit); ScopeExit is for the one-off.
// Non-copyable and non-movable so the cleanup cannot run twice.
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
