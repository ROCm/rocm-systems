/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_HOST_SCOPEDHOOK_H_
#define RCCL_TEST_HOST_SCOPEDHOOK_H_

#include <functional>
#include <utility>

// ScopedHook -- RAII wrapper around a controllable seam
//
// Three jobs in one type:
//   1. Installs the test's behaviour on construction.
//   2. Counts calls automatically (.calls), so tests don't hand-roll a
//      separate `int xCalls = 0; ++xCalls` per hook.
//   3. Restores the previous behaviour on destruction. This means tests
//      don't have to inherit the fixture or rely on ResetNcclCudaFakes() to
//      avoid contaminating each other -- the hook's lifetime ends with
//      the ScopedHook local, before the captured stack locals die.
//
// Usage (CTAD picks up the signature from the hook variable):
//   ScopedHook memGet(g_hipMemGetAddressRange,
//       [&](hipDeviceptr_t* pb, std::size_t* ps, hipDeviceptr_t) {
//           if (pb) *pb = ...;
//           return hipSuccess;
//       });
//   ...
//   EXPECT_EQ(memGet.calls, 1);
//
// Non-movable + non-copyable because the installed lambda captures `this`
// to bump the counter.
template <typename FnSig>
class ScopedHook;

template <typename R, typename... Args>
class ScopedHook<R(Args...)> {
public:
    template <typename Callable>
    ScopedHook(std::function<R(Args...)>& slot, Callable fn)
        : slot_(slot), saved_(std::move(slot))
    {
        slot_ = [this, fn = std::move(fn)](Args... args) -> R {
            ++calls;
            return fn(std::forward<Args>(args)...);
        };
    }
    ~ScopedHook() { slot_ = std::move(saved_); }

    ScopedHook(const ScopedHook&)            = delete;
    ScopedHook& operator=(const ScopedHook&) = delete;
    ScopedHook(ScopedHook&&)                 = delete;
    ScopedHook& operator=(ScopedHook&&)      = delete;

    int calls = 0;
private:
    std::function<R(Args...)>& slot_;
    std::function<R(Args...)>  saved_;
};

// CTAD: deduce R(Args...) from the std::function<R(Args...)>& argument so
// call sites don't have to spell out the signature.
template <typename R, typename... Args, typename Callable>
ScopedHook(std::function<R(Args...)>&, Callable) -> ScopedHook<R(Args...)>;

#endif  // RCCL_TEST_HOST_SCOPEDHOOK_H_
