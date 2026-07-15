// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <utility>

namespace roctx_recordfn::detail
{

// Runs an action when it leaves scope unless dismiss() is called first. The
// action runs at most once and never throws.
template<typename Fn>
class ScopeGuard
{
public:
    explicit ScopeGuard(Fn fn)
        : fn_(std::move(fn))
    {
    }

    ~ScopeGuard()
    {
        if (active_)
        {
            try
            {
                fn_();
            }
            catch (...)
            {
            }
        }
    }

    void dismiss() { active_ = false; }

    ScopeGuard(const ScopeGuard&)            = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
    ScopeGuard(ScopeGuard&&)                 = delete;
    ScopeGuard& operator=(ScopeGuard&&)      = delete;

private:
    Fn   fn_;
    bool active_ = true;
};

template<typename Fn>
ScopeGuard<Fn> make_scope_guard(Fn fn)
{
    return ScopeGuard<Fn>(std::move(fn));
}

}  // namespace roctx_recordfn::detail
