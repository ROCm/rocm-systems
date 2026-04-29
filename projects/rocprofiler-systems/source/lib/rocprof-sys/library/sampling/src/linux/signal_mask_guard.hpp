// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// NOTE: This header is in src/ (not include/sampling/) because it references
// sigset_t and pthread_sigmask. Only included by Linux implementation files.

#include "signal_set.hpp"

#include <csignal>
#include <set>
#include <string_view>

namespace rocprofsys::sampling
{

// RAII pthread_sigmask wrapper.
// Blocks signals on construction; unblocks (restores) on destruction.
// On pthread_sigmask failure, calls fatal_error_policy::fatal() — EC-8.
template <class FatalErrorPolicy>
class signal_mask_guard
{
public:
    signal_mask_guard(std::set<int> const& signals, int how,
                      FatalErrorPolicy& fatal) noexcept
    : fatal_(fatal)
    , active_(false)
    {
        signal_set ss(signals);
        int        err = ::pthread_sigmask(how, ss.get(), &old_);
        if(err != 0)
        {
            fatal_.fatal(__FILE__, __LINE__, "pthread_sigmask failed: errno={}", err);
        }
        active_ = true;
    }

    ~signal_mask_guard() noexcept
    {
        if(active_)
        {
            ::pthread_sigmask(SIG_SETMASK, &old_, nullptr);
        }
    }

    signal_mask_guard(signal_mask_guard const&)            = delete;
    signal_mask_guard& operator=(signal_mask_guard const&) = delete;

    void release() noexcept { active_ = false; }

private:
    FatalErrorPolicy& fatal_;
    sigset_t          old_    = {};
    bool              active_ = false;
};

}  // namespace rocprofsys::sampling
