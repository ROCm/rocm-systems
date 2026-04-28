// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// NOTE: This header is in src/ (not include/sampling/) because it references
// sigset_t which is a POSIX type. It is only included by Linux implementation files.

#include <csignal>
// <csignal> transitively defines sigmask(sig) as a 1-arg macro on some POSIX systems.
// Undefine it so that policy types using "sigmask" as a method name are not mangled.
#ifdef sigmask
#    undef sigmask
#endif

#include <set>

namespace rocprofsys::sampling
{

// sigset_t value type constructed from a std::set<int>.
struct signal_set
{
    sigset_t sigset = {};

    explicit signal_set(std::set<int> const& signals) noexcept
    {
        sigemptyset(&sigset);
        for(int sig : signals)
            sigaddset(&sigset, sig);
    }

    explicit signal_set() noexcept { sigemptyset(&sigset); }

    [[nodiscard]] sigset_t const* get() const noexcept { return &sigset; }
    sigset_t*                     get() noexcept { return &sigset; }
};

}  // namespace rocprofsys::sampling
