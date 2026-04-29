// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// NOTE: This header is in src/ (not include/sampling/) because it references
// sigset_t which is a POSIX type. It is only included by Linux implementation files.

#include <csignal>
#include <set>
#include <string>

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

// Comma-separated representation of a signal set for log output.
[[nodiscard]] inline std::string
join_with_comma(std::set<int> const& signals)
{
    std::string out;
    bool        first = true;
    for(int sig : signals)
    {
        if(!first) out += ", ";
        out += std::to_string(sig);
        first = false;
    }
    return out;
}

}  // namespace rocprofsys::sampling
