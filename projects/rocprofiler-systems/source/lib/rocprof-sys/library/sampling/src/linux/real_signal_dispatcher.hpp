// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// real_signal_dispatcher: wraps pthread_sigmask for the signal_dispatcher policy concept.
// No main-lib deps — POSIX only.
//
// NFR-PORT-3: lives under src/linux/ — not under include/sampling/.

#include <csignal>
#include <pthread.h>

// POSIX defines sigmask(sig) as a 1-arg macro; undefine so our method named
// 'sigmask' is not subject to macro expansion.
#ifdef sigmask
#    undef sigmask
#endif

namespace rocprofsys::sampling
{

class real_signal_dispatcher
{
public:
    int sigmask(int how, void const* set, void* oldset) noexcept
    {
        return ::pthread_sigmask(how, static_cast<sigset_t const*>(set),
                                 static_cast<sigset_t*>(oldset));
    }
};

}  // namespace rocprofsys::sampling
