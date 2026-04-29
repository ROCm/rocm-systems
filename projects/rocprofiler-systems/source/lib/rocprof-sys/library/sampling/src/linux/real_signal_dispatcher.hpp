// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// real_signal_dispatcher: wraps pthread_sigmask for the signal_dispatcher policy concept.
// No main-lib deps — POSIX only.
//
// NFR-PORT-3: lives under src/linux/ — not under include/sampling/.

#include "sampling/platform_traits.hpp"

#include <csignal>
#include <pthread.h>

namespace rocprofsys::sampling
{

class real_signal_dispatcher
{
public:
    int apply_sigmask(int how, signal_set_handle set,
                      signal_set_mutable_handle oldset) noexcept
    {
        return ::pthread_sigmask(how, static_cast<sigset_t const*>(set),
                                 static_cast<sigset_t*>(oldset));
    }
};

}  // namespace rocprofsys::sampling
