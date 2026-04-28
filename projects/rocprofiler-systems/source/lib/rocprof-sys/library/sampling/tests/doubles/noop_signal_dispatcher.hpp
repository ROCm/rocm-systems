// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Thread-safe no-op signal dispatcher for stress tests.
// Always succeeds; no shared state — safe to call concurrently from K threads.
// Use this instead of recording_signal_dispatcher in multi-threaded test scenarios.
//
// Some POSIX systems define sigmask(sig) as a 1-arg macro via <signal.h>.
#ifdef sigmask
#    undef sigmask
#endif

namespace rocprofsys::sampling::test
{

struct noop_signal_dispatcher
{
    int sigmask(int /*how*/, void const* /*set*/, void* /*oldset*/) noexcept { return 0; }
};

}  // namespace rocprofsys::sampling::test
