// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Thread-safe no-op signal dispatcher for stress tests.
// Always succeeds; no shared state — safe to call concurrently from K threads.
// Use this instead of recording_signal_dispatcher in multi-threaded test scenarios.

namespace rocprofsys::sampling::test
{

struct noop_signal_dispatcher
{
    int apply_sigmask(int /*how*/, void const* /*set*/, void* /*oldset*/) noexcept
    {
        return 0;
    }
};

}  // namespace rocprofsys::sampling::test
