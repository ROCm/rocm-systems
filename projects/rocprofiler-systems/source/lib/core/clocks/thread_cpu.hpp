// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/clocks/read.hpp"

#include <chrono>
#include <time.h>

namespace rocprofsys::clocks::thread_cpu
{
/// CPU time used so far by the calling thread (`CLOCK_THREAD_CPUTIME_ID`); zero if
/// the clock fails. Only the difference between two reads on one thread is meaningful.
/// Async-signal-safe: the sampler's signal handler calls it.
[[nodiscard]] inline std::chrono::nanoseconds
now() noexcept
{
    return detail::read(CLOCK_THREAD_CPUTIME_ID);
}
}  // namespace rocprofsys::clocks::thread_cpu
