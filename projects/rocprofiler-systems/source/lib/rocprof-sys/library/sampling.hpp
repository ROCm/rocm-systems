// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/timemory.hpp"
#include "library/components/backtrace.hpp"
#include "library/components/backtrace_metrics.hpp"
#include "library/components/backtrace_timestamp.hpp"
#include "library/components/callchain.hpp"
#include "library/thread_data.hpp"

#include <timemory/macros/language.hpp>
#include <timemory/variadic/types.hpp>

#include <cstdint>
#include <memory>
#include <set>
#include <type_traits>

namespace rocprofsys
{
namespace sampling
{
unique_ptr_t<std::set<int>>&
get_signal_types(std::int64_t _tid);

std::set<int>
setup();

/// Deactivates sampling for the calling thread. In a child process, releases only the
/// calling thread's sampler and returns an empty set; otherwise returns the configured
/// signal types. Every current caller discards the return value.
std::set<int>
shutdown();

void
block_samples();

void
unblock_samples();

void block_signals(std::set<int> = {});

void unblock_signals(std::set<int> = {});

void
post_process();

void
postfork_parent_reinit();

void
postfork_child_cleanup();

/// Releases ownership of every inherited sampler slot without running destructors.
/// Fork-child only: the pre-fork threads no longer exist in the child. Calling this from
/// a live multi-threaded process clears sampler slots belonging to threads still inside
/// configure(), which can then observe a null sampler.
/// @note Exceptions must not escape the pthread_atfork() child handler; failures
/// terminate.
void
postfork_child_release_samplers() noexcept;

void
prefork_lock_pmc_sampler();

void
postfork_parent_unlock_pmc_sampler();

void
postfork_child_reset_pmc_sampler_lock();

void
pause();

void
resume();

}  // namespace sampling
}  // namespace rocprofsys
