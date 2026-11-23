// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/agent.hpp"
#include "core/state.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <rocprofiler-sdk/version.h>

namespace rocprofsys::pmc
{
std::atomic<state::process::State>&
get_state();

void
setup();

void
config();

void
sample();

void
shutdown();

void
post_process();

void set_state(state::process::State);

void
pause();

void
flush_pending_pause();

void
postfork_child_cleanup();

void
postfork_parent_reinit();

void
prefork_lock_sampler();

void
postfork_parent_unlock_sampler();

void
postfork_child_reset_sampler_lock();

#if ROCPROFILER_VERSION >= 600
/**
 * Creates the GPU perf-counter collector of the current session. The provider behind it
 * is created on the first call and reused afterwards, because the SDK cannot destroy
 * the contexts it creates.
 */
void
register_gpu_perf_counter_source(const std::vector<std::shared_ptr<agent>>& agent_list);

/**
 * Whether @p context_handle is an SDK context of the GPU perf-counter provider. The
 * collector starts these itself, so an attach must not start them.
 */
[[nodiscard]] bool
owns_gpu_perf_counter_context(std::uint64_t context_handle);
#endif

}  // namespace rocprofsys::pmc
