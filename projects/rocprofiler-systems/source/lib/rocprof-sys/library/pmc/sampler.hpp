// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/state.hpp"
#include "library/pmc/device_providers/rocprofiler_sdk/provider.hpp"

#include <atomic>
#include <cstdint>
#include <vector>

namespace rocprofsys::pmc
{
std::atomic<State>&
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

void set_state(State);

void
pause();

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

void
register_gpu_perf_counter_source(
    uint64_t                                                            context_handle,
    const std::vector<device_providers::rocprofiler_sdk::agent_handle>& agent_handles);

}  // namespace rocprofsys::pmc
