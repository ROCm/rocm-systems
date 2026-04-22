// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/state.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
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
register_gpu_perf_counter_source(const std::vector<std::shared_ptr<agent>>& agent_list);

}  // namespace rocprofsys::pmc
