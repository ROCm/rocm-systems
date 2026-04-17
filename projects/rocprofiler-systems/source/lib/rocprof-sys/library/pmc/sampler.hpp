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
register_sdk_pmc_source(
    uint64_t context_handle, const std::vector<uint64_t>& agent_ids,
    const std::vector<uint64_t>&                 profile_configs,
    const std::vector<size_t>&                   device_indices,
    const std::vector<std::vector<std::string>>& counter_names_per_agent,
    const std::vector<
        std::vector<device_providers::rocprofiler_sdk::counter_instance_info>>&
        instance_infos_per_agent,
    const std::vector<std::vector<device_providers::rocprofiler_sdk::counter_metadata>>&
        counter_meta_per_agent);

}  // namespace rocprofsys::pmc
