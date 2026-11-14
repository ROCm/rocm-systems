// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/agent.hpp"
#include "core/collection/source_interface.hpp"

#include <memory>
#include <vector>

namespace rocprofsys::pmc
{
/**
 * The sources (pollers) that exist at setup(): CPU always; GPU (AMD SMI) and NIC when
 * ROCPROFSYS_USE_AMD_SMI and built; hipFile when ROCPROFSYS_USE_HIPFILE and built.
 * Providers are created here; a provider that fails to start is logged once and its
 * pollers are left out.
 */
[[nodiscard]] std::vector<std::unique_ptr<collection::source_interface>>
make_default_sources();

/**
 * The GPU hardware counter source. SDK counter contexts can only be created while the
 * SDK is configured (tool_init), so this is called from there.
 * @return null when no counter is configured, the SDK is older than 6.0, or setup failed
 */
[[nodiscard]] std::unique_ptr<collection::source_interface>
make_gpu_perf_counter_source(const std::vector<std::shared_ptr<agent>>& gpu_agents);
}  // namespace rocprofsys::pmc
