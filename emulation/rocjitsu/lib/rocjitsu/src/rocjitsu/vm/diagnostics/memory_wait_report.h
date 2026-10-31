// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_wait_report.h
/// @brief The hazard record of a core memory-wait or XCNT site.

#pragma once

#include "rocjitsu/vm/amdgpu/memory_wait_scoreboard.h"

namespace rocjitsu {
namespace amdgpu {
class ComputeUnitCore;
class Wavefront;
} // namespace amdgpu

namespace diagnostics {

class HazardSites;

/// @brief Build and publish the record of a site that HazardSites::classify() returned as New.
/// @details Decodes the producer, the consumer and the straight-line code between them
/// with @p cu's decoder, on the thread that runs @p cu. The decode never raises a guest
/// fault and never fills the execution decode cache.
void publish_memory_wait(HazardSites &sites, amdgpu::ComputeUnitCore &cu,
                         const amdgpu::Wavefront &wf,
                         const amdgpu::MemoryWaitScoreboard::Hazard &hazard);

} // namespace diagnostics
} // namespace rocjitsu
