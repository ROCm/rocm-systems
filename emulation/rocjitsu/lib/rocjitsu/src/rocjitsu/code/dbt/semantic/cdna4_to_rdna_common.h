// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file semantic/cdna4_to_rdna_common.h
/// @brief Shared CDNA4-to-RDNA semantic expansion helpers.

#pragma once

#include <cstdint>
#include <vector>

namespace rocjitsu {

class Instruction;
class LivenessAnalysis;
struct LaneLayout;

/// @brief Expand CDNA4 v_lshl_add_u64 into the RDNA carry-chain sequence.
std::vector<uint32_t> expand_cdna4_v_lshl_add_u64_for_rdna(const Instruction &inst,
                                                           uint32_t host_arch, uint64_t offset,
                                                           const LivenessAnalysis &liveness,
                                                           const LaneLayout *guest_layout,
                                                           const LaneLayout *host_layout);

} // namespace rocjitsu
