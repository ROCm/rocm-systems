// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_PATCH_CONSAN_RELAY_TARGET_OPS_H
#define ROCJITSU_CODE_PATCH_CONSAN_RELAY_TARGET_OPS_H

#include <cstdint>
#include <vector>

namespace rocjitsu {

struct ConSanTargetProfile;

inline constexpr uint64_t kLdsCheckTrapIndirectJumpWords = 7u;
inline constexpr uint32_t kScRelayEntryWords = 8u;
inline constexpr uint32_t kScRelayTailRestoreWords = 1u;
inline constexpr uint32_t kScRelayAppendedOverheadWords = 9u;
inline constexpr uint32_t kScRelayReservoirSlotStrideWords = 16u;
inline constexpr uint32_t kScRelayReservoirDenseEdgeWords = 16u;
inline constexpr uint16_t kRdna4VccLo = 106u;
inline constexpr uint16_t kRdna4ExecHi = 127u;

[[nodiscard]] std::vector<uint64_t> sc_relay_reservoir_slot_offsets(uint64_t anchor_offset,
                                                                    uint32_t original_size);

[[nodiscard]] std::vector<uint64_t> sc_relay_reservoir_all_slot_offsets(uint64_t anchor_offset,
                                                                        uint32_t original_size);

[[nodiscard]] bool append_lds_check_trap_indirect_jump(std::vector<uint32_t> &words,
                                                       uint64_t words_text_offset,
                                                       uint64_t target_text_offset,
                                                       uint16_t pc_sgpr, uint16_t saved_scc_sgpr,
                                                       const ConSanTargetProfile &target);

[[nodiscard]] bool append_lds_check_trap_indirect_return(
    std::vector<uint32_t> &words, uint64_t words_text_offset, uint64_t target_text_offset,
    uint16_t pc_sgpr, uint16_t captured_scc_sgpr, const ConSanTargetProfile &target);

} // namespace rocjitsu

#endif // ROCJITSU_CODE_PATCH_CONSAN_RELAY_TARGET_OPS_H
