// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_CODE_PATCH_CONSAN_RELAY_TARGET_OPS_H
#define ROCJITSU_CODE_PATCH_CONSAN_RELAY_TARGET_OPS_H

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"

#include <cstdint>
#include <vector>

namespace rocjitsu {

struct ConSanTargetProfile;

inline constexpr uint64_t kLdsCheckTrapIndirectJumpWords = 7u;
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
