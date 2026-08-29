// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_relay_target_ops.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <algorithm>
#include <limits>

namespace rocjitsu {

std::vector<uint64_t> sc_relay_reservoir_slot_offsets(uint64_t anchor_offset,
                                                      uint32_t original_size) {
  const uint64_t payload_begin = anchor_offset + kScRelayEntryWords * sizeof(uint32_t);
  const uint64_t payload_end =
      anchor_offset + original_size - kScRelayTailRestoreWords * sizeof(uint32_t);
  std::vector<uint64_t> slots;
  if (payload_end <= payload_begin)
    return slots;
  for (uint64_t offset = payload_begin; offset < payload_end;
       offset += kScRelayReservoirSlotStrideWords * sizeof(uint32_t))
    slots.push_back(offset);
  const uint64_t dense_edge_bytes = kScRelayReservoirDenseEdgeWords * sizeof(uint32_t);
  for (uint64_t offset = payload_begin;
       offset < std::min(payload_end, payload_begin + dense_edge_bytes); offset += sizeof(uint32_t))
    slots.push_back(offset);
  const uint64_t dense_tail_begin = payload_end > dense_edge_bytes
                                        ? std::max(payload_begin, payload_end - dense_edge_bytes)
                                        : payload_begin;
  for (uint64_t offset = dense_tail_begin; offset < payload_end; offset += sizeof(uint32_t))
    slots.push_back(offset);
  std::ranges::sort(slots);
  slots.erase(std::ranges::unique(slots).begin(), slots.end());
  return slots;
}

std::vector<uint64_t> sc_relay_reservoir_all_slot_offsets(uint64_t anchor_offset,
                                                          uint32_t original_size) {
  const uint64_t payload_begin = anchor_offset + kScRelayEntryWords * sizeof(uint32_t);
  const uint64_t payload_end =
      anchor_offset + original_size - kScRelayTailRestoreWords * sizeof(uint32_t);
  std::vector<uint64_t> slots;
  if (payload_end <= payload_begin)
    return slots;
  slots.reserve(static_cast<size_t>((payload_end - payload_begin) / sizeof(uint32_t)));
  for (uint64_t offset = payload_begin; offset < payload_end; offset += sizeof(uint32_t))
    slots.push_back(offset);
  return slots;
}

bool append_lds_check_trap_indirect_jump(std::vector<uint32_t> &words, uint64_t words_text_offset,
                                         uint64_t target_text_offset, uint16_t pc_sgpr,
                                         uint16_t saved_scc_sgpr,
                                         const ConSanTargetProfile &target) {
  const rj_code_arch_t arch = target.arch;
  const auto save_scc = instrumentation::build_s_cselect_b32(
      saved_scc_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
  if (!save_scc)
    return false;
  words.push_back(*save_scc);
  const uint64_t getpc_text_offset =
      words_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  words.push_back(build_s_getpc_b64(pc_sgpr, arch));
  const uint64_t pc_after_getpc = getpc_text_offset + sizeof(uint32_t);
  if (target_text_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      pc_after_getpc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  const int64_t delta =
      static_cast<int64_t>(target_text_offset) - static_cast<int64_t>(pc_after_getpc);
  if (!append_pc_delta_builder(words, arch, pc_sgpr, delta, /*minimum_words=*/3u,
                               /*prefer_literal64=*/
                               target.direct_call_form == ConSanDirectCallForm::SCallI64))
    return false;
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(saved_scc_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_scc)
    return false;
  words.push_back(*restore_scc);
  words.push_back(build_s_setpc_b64(pc_sgpr, arch));
  return true;
}

bool append_lds_check_trap_indirect_return(std::vector<uint32_t> &words, uint64_t words_text_offset,
                                           uint64_t target_text_offset, uint16_t pc_sgpr,
                                           uint16_t captured_scc_sgpr,
                                           const ConSanTargetProfile &target) {
  const rj_code_arch_t arch = target.arch;
  words.push_back(build_s_nop(0, arch));
  const uint64_t getpc_text_offset =
      words_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  words.push_back(build_s_getpc_b64(pc_sgpr, arch));
  const uint64_t pc_after_getpc = getpc_text_offset + sizeof(uint32_t);
  if (target_text_offset > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) ||
      pc_after_getpc > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return false;
  const int64_t delta =
      static_cast<int64_t>(target_text_offset) - static_cast<int64_t>(pc_after_getpc);
  if (!append_pc_delta_builder(words, arch, pc_sgpr, delta, /*minimum_words=*/3u,
                               /*prefer_literal64=*/
                               target.direct_call_form == ConSanDirectCallForm::SCallI64))
    return false;
  const auto restore_scc =
      instrumentation::build_s_cmp_lg_u32(captured_scc_sgpr, scalar_positive_inline_u32(0), arch);
  if (!restore_scc)
    return false;
  words.push_back(*restore_scc);
  words.push_back(build_s_setpc_b64(pc_sgpr, arch));
  return true;
}

} // namespace rocjitsu
