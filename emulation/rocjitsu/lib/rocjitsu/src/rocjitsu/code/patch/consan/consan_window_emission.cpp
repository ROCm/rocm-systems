// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_window_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_memory_emission.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu::consan::detail {

bool append_causal_window_validation(std::vector<uint32_t> &words, InstructionSequence &sequence,
                                     const CausalWindowValidationRequest &request) {
  const auto narrow_current = [&]() -> bool {
    const auto narrow = instrumentation::build_s_and_saveexec_b64(request.temporary_exec_sgpr,
                                                                  kAmdGpuVccLo, request.arch);
    return sequence.emit(narrow) &&
           sequence.emit_branch(request.mismatch_label, InstructionSequence::BranchKind::SccZero);
  };
  const auto narrow_equal_literal = [&](uint32_t offset, uint32_t literal) -> bool {
    const auto mov =
        instrumentation::build_v_mov_b32_literal(request.expected_vgpr, literal, request.arch);
    const auto compare = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(request.expected_vgpr), request.value_vgpr, request.arch);
    return detail::append_load_u32_vgpr_at_offset(words, request.address_vgpr, offset,
                                                  request.value_vgpr, request.arch) &&
           sequence.emit_all(mov, compare) && narrow_current();
  };
  const auto narrow_equal_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) -> bool {
    const auto compare = instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr),
                                                                 request.value_vgpr, request.arch);
    return detail::append_load_u32_vgpr_at_offset(words, request.address_vgpr, offset,
                                                  request.value_vgpr, request.arch) &&
           sequence.emit(compare) && narrow_current();
  };
  const auto narrow_equal_dispatch_id = [&](uint32_t offset, bool high_word) -> bool {
    return detail::append_load_u32_vgpr_at_offset(words, request.address_vgpr, offset,
                                                  request.value_vgpr, request.arch) &&
           detail::append_compare_report_dispatch_id_word(words, request.dispatch_id,
                                                          request.value_vgpr, request.expected_vgpr,
                                                          high_word, request.arch) &&
           narrow_current();
  };
  const auto narrow_equal_workgroup = [&](uint32_t offset, const WorkgroupSource &source) -> bool {
    if (!source.has_value())
      return narrow_equal_literal(offset, 0u);
    return detail::append_workgroup_source_value(words, source, request.expected_vgpr,
                                                 request.arch) &&
           narrow_equal_vgpr(offset, request.expected_vgpr);
  };

  return narrow_equal_literal(offsetof(CausalWindow, generation),
                              static_cast<uint32_t>(request.generation)) &&
         narrow_equal_literal(offsetof(CausalWindow, generation) + 4u,
                              static_cast<uint32_t>(request.generation >> 32u)) &&
         narrow_equal_dispatch_id(offsetof(CausalWindow, dispatch_id),
                                  /*high_word=*/false) &&
         narrow_equal_dispatch_id(offsetof(CausalWindow, dispatch_id) + 4u,
                                  /*high_word=*/true) &&
         narrow_equal_workgroup(offsetof(CausalWindow, workgroup_x), request.workgroup_sources.x) &&
         narrow_equal_workgroup(offsetof(CausalWindow, workgroup_y), request.workgroup_sources.y) &&
         narrow_equal_workgroup(offsetof(CausalWindow, workgroup_z), request.workgroup_sources.z) &&
         narrow_equal_vgpr(offsetof(CausalWindow, epoch), request.epoch_vgpr) &&
         narrow_equal_vgpr(offsetof(CausalWindow, first_entry), request.first_entry_vgpr) &&
         narrow_equal_literal(offsetof(CausalWindow, entry_count), 1u) &&
         narrow_equal_literal(offsetof(CausalWindow, publication_state),
                              static_cast<uint32_t>(CausalPublicationState::Ready)) &&
         narrow_equal_workgroup(offsetof(CausalWindow, cluster_workgroup_id),
                                request.workgroup_sources.cluster_workgroup_id);
}

} // namespace rocjitsu::consan::detail
