// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// InlineShadow private-epoch barrier body emission. This fragment is included
/// by the shared barrier transaction after its common helper definitions.
[[nodiscard]] std::optional<std::vector<uint32_t>>
build_inline_shadow_private_epoch_barrier_cave_words(
    const ConSanBarrierSite &site, const MoiInlinePrivateEpochBarrierEmissionPlan &plan,
    const VgprSpillSequence &spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t original_barrier_word, uint64_t cave_text_offset, uint64_t return_text_offset,
    std::optional<uint16_t> call_return_sgpr, std::optional<uint16_t> selectable_vgpr_bank_mode,
    std::vector<std::string> &errors) {
  const auto load_epoch =
      instrumentation::build_private_load_b32(plan.scratch_vgpr, plan.epoch_offset, arch);
  const auto wait_load = instrumentation::build_s_wait_private_load0(arch);
  const auto increment_epoch = instrumentation::build_v_add_u32(
      plan.scratch_vgpr, scalar_positive_inline_u32(1), plan.scratch_vgpr, arch);
  const auto saturate_epoch = instrumentation::build_v_min_u32_literal(
      plan.scratch_vgpr, consan_moi_exact_shadow::max_epoch, plan.scratch_vgpr, arch);
  const auto store_epoch =
      instrumentation::build_private_store_b32(plan.scratch_vgpr, plan.epoch_offset, arch);
  const auto wait_store = instrumentation::build_s_wait_private_store0(arch);
  const auto drain_guest_vmem = instrumentation::build_s_wait_global_load0(arch);
  if (!load_epoch || !wait_load || !increment_epoch || !saturate_epoch || !store_epoch ||
      !wait_store || !drain_guest_vmem) {
    errors.emplace_back("ConSan MOI inline-shadow barrier could not encode private epoch update");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  const bool control_selectable_vgpr_bank = consan_arch_has_selectable_vgpr_bank(arch);
  const bool restore_guest_vgpr_bank =
      control_selectable_vgpr_bank && selectable_vgpr_bank_mode.value_or(0u) != 0u;
  words.reserve(
      3u + spill.save_words.size() + load_epoch->size() + store_epoch->size() +
      saturate_epoch->size() + spill.restore_words.size() +
      (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u) +
      3u + (control_selectable_vgpr_bank ? 1u : 0u) + (restore_guest_vgpr_bank ? 1u : 0u));
  words.push_back(original_barrier_word);
  // A guest load may intentionally remain outstanding across the barrier and
  // be consumed after a following waitcnt. Drain it before saving a
  // spill-backed temporary: otherwise the pending load can complete between
  // our save and restore, and the restore overwrites the guest's new value.
  words.push_back(*drain_guest_vmem);
  if (control_selectable_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(selectable_vgpr_bank_mode.value_or(0u)), 0u, arch));
  words.insert(words.end(), spill.save_words.begin(), spill.save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  words.insert(words.end(), load_epoch->begin(), load_epoch->end());
  words.push_back(*wait_load);
  words.insert(words.end(), increment_epoch->begin(), increment_epoch->end());
  words.insert(words.end(), saturate_epoch->begin(), saturate_epoch->end());
  words.insert(words.end(), store_epoch->begin(), store_epoch->end());
  words.push_back(*wait_store);
  if (!append_inline_shadow_barrier_epilogue(words, &spill, scalar_spill, selectable_vgpr_bank_mode,
                                             cave_text_offset, return_text_offset, call_return_sgpr,
                                             plan.indirect_jump, arch)) {
    errors.emplace_back("ConSan MOI inline-shadow private epoch could not encode long return at " +
                        std::to_string(site.text_offset));
    return std::nullopt;
  }
  return words;
}
