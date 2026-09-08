// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// InlineShadow scalar- and VGPR-epoch barrier body emission. This fragment
/// is included by the shared barrier transaction after its common helpers.
void append_inline_shadow_barrier_epilogue(std::vector<uint32_t> &words,
                                           const VgprSpillSequence *spill,
                                           const SgprSpillSequence *scalar_spill,
                                           std::optional<uint16_t> selectable_vgpr_bank_mode,
                                           rj_code_arch_t arch) {
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (consan_arch_has_selectable_vgpr_bank(arch) && selectable_vgpr_bank_mode.value_or(0u) != 0u)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*selectable_vgpr_bank_mode), arch));
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_barrier_epoch_cave_words(
    const MoiInlineBarrierEpochEmissionPlan &plan, rj_code_arch_t arch,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    uint32_t original_barrier_word, std::optional<uint16_t> selectable_vgpr_bank_mode,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset) {
  const bool publish_visible_evidence = plan.visible_evidence.has_value();
  const bool control_selectable_vgpr_bank = consan_arch_has_selectable_vgpr_bank(arch);
  const bool restore_guest_vgpr_bank =
      control_selectable_vgpr_bank && selectable_vgpr_bank_mode.value_or(0u) != 0u;
  const auto append_visible_evidence = [&](std::vector<uint32_t> &words) {
    if (!publish_visible_evidence)
      return true;
    const MoiInlineBarrierVisibleEvidencePlan &visible = *plan.visible_evidence;
    const uint16_t active_exec_sgpr = static_cast<uint16_t>(visible.exec_save_sgpr + 12u);
    const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(visible.exec_save_sgpr + 14u);
    const auto restore_exec =
        instrumentation::build_s_mov_b64(kAmdGpuExecLo, active_exec_sgpr, arch);
    if (!restore_exec || append_publish_first_active_lane_visible_evidence_if_zero(
                             words, visible.counter_address, visible.result_vgpr,
                             visible.address_vgpr, active_exec_sgpr, temporary_exec_sgpr,
                             arch) != MoiVisibleEvidencePublicationResult::Appended)
      return false;
    words.push_back(*restore_exec);
    return true;
  };
  if (plan.persistent_epoch_sgpr) {
    const MoiSpecialStateSgprs &special_state = *plan.special_state;
    const auto save_scc = instrumentation::build_s_cselect_b32(special_state.scc_save_sgpr,
                                                               scalar_positive_inline_u32(1),
                                                               scalar_positive_inline_u32(0), arch);
    const auto at_maximum = instrumentation::build_s_cmp_eq_u32(*plan.persistent_epoch_sgpr,
                                                                /*literal=*/255u, arch);
    // The compare consumes the following literal word. If the epoch is already
    // saturated, skip the two-word CDNA4-compatible increment sequence. This
    // needs no temporary SGPRs: in particular, the SCC snapshot at exec-save
    // +4 must remain intact until restore_scc.
    const auto skip_increment = instrumentation::build_s_cbranch_scc1(/*offset_dwords=*/2, arch);
    const auto restore_scc = instrumentation::build_s_cmp_lg_u32(
        special_state.scc_save_sgpr, scalar_positive_inline_u32(0), arch);
    if (!save_scc || !at_maximum || !skip_increment || !restore_scc) {
      errors.emplace_back("ConSan MOI scalar inline barrier could not encode epoch state");
      return std::nullopt;
    }
    std::vector<uint32_t> words;
    words.reserve(
        48u + (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
        (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u));
    // The guest instruction after a patched control transfer may update the
    // VGPR-bank mode as the cave begins executing.  Establish the cave's
    // low-bank invariant even when the replaced barrier itself used mode 0.
    if (control_selectable_vgpr_bank)
      words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
          static_cast<uint8_t>(selectable_vgpr_bank_mode.value_or(0u)), 0u, arch));
    if (spill)
      words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
    if (scalar_spill)
      words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
    if (publish_visible_evidence) {
      if (!append_save_moi_special_state(words, plan.special_state, arch)) {
        errors.emplace_back("ConSan MOI inline barrier could not save visible-evidence state");
        return std::nullopt;
      }
    } else {
      words.push_back(*save_scc);
    }
    if (guest_instruction_offset)
      *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    words.push_back(original_barrier_word);
    words.push_back(*at_maximum);
    words.push_back(consan_moi_exact_shadow::max_epoch);
    words.push_back(*skip_increment);
    if (!append_moi_entry_salu_write(words,
                                     build_s_add_u32(*plan.persistent_epoch_sgpr,
                                                     *plan.persistent_epoch_sgpr,
                                                     scalar_positive_inline_u32(1), arch),
                                     arch)) {
      errors.emplace_back("ConSan MOI scalar inline barrier could not encode saturating add");
      return std::nullopt;
    }
    if (publish_visible_evidence) {
      if (!append_visible_evidence(words) ||
          !append_restore_moi_special_state(words, plan.special_state, arch)) {
        errors.emplace_back("ConSan MOI inline barrier could not publish visible evidence");
        return std::nullopt;
      }
    } else {
      words.push_back(*restore_scc);
    }
    append_inline_shadow_barrier_epilogue(words, spill, scalar_spill, selectable_vgpr_bank_mode,
                                          arch);
    return words;
  }
  const auto increment_epoch = instrumentation::build_v_add_u32(
      *plan.epoch_vgpr, scalar_positive_inline_u32(1), *plan.epoch_vgpr, arch);
  const auto saturate_epoch = instrumentation::build_v_min_u32_literal(
      *plan.epoch_vgpr, consan_moi_exact_shadow::max_epoch, *plan.epoch_vgpr, arch);
  if (!increment_epoch || !saturate_epoch) {
    errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch could not encode saturating epoch add");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(
      40u + (spill ? spill->save_words.size() + spill->restore_words.size() : 0u) +
      (scalar_spill ? scalar_spill->save_words.size() + scalar_spill->restore_words.size() : 0u) +
      (control_selectable_vgpr_bank ? 1u : 0u) + (restore_guest_vgpr_bank ? 1u : 0u));
  if (guest_instruction_offset)
    *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  words.push_back(original_barrier_word);
  if (control_selectable_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(selectable_vgpr_bank_mode.value_or(0u)), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (publish_visible_evidence && !append_save_moi_special_state(words, plan.special_state, arch)) {
    errors.emplace_back("ConSan MOI inline barrier could not save visible-evidence state");
    return std::nullopt;
  }
  words.insert(words.end(), increment_epoch->begin(), increment_epoch->end());
  words.insert(words.end(), saturate_epoch->begin(), saturate_epoch->end());
  if (publish_visible_evidence &&
      (!append_visible_evidence(words) ||
       !append_restore_moi_special_state(words, plan.special_state, arch))) {
    errors.emplace_back("ConSan MOI inline barrier could not publish visible evidence");
    return std::nullopt;
  }
  append_inline_shadow_barrier_epilogue(words, spill, scalar_spill, selectable_vgpr_bank_mode,
                                        arch);
  return words;
}

[[nodiscard]] std::optional<PlannedBarrierLowering>
plan_inline_shadow_private_barrier(MoiPrivateBarrierPlanningContext context) {
  auto &[bytes, options, point, candidate, site, resource_plan, access_patch, semantics,
         spill_managers, arch, inventory, assignments, warnings] = context;
  const auto owners = inventory.kernel_descriptors(resource_plan.owner_kernel_ids);
  if (!owners) {
    warnings.emplace_back("ConSan MOI private barrier lost its kernel owner identity");
    return std::nullopt;
  }
  const ConSanMoiPrivateStateLayout &private_state_layout = *access_patch.private_state_layout;
  const uint32_t ephemeral_base = private_state_layout.ephemeral_base;
  const auto private_limit = consan_address_free_private_limit(arch);
  if (!private_limit) {
    warnings.emplace_back("ConSan MOI private epoch barrier has no address-free scratch support");
    return std::nullopt;
  }
  SpillManager manager(ephemeral_base, *private_limit);
  auto spill = build_vgpr_spill_sequence(manager, *access_patch.scratch_vgpr,
                                         /*vgpr_count=*/1, arch);
  if (!spill) {
    warnings.emplace_back("ConSan MOI private epoch barrier could not preserve its temporary VGPR");
    return std::nullopt;
  }
  const uint32_t required_private_bytes =
      std::max(access_patch.required_private_segment_size, spill->total_private_bytes);
  ConSanPatchAbiEffects patch_abi = plan_barrier_patch_abi(
      *owners, access_patch.scratch_vgpr, private_state_layout, &*spill, required_private_bytes);
  return PlannedBarrierLowering{
      .candidate = candidate,
      .site = &site,
      .body =
          MoiInlinePrivateEpochBarrierEmissionPlan{
              .scratch_vgpr = *access_patch.scratch_vgpr,
              .epoch_offset = private_state_layout.epoch_offset,
          },
      .spill = std::move(spill),
      .scalar_spill = std::nullopt,
      .patch_abi = std::move(patch_abi),
  };
}

[[nodiscard]] std::optional<MoiBarrierBodyPlan>
plan_inline_shadow_barrier_body(const ConSanOptions &options, ConSanMoiOperatingPoint &point,
                                const MoiScalarAbiPlan &scalar_abi,
                                std::optional<uint16_t> scratch_vgpr, std::span<const uint64_t>,
                                rj_code_arch_t arch, MoiOwnerAssignments,
                                std::vector<std::string> &, std::vector<std::string> &errors) {
  auto plan = plan_inline_shadow_barrier_epoch_emission(options, point, scalar_abi, scratch_vgpr,
                                                        arch, errors);
  if (!plan)
    return std::nullopt;
  return MoiBarrierBodyPlan{std::move(*plan)};
}

void try_apply_inline_shadow_barrier_patch(std::span<const uint8_t> bytes,
                                           const ConSanOptions &options,
                                           const ConSanMoiOperatingPoint &operating_point,
                                           rj_code_arch_t arch,
                                           const MoiObjectModeSemantics &semantics,
                                           ConSanTransformArtifacts &result) {
  if (!options.moi_track_barriers)
    return;
  ConSanMoiOperatingPoint barrier_point = operating_point;
  if (barrier_point.has_inline_moi_scalar_spill())
    barrier_point.moi_exec_save_sgpr.reset();
  try_apply_inline_shadow_barrier_transaction(
      bytes, options, plan_inline_shadow_barrier_body, plan_inline_shadow_private_barrier,
      barrier_point, arch, semantics, result);
}
