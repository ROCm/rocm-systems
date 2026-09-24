[[nodiscard]] OwnerEpochVgprSources materialize_owner_epoch_tail(uint16_t exclusive_tail,
                                                                 bool needs_owner) {
  std::optional<uint16_t> owner;
  if (needs_owner)
    owner = --exclusive_tail;
  return {.owner = owner, .epoch = --exclusive_tail};
}

void try_apply_direct_watchpoint_patch(std::span<const uint8_t> bytes, const Options &options,
                                       const OperatingPoint &operating_point, rj_code_arch_t arch,
                                       std::span<const Candidate> admitted,
                                       const ObjectPlan &object_plan, TransformArtifacts &result) {
  if (!options.report_buffer_address)
    return;
  const TargetProfile *target = target_profile(arch);
  if (target == nullptr) {
    result.warnings.emplace_back("ConSan probe does not support this architecture");
    return;
  }
  const auto add_unique_warning = [&result](std::string_view warning) {
    if (std::ranges::none_of(result.warnings, [warning](const std::string &existing) {
          return existing == warning;
        }))
      result.warnings.emplace_back(warning);
  };

  const ReportBufferLayout &layout = object_plan.report_layout;
  const uint32_t reserved_atomic_patches = object_plan.reserved_atomic_patch_count;
  const uint32_t max_access_record_slots = layout.watchpoint_capacity - reserved_atomic_patches;
  if (max_access_record_slots == 0) {
    if (layout.watchpoint_capacity == 0) {
      result.warnings.emplace_back(
          "ConSan probe requires room for the report header and one sampled entry");
    }
    return;
  }
  // ConSan atomic metadata wraps the relocated guest instruction in a
  // second text-growth pass. Keep dual-role access/ordering sites at the tail
  // so their one-word nested anchors remain branch-reachable in large images.
  std::vector<const Candidate *> candidates =
      order_replay_access_candidates(admitted, result.observation_plan(), options.track_atomics);
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan probe found no native LDS or likely group flat load/store candidate");
    return;
  }
  const size_t max_access_patch_count =
      options.max_patches_is_expert_limit
          ? std::min<size_t>(options.max_patches - reserved_atomic_patches, candidates.size())
          : candidates.size();
  if (max_access_patch_count == 0)
    return;

  uint32_t filter_candidate_count = 0;
  uint32_t filter_range_count = 0;
  for (uint32_t i = 0; i < candidates.size(); ++i) {
    if (i % options.sample_stride == options.sample_offset) {
      ++filter_candidate_count;
      const auto &ranges = candidates[i]->site().ranges;
      if (ranges.size() <= std::numeric_limits<uint32_t>::max() - filter_range_count)
        filter_range_count += static_cast<uint32_t>(ranges.size());
    }
  }
  uint32_t window_bank_count = 1;
  if (filter_range_count != 0) {
    const uint32_t available_per_range = max_access_record_slots / filter_range_count;
    const uint32_t requested_banks = options.watchpoint_banks == 0 ? 8u : options.watchpoint_banks;
    while (window_bank_count < requested_banks && window_bank_count * 2u <= available_per_range)
      window_bank_count *= 2u;
  }

  struct PlannedPatch : PlannedProbeResources {
    const Candidate *candidate = nullptr;
    OperatingPoint point;
    uint32_t access_range_count = 0;
    uint64_t workgroup_descriptor_file_offset = 0;
    AccessEmissionPlan emission;
    std::optional<RuntimeWorkgroupGatePlan> runtime_fast_gate;
    std::optional<WorkitemOwnerDerivationPlan> owner_derivation;
    uint32_t probe_guest_instruction_offset = 0u;
    uint32_t guest_instruction_word_count = 0u;
    bool spill_overlaps_guest_operands = false;
    bool wrap_embedded_guest_vgpr_bank = false;
    std::optional<uint16_t> wave_sched_mode;
  };
  std::vector<PlannedPatch> planned_patches;
  planned_patches.reserve(candidates.size());
  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  uint32_t candidate_index = 0;
  uint32_t next_record_index = 0;
  bool capacity_limited = false;
  bool patch_limited = false;
  SpillManagers spill_managers;
  PrivateStateLayoutCache private_layout_cache;
  for (const Candidate *candidate_ptr : candidates) {
    const uint32_t current_sample_index = candidate_index++;
    if (current_sample_index % options.sample_stride != options.sample_offset)
      continue;
    if (planned_patches.size() == max_access_patch_count) {
      patch_limited = true;
      break;
    }

    const Candidate &candidate = *candidate_ptr;
    const auto &access_ranges = candidate.site().ranges;
    if (access_ranges.empty())
      continue;
    const uint32_t range_count = static_cast<uint32_t>(access_ranges.size());
    const uint32_t required_slots = range_count * window_bank_count;
    if (required_slots > max_access_record_slots - next_record_index) {
      capacity_limited = true;
      continue;
    }
    const CandidateResourcePlan *candidate_resource_plan =
        resource_plan_for_candidate(result.resource_plans, candidate);
    if (candidate_resource_plan == nullptr)
      continue;
    const auto owner_descriptors =
        result.program_inventory.kernel_descriptors(candidate_resource_plan->owner_kernel_ids);
    if (!owner_descriptors)
      continue;
    OperatingPoint candidate_point = operating_point;
    if (!apply_transient_sgpr_assignment(options, candidate_point, result.operating_point,
                                         *owner_descriptors) ||
        !apply_persistent_vgpr_assignment(candidate_point, result.operating_point,
                                          *owner_descriptors)) {
      result.warnings.emplace_back(
          "ConSan probe lost its owner-local persistent or transient assignment");
      continue;
    }
    const bool planned_spill_backed_operand_recovery =
        std::ranges::any_of(candidate_resource_plan->alternatives, [](const auto &alternative) {
          return alternative.kind == ResourcePlanAlternativeKind::SpillBackedOperandRecovery &&
                 alternative.outcome == ResourcePlanAlternativeOutcome::Selected;
        });
    const AccessResourceFacts access_resource_facts =
        resolve_access_resource_facts(candidate_point, candidate, arch);
    const uint16_t expected_scratch_count = direct_scratch_count(options, access_resource_facts);
    const auto resources = resolve_scratch(result.resource_plans, result.program_inventory,
                                           candidate, candidate_point, expected_scratch_count);
    if (!resources) {
      continue;
    }
    const std::optional<uint64_t> workgroup_descriptor =
        common_record_owner_descriptor(bytes, *resources, arch, result.warnings);
    if (!workgroup_descriptor)
      continue;
    std::optional<PrivateStateLayout> private_layout;
    std::optional<WorkitemOwnerDerivationPlan> owner_derivation;
    if (candidate_point.automatic_private_epoch) {
      const std::optional<uint64_t> cache_descriptor =
          resources->owner_descriptor_file_offsets.size() == 1u
              ? std::optional{resources->owner_descriptor_file_offsets.front()}
              : std::nullopt;
      private_layout = private_layout_cache.resolve(
          cache_descriptor, result.program_inventory, *resources, arch, result.warnings,
          {.owner = options.owner_source == OwnerSource::WorkitemId, .exact_workgroup = true});
      if (private_layout && private_layout->owner_offset) {
        const auto owner_shift =
            common_workitem_owner_shift(bytes, *resources, arch, result.warnings);
        if (owner_shift) {
          owner_derivation = WorkitemOwnerDerivationPlan{
              .entry_workitem_x_private_offset = private_layout->owner_offset,
              .wave_size_shift = *owner_shift,
          };
        }
      }
      if (!private_layout || (private_layout->owner_offset && !owner_derivation))
        continue;
    }
    auto probe = plan_probe_resources(result.program_inventory, *resources, options, options,
                                      candidate_point, spill_managers, arch, private_layout,
                                      candidate_point.has_compact_scalar_spill(), result.warnings);
    if (!probe)
      continue;
    auto &spill = probe->spill;
    bool spill_overlaps_guest_operands = false;
    bool spill_backed_operand_recovery = false;
    if (spill && planned_spill_backed_operand_recovery &&
        access_resource_facts.supports_native_lds_spill_recovery) {
      std::vector<std::string> ignored_overlap_errors;
      spill_overlaps_guest_operands = reject_candidate_scratch_range_overlap(
          candidate, resources->base, resources->count, ignored_overlap_errors);
      spill_backed_operand_recovery = spill_overlaps_guest_operands;
    } else if (spill &&
               access_can_emit_spill_over_guest_operands(access_resource_facts, candidate)) {
      std::vector<std::string> ignored_overlap_errors;
      spill_overlaps_guest_operands = reject_candidate_scratch_range_overlap(
          candidate, resources->base, resources->count, ignored_overlap_errors);
    }
    std::vector<std::string> candidate_errors;
    OwnerEpochVgprSources owner_epoch_vgprs =
        owner_epoch_vgpr_sources(candidate_point.owner_epoch_vgprs);
    if (private_layout || candidate_point.persistent_sgprs.complete())
      owner_epoch_vgprs =
          materialize_owner_epoch_tail(static_cast<uint16_t>(resources->base + resources->count),
                                       !private_layout || private_layout->owner_offset.has_value());
    const bool has_private_workgroup_sources =
        private_layout && private_layout->exact_workgroup_offsets.complete();
    const auto workgroup_sources = exact_entry_workgroup_sources(
        candidate_point, private_layout ? &private_layout->exact_workgroup_offsets : nullptr);
    if (!workgroup_sources) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }
    const std::optional<RuntimeWorkgroupGatePlan> eligible_runtime_fast_gate =
        !candidate_point.has_compact_scalar_spill() && !has_private_workgroup_sources &&
                candidate_point.exec_save_sgpr
            ? plan_runtime_workgroup_gate({.exec_save_sgpr = *candidate_point.exec_save_sgpr,
                                           .sample_stride = options.runtime_sample_stride,
                                           .sample_offset = options.runtime_sample_offset,
                                           .dispatch_id_sgpr = candidate_point.dispatch_sgpr.base(),
                                           .literal_dispatch_id = options.report_dispatch_id},
                                          *workgroup_sources)
            : std::nullopt;
    const bool runtime_workgroup_gate_in_body =
        options.runtime_sample_stride > 1u && !eligible_runtime_fast_gate;
    const auto prior_patch = std::find_if(
        planned_patches.rbegin(), planned_patches.rend(), [&](const PlannedPatch &planned) {
          return planned.workgroup_descriptor_file_offset == *workgroup_descriptor;
        });
    const std::optional<uint32_t> prior_first_record_index =
        prior_patch == planned_patches.rend()
            ? std::nullopt
            : std::optional<uint32_t>(prior_patch->emission.first_record_index);
    const uint32_t prior_access_range_count =
        prior_patch == planned_patches.rend() ? 0u : prior_patch->access_range_count;
    AccessEmissionPlan emission{
        .check = options.device_conflict_check,
        .record_publication_sequence = layout.publication_event_capacity != 0u,
        .uniform_lds_address = candidate.site().uniform_lds_address,
        .supercollider_delay_mode = options.supercollider_delay_mode,
        .delay_count = options.supercollider_delay_nops,
        .delay_variable_source = options.supercollider_delay_var_ssrc,
        .supercollider_report_buffer_address = *options.report_buffer_address,
        .report_generation = options.report_generation,
        .runtime_sample_stride = options.runtime_sample_stride,
        .runtime_sample_offset = options.runtime_sample_offset,
        .cell_selection = options.cell_selector(),
        .exec_save_sgpr = candidate_point.exec_save_sgpr,
        .automatic_private_epoch = candidate_point.automatic_private_epoch,
        .persistent_sgprs = candidate_point.persistent_sgprs,
        .dispatch_id = detail::bound_dispatch_id_sources({candidate_point, options}),
        .workgroup_sources = *workgroup_sources,
        .owner_epoch_vgprs = owner_epoch_vgprs,
        .scratch_vgpr = resources->base,
        .base_scratch_vgpr_count = static_cast<uint16_t>(
            (options.device_conflict_check ? 7u : 5u) +
            (options.runtime_sample_stride > 1u || options.cell_selector().stride > 1u ||
                     window_bank_count > 1u
                 ? 1u
                 : 0u)),
        .scratch_vgpr_count = expected_scratch_count,
        .supports_native_lds_spill_recovery =
            access_resource_facts.supports_native_lds_spill_recovery,
        .first_record_index = next_record_index,
        .prior_first_record_index = prior_first_record_index,
        .prior_access_range_count = prior_access_range_count,
        .window_bank_count = window_bank_count,
        .causal_windows_offset = layout.causal_windows_offset,
        .watchpoints_offset = layout.watchpoints_offset,
        .pending_acquires_offset = layout.pending_acquires_offset,
        .pending_acquire_owner_bank_count = pending_acquire_owner_bank_count(
            layout.pending_acquire_capacity, layout.causal_window_capacity),
        .spill_overlaps_guest_operands = spill_overlaps_guest_operands,
        .spill_backed_operand_recovery = spill_backed_operand_recovery,
        .private_epoch_offset =
            private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        .owner_derivation = owner_derivation,
        .runtime_workgroup_gate_in_body = runtime_workgroup_gate_in_body,
    };
    uint32_t probe_guest_instruction_offset = 0;
    uint32_t guest_instruction_word_count = 0;
    auto words = build_direct_watchpoint_words(
        bytes, candidate, *workgroup_descriptor, emission, spill ? &*spill : nullptr, arch,
        candidate_errors, &probe_guest_instruction_offset, &guest_instruction_word_count);
    if (!words) {
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const bool wrap_embedded_guest_vgpr_bank =
        target->has_selectable_vgpr_bank && candidate.incoming_vgpr_bank_mode.value_or(0u) != 0u;

    PlannedPatch planned;
    static_cast<PlannedProbeResources &>(planned) = std::move(*probe);
    planned.candidate = candidate_ptr;
    planned.point = std::move(candidate_point);
    planned.access_range_count = range_count;
    planned.workgroup_descriptor_file_offset = *workgroup_descriptor;
    planned.emission = std::move(emission);
    if (eligible_runtime_fast_gate)
      planned.runtime_fast_gate = eligible_runtime_fast_gate;
    if (runtime_workgroup_gate_in_body) {
      add_unique_warning("ConSan runtime workgroup selection uses the spill-safe body gate");
    }
    planned.owner_derivation = owner_derivation;
    planned.probe_guest_instruction_offset = probe_guest_instruction_offset;
    planned.guest_instruction_word_count = guest_instruction_word_count;
    planned.spill_overlaps_guest_operands = spill_overlaps_guest_operands;
    planned.wrap_embedded_guest_vgpr_bank = wrap_embedded_guest_vgpr_bank;
    if (const ProgramContainer *container =
            result.program_inventory.container(candidate.site().container)) {
      planned.wave_sched_mode = wave_sched_mode_at(
          arch, active_bytes, container->text_file_offset, container->entry_text_offset,
          container->text_file_offset + candidate.anchor());
    }
    planned_patches.push_back(std::move(planned));
    next_record_index += required_slots;
    if (next_record_index == max_access_record_slots)
      break;
  }

  if (planned_patches.empty()) {
    if (capacity_limited) {
      result.warnings.emplace_back(
          "ConSan probe report capacity cannot retain every range of a selected "
          "load/store site");
    } else if (options.sample_stride != 1 || options.sample_offset != 0) {
      result.warnings.emplace_back(
          "ConSan probe selected no candidate after applying the sample filter");
    }
    return;
  }

  if ((capacity_limited || patch_limited || next_record_index == max_access_record_slots) &&
      filter_candidate_count > planned_patches.size()) {
    result.warnings.emplace_back(
        "ConSan probe limited sampled patches to " + std::to_string(planned_patches.size()) +
        " of " + std::to_string(filter_candidate_count) +
        " sample-filter-selected candidates by report capacity or max-patch budget");
  }

  DescriptorVgprRequirements descriptor_requirements;
  DescriptorSgprRequirements scalar_requirements;
  DescriptorPrivateRequirements private_requirements;
  for (const PlannedPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    note_sgpr_requirements(scalar_requirements, planned_patch.resources, options, options,
                           planned_patch.point, arch);
    note_probe_private_requirements(private_requirements, planned_patch);
  }

  std::vector<TextFragment> fragments;
  fragments.reserve(planned_patches.size());
  for (const PlannedPatch &planned_patch : planned_patches) {
    const Candidate &candidate = *planned_patch.candidate;
    uint32_t probe_guest_offset = 0u;
    uint32_t guest_word_count = 0u;
    auto probe_words = build_direct_watchpoint_words(
        bytes, candidate, planned_patch.workgroup_descriptor_file_offset, planned_patch.emission,
        planned_patch.spill ? &*planned_patch.spill : nullptr, arch, result.errors,
        &probe_guest_offset, &guest_word_count);
    if (!probe_words)
      return;
    std::vector<uint32_t> entry_prefix_words;
    std::optional<uint32_t> runtime_gate_branch;
    if (planned_patch.runtime_fast_gate) {
      auto gate = build_runtime_workgroup_gate_prefix(
          *planned_patch.runtime_fast_gate, planned_patch.emission.workgroup_sources, arch);
      if (!gate) {
        result.errors.emplace_back("ConSan probe could not encode its runtime workgroup gate");
        return;
      }
      runtime_gate_branch = gate->bypass_branch_word;
      entry_prefix_words = std::move(gate->words);
    }
    std::optional<uint32_t> empty_exec_branch;
    if (planned_patch.scalar_spill) {
      empty_exec_branch = static_cast<uint32_t>(entry_prefix_words.size());
      const auto guard = instrumentation::build_s_cbranch_execz(0, arch);
      if (!guard) {
        result.errors.emplace_back("ConSan probe could not encode its empty-wave guard");
        return;
      }
      entry_prefix_words.push_back(*guard);
    }

    AppendedBodyOptions body_options(arch);
    body_options.scalar_save_words =
        planned_patch.scalar_spill
            ? std::span<const uint32_t>(planned_patch.scalar_spill->save_words)
            : std::span<const uint32_t>{};
    body_options.scalar_restore_words =
        planned_patch.scalar_spill
            ? std::span<const uint32_t>(planned_patch.scalar_spill->restore_words)
            : std::span<const uint32_t>{};
    body_options.incoming_vgpr_bank_mode = candidate.incoming_vgpr_bank_mode;
    body_options.incoming_wave_sched_mode = planned_patch.wave_sched_mode;
    body_options.entry_prefix_words = entry_prefix_words;
    body_options.probe_guest_instruction_offset = probe_guest_offset;
    body_options.guest_instruction_word_count = guest_word_count;
    uint32_t body_guest_offset = 0u;
    body_options.body_guest_instruction_offset = &body_guest_offset;
    body_options.wrap_embedded_guest_vgpr_bank = planned_patch.wrap_embedded_guest_vgpr_bank;
    if (access_requires_high_bank_address_capture(candidate, arch)) {
      body_options.high_bank_address_source = candidate.site().lowering.form->address_vgpr;
      body_options.high_bank_address_destination = static_cast<uint16_t>(
          planned_patch.emission.scratch_vgpr + planned_patch.emission.base_scratch_vgpr_count);
    }
    auto body =
        assemble_appended_body({.spill = planned_patch.spill ? &*planned_patch.spill : nullptr,
                                .displaced_tail_words = {},
                                .body_size = 0u,
                                .guest_instruction_size = candidate.size()},
                               *probe_words, "sampled probe", result.errors, body_options);
    if (!body)
      return;

    PatchInfo patch;
    patch.kind = PatchKind::TrampolineWatchpointStore;
    patch.anchor_offset = candidate.anchor();
    patch.original_size = candidate.size();
    note_probe_patch_info(patch, planned_patch);
    auto access = make_access_attribution(result.observation_plan(), candidate);
    if (!access) {
      result.errors.emplace_back("ConSan probe produced an invalid intent-bound lowering");
      return;
    }
    auto mapping = make_access_runtime_mapping(
        std::move(*access), candidate.site().kind, planned_patch.emission.first_record_index,
        planned_patch.access_range_count, window_bank_count, patch);
    if (!mapping) {
      result.errors.emplace_back("ConSan probe lost its runtime attribution");
      return;
    }
    auto fragment = make_around_text_fragment(
        std::move(*body), body_guest_offset, candidate.size(), candidate.intent_ids,
        std::move(*mapping), std::move(patch), result.errors, "ConSan probe",
        guest_word_count * static_cast<uint32_t>(sizeof(uint32_t)));
    if (!fragment)
      return;
    if (runtime_gate_branch) {
      fragment->branch_fixups.push_back(
          {.word_index = *runtime_gate_branch, .target = TextFragmentBranchTarget::Bypass});
    }
    if (empty_exec_branch) {
      fragment->branch_fixups.push_back(
          {.word_index = *empty_exec_branch, .target = TextFragmentBranchTarget::Bypass});
    }
    fragments.push_back(std::move(*fragment));
  }

  const DescriptorLdsRequirements no_lds;
  (void)stage_text_rewrites(
      code_object, arch,
      {descriptor_requirements, scalar_requirements, private_requirements, no_lds},
      descriptor_mutation_policy(&options, arch), "ConSan probe", std::move(fragments), result);
}
