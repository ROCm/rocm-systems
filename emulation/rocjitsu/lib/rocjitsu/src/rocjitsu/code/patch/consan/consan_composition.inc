namespace {

using ConSanDescriptorNameIndex = std::unordered_map<std::string, uint64_t>;
using ConSanDescriptorOwnerIndex = std::unordered_map<uint64_t, uint64_t>;

[[nodiscard]] uint64_t measure_descriptor_name_index(const void *context) noexcept {
  const auto &index = *static_cast<const ConSanDescriptorNameIndex *>(context);
  uint64_t bytes = byte_accounting::saturating_multiply(
      index.size(), sizeof(ConSanDescriptorNameIndex::value_type) +
                        amdgpu_code_object_detail::kAssociativeEntryBookkeepingBytes);
  for (const auto &[name, offset] : index) {
    (void)offset;
    bytes = byte_accounting::saturating_add(bytes, name.capacity() + 1u);
  }
  return bytes;
}

[[nodiscard]] uint64_t measure_descriptor_owner_index(const void *context) noexcept {
  const auto &index = *static_cast<const ConSanDescriptorOwnerIndex *>(context);
  return byte_accounting::saturating_multiply(
      index.size(), sizeof(ConSanDescriptorOwnerIndex::value_type) +
                        amdgpu_code_object_detail::kAssociativeEntryBookkeepingBytes);
}

/// Install the transaction's immutable observation product into native
/// lowering state. The fallback exists for focused lowerer tests and legacy
/// mechanism entry points; it uses the same common assembly authority and is
/// deliberately outside both engine implementations.
[[nodiscard]] bool
initialize_consan_lowering_observation(const ConSanOptions &options,
                                       ConSanTransformArtifacts &result,
                                       const ConSanLoweringObservation *prepared_observation) {
  if (prepared_observation != nullptr) {
    if (!prepared_observation->well_formed()) {
      result.errors.emplace_back("ConSan received an invalid prepared observation product");
      return false;
    }
    result.coverage_ledger = prepared_observation->initial_coverage;
    return true;
  }

  ConSanObservationProduct observation = assemble_consan_observation_product(
      result.program_inventory, static_cast<const ConSanRequest &>(options),
      static_cast<const ConSanDebugOverrides &>(options));
  const bool valid = observation.valid();
  result.coverage_ledger = std::move(observation.initial_coverage);
  result.errors.insert(result.errors.end(),
                       std::make_move_iterator(observation.diagnostics.begin()),
                       std::make_move_iterator(observation.diagnostics.end()));
  return valid;
}

[[nodiscard]] bool ranges_overlap(uint64_t first_offset, uint32_t first_size,
                                  uint64_t second_offset, uint32_t second_size) {
  return first_offset < second_offset + second_size && second_offset < first_offset + first_size;
}

[[nodiscard]] std::optional<std::vector<ConSanPerturbationPlan>>
carry_perturbation_plans_across_mutation(const ProgramInventory &pristine_inventory,
                                         const ConSanPerturbationPlanningState &planning,
                                         const ConSanTransformArtifacts &mutation_stage,
                                         std::string *error) {
  std::vector<ConSanPerturbationPlan> carried;
  carried.reserve(planning.plans.size());
  for (const ConSanPerturbationPlan &plan : planning.plans) {
    const ConSanPerturbationCandidate &candidate = plan.candidate;
    const SynchronizationInventoryView sync = pristine_inventory.sync();
    const ConSanSyncSequence *source_sequence = sync.find_sequence(candidate.sequence);
    const ConSanSyncEvent *source_anchor = sync.find_event(candidate.anchor_event);
    if (!candidate.eligible || source_sequence == nullptr || source_anchor == nullptr ||
        plan.sequence_identity.empty() || plan.container_name.empty() || plan.anchor_size == 0u) {
      *error = "selected pristine perturbation identity is incomplete or stale";
      return std::nullopt;
    }

    uint64_t translated_offset = plan.anchor_text_offset;
    bool overlaps_atomic_mutation = false;
    bool removed_cache_boundary = false;
    bool sequence_semantics_weakened = false;
    const auto dropped =
        std::ranges::find_if(mutation_stage.patches, [&](const ConSanPatchMutationProduct &patch) {
          return patch.kind == ConSanPatchKind::InlineBarrierNopRewrite &&
                 ranges_overlap(plan.anchor_text_offset, plan.anchor_size,
                                patch.anchor_offset, patch.original_size);
        });
    if (dropped != mutation_stage.patches.end()) {
      *error = "selected perturbation edge was destroyed by the barrier drop";
      return std::nullopt;
    }

    std::vector<const ConSanPatchMutationProduct *> move_sources;
    const ConSanPatchMutationProduct *move_target = nullptr;
    for (const ConSanPatchMutationProduct &patch : mutation_stage.patches) {
      if (patch.kind == ConSanPatchKind::InlineBarrierMoveSourceRewrite)
        move_sources.push_back(&patch);
      else if (patch.kind == ConSanPatchKind::InlineBarrierMoveTargetRewrite)
        move_target = &patch;
    }
    const auto selected_source =
        std::ranges::find_if(move_sources, [&](const ConSanPatchMutationProduct *patch) {
          return ranges_overlap(plan.anchor_text_offset, plan.anchor_size,
                                patch->anchor_offset, patch->original_size);
        });
    if (selected_source != move_sources.end()) {
      if (move_sources.size() != 2u || move_target == nullptr ||
          !move_target->barrier_move_direction || move_target->trampoline_size == 0u) {
        *error = "barrier move lacks one exact whole-pair ownership record";
        return std::nullopt;
      }
      const uint64_t first_source =
          std::min(move_sources[0]->anchor_offset, move_sources[1]->anchor_offset);
      const uint64_t source_delta = plan.anchor_text_offset - first_source;
      const uint64_t pair_offset =
          move_target->trampoline_offset +
          (*move_target->barrier_move_direction == ConSanBarrierMoveDirection::Later
               ? move_target->original_size
               : 0u);
      translated_offset = pair_offset + source_delta;
    }

    for (const ConSanPatchMutationProduct &patch : mutation_stage.patches) {
      if (patch.phase == ConSanPatchPhase::Mutation &&
          patch.kind == ConSanPatchKind::InlineAtomicScopeRewrite) {
        const auto event =
            std::ranges::find_if(pristine_inventory.sync().sync_events, [&](const auto &item) {
              const ConSanProgramSite *source = pristine_inventory.program_site(item.source_site);
              return source != nullptr && source->container.name == plan.container_name &&
                     source->container.is_kernel() == plan.in_kernel &&
                     item.text_offset() == patch.anchor_offset &&
                     source->size() == patch.original_size;
            });
        if (event != pristine_inventory.sync().sync_events.end() && source_sequence != nullptr &&
            std::ranges::find(source_sequence->member_event_ids,
                              pristine_inventory.sync().event_id(*event)) !=
                source_sequence->member_event_ids.end()) {
          sequence_semantics_weakened = true;
        }
      }
      if (patch.phase != ConSanPatchPhase::Mutation || patch.original_size == 0u ||
          !ranges_overlap(plan.anchor_text_offset, plan.anchor_size, patch.anchor_offset,
                          patch.original_size))
        continue;
      const bool exact_alias = plan.anchor_text_offset == patch.anchor_offset &&
                               plan.anchor_size == patch.original_size;
      const bool atomic_mutation = patch.kind == ConSanPatchKind::InlineAtomicAddressRewrite ||
                                   patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite ||
                                   patch.kind == ConSanPatchKind::InlineAtomicScopeRewrite;
      if (!atomic_mutation)
        continue;
      if (!exact_alias) {
        *error = "selected perturbation edge partially overlaps a mutation";
        return std::nullopt;
      }
      overlaps_atomic_mutation = true;
      removed_cache_boundary = patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite;
    }

    ConSanPerturbationPlan translated = plan;
    translated.anchor_text_offset = translated_offset;
    translated.source_sequence_identity = source_sequence->identity;
    translated.overlaps_atomic_mutation = overlaps_atomic_mutation;
    translated.removed_cache_boundary = removed_cache_boundary;
    translated.source_sequence_semantics_weakened = sequence_semantics_weakened;
    carried.push_back(std::move(translated));
  }
  return carried;
}

[[nodiscard]] ConSanTransformArtifacts rollback_composite_mutation(ConSanTransformArtifacts result,
                                                                   std::string reason) {
  result.outcome = ConSanTransformOutcome::Unsupported;
  result.discard_candidate_modification();
  result.mutation.fault.applied = 0;
  result.mutation.perturbation.applied = 0;
  result.warnings.emplace_back("ConSan rolled back staged fault/perturbation composition: " +
                               std::move(reason));
  return result;
}

[[nodiscard]] ConSanTransformArtifacts
lower_consan_stage(std::span<const uint8_t> code_object_bytes, const ConSanOptions &options,
                   const ConSanMoiOperatingPoint &initial_operating_point,
                   std::span<const ConSanPerturbationPlan> carried_plans,
                   ConSanPerturbationPlanningState *inspected_perturbation,
                   const ConSanPreappliedMutationLayout &preapplied_mutation, bool apply_fault_plan,
                   bool require_exactly_one_applied_fault, ConSanLoweringExtent extent,
                   const ConSanLoweringObservation *observation);

[[nodiscard]] ConSanTransformArtifacts
try_patch_consan_impl(std::span<const uint8_t> code_object_bytes, const ConSanOptions &options,
                      const ConSanMoiOperatingPoint &initial_operating_point,
                      ConSanPerturbationPlanningState *inspected_perturbation,
                      const ConSanPreappliedMutationLayout &preapplied_mutation,
                      ConSanLoweringExtent extent, const ConSanLoweringObservation *observation) {
  const bool has_fault = options.has_fault_mutation();
  const bool has_perturbation = options.sc_perturb_kind != ConSanPerturbationKind::None;
  if (!has_fault || options.fault_dry_run) {
    return lower_consan_stage(code_object_bytes, options, initial_operating_point, {},
                              inspected_perturbation, preapplied_mutation, false, false, extent,
                              observation);
  }

  const major_image_ownership::ScopedPhase incremental_phase(
      major_image_ownership::Phase::IncrementalPatch, /*only_if_none=*/true);
  ConSanPerturbationPlanningState local_perturbation;
  ConSanPerturbationPlanningState &perturbation =
      inspected_perturbation == nullptr ? local_perturbation : *inspected_perturbation;
  ConSanOptions planning_options = options;
  planning_options.fault_dry_run = true;
  if (!has_perturbation)
    planning_options.fault_require_exactly_one = false;
  ConSanPerturbationPlanningState pristine_perturbation;
  ConSanTransformArtifacts pristine_mutation_plan = lower_consan_stage(
      code_object_bytes, planning_options, initial_operating_point, {}, &pristine_perturbation, {},
      true, options.fault_require_exactly_one, ConSanLoweringExtent::Complete, nullptr);
  if (pristine_mutation_plan.outcome == ConSanTransformOutcome::Unsupported)
    return pristine_mutation_plan;
  const size_t retained_planned_faults = pristine_mutation_plan.mutation.fault.planned;
  const std::vector<ConSanFaultMutationPlan> retained_fault_plans =
      pristine_mutation_plan.fault_plans;
  if (has_perturbation &&
      (!pristine_mutation_plan.errors.empty() || pristine_mutation_plan.fault_plans.empty() ||
       pristine_perturbation.plans.empty())) {
    pristine_mutation_plan.outcome = ConSanTransformOutcome::Unsupported;
    pristine_mutation_plan.warnings.emplace_back(
        has_perturbation
            ? "ConSan composite planning did not select exact fault and perturbation plans"
            : "ConSan mutation planning did not select an exact fault plan");
    return pristine_mutation_plan;
  }
  // The pristine mutation stage both resolved and applied the fault.
  // Its replacement must be instrumented so probes model the instruction
  // that the GPU will actually execute.
  ConSanTransformArtifacts fault_result =
      has_perturbation ? pristine_mutation_plan : std::move(pristine_mutation_plan);
  fault_result = finalize_consan_result(std::move(fault_result), code_object_bytes,
                                        options.moi_report_dispatch_id);
  if (fault_result.outcome != ConSanTransformOutcome::ModifiedValid)
    return fault_result;
  const major_image_ownership::ScopedOwner mutation_image_owner(
      major_image_ownership::OwnerKind::CompositeImage, fault_result.replacement);
  const major_image_ownership::ScopedPhase composite_phase(
      major_image_ownership::Phase::CompositeIncrementalPatch);

  ConSanOptions instrument_options = options;
  ConSanPreappliedMutationLayout instrument_preapplied;
  clear_consan_fault_mutations(instrument_options);
  // Cardinality was enforced on the mutation stage. The subsequent
  // instrumentation-only pass has no mutation of its own to count.
  instrument_options.fault_require_exactly_one = false;
  instrument_options.abort_unmatched_barrier_wait = false;
  for (const ConSanPatchMutationProduct &patch : fault_result.patches) {
    if (patch.original_size != 0) {
      instrument_preapplied.reserved_ranges.push_back(
          {.text_offset = patch.anchor_offset, .size = patch.original_size});
    }
    if (patch.trampoline_size != 0) {
      instrument_preapplied.reserved_ranges.push_back(
          {.text_offset = patch.trampoline_offset, .size = patch.trampoline_size});
    }
    if (patch.kind != ConSanPatchKind::InlineBarrierMoveTargetRewrite || patch.trampoline_size == 0)
      continue;
    const auto owner =
        std::ranges::find_if(fault_result.program_inventory.kernels(), [&](const auto &kernel) {
          return kernel.has_text_range && patch.anchor_offset >= kernel.entry_text_offset &&
                 patch.anchor_offset - kernel.entry_text_offset < kernel.code_size;
        });
    if (owner == fault_result.program_inventory.kernels().end()) {
      fault_result.errors.emplace_back(
          "ConSan markerless barrier move could not retain fault-cave kernel ownership");
      return fault_result;
    }
    instrument_preapplied.code_ranges.push_back(
        {.text_offset = patch.trampoline_offset,
         .size = patch.trampoline_size,
         .kernel_name = owner->name,
         .continuation_text_offset = patch.anchor_offset + patch.original_size});
  }
  std::vector<ConSanPerturbationPlan> staged_carried;
  if (has_perturbation) {
    std::string carry_error;
    auto carried =
        carry_perturbation_plans_across_mutation(pristine_mutation_plan.program_inventory,
                                                 pristine_perturbation, fault_result, &carry_error);
    if (!carried)
      return rollback_composite_mutation(std::move(fault_result), std::move(carry_error));
    instrument_options.sc_perturb_identity.clear();
    instrument_options.sc_perturb_index = 0;
    staged_carried = std::move(*carried);
  }
  const std::span<const ConSanPerturbationPlan> instrument_carried = staged_carried;
  ConSanTransformArtifacts instrument_result = lower_consan_stage(
      fault_result.replacement, instrument_options, initial_operating_point, instrument_carried,
      &perturbation, instrument_preapplied, false, false, ConSanLoweringExtent::Complete, nullptr);
  if (!instrument_result.errors.empty())
    instrument_result.outcome = ConSanTransformOutcome::Invalid;
  else if (instrument_result.outcome != ConSanTransformOutcome::ModifiedValid &&
           summarize_consan_resource_plans(instrument_result.resource_plans).unsupported_plans != 0)
    instrument_result.outcome = ConSanTransformOutcome::Unsupported;
  const major_image_ownership::ScopedOwner instrument_result_owner(
      major_image_ownership::OwnerKind::ResultImage, instrument_result.replacement);
  // Resource ownership is a logical kernel identity, but instrumentation of
  // a staged mutation records descriptor offsets in the mutation image.
  // Normalize those offsets before merging stages: an intermediate offset
  // can otherwise collide with a different pristine kernel descriptor.
  ConSanDescriptorNameIndex pristine_descriptor_offsets;
  const major_image_ownership::ScopedOwner pristine_index_owner(
      major_image_ownership::OwnerKind::CompactIndex, &pristine_descriptor_offsets,
      measure_descriptor_name_index);
  {
    AmdGpuCodeObject pristine_code_object(code_object_bytes.data(), code_object_bytes.size());
    pristine_descriptor_offsets.reserve(pristine_code_object.kernels().size());
    for (const AmdGpuKernelInfo &kernel : pristine_code_object.kernels())
      pristine_descriptor_offsets.emplace(kernel.name, kernel.descriptor_file_offset);
    pristine_index_owner.checkpoint();
  }
  ConSanDescriptorOwnerIndex pristine_owner_offsets;
  const major_image_ownership::ScopedOwner owner_index_owner(
      major_image_ownership::OwnerKind::CompactIndex, &pristine_owner_offsets,
      measure_descriptor_owner_index);
  {
    AmdGpuCodeObject staged_code_object(fault_result.replacement.data(),
                                        fault_result.replacement.size());
    for (const AmdGpuKernelInfo &staged_kernel : staged_code_object.kernels()) {
      const auto pristine = pristine_descriptor_offsets.find(staged_kernel.name);
      if (pristine != pristine_descriptor_offsets.end()) {
        pristine_owner_offsets.emplace(staged_kernel.descriptor_file_offset, pristine->second);
      }
    }
    owner_index_owner.checkpoint();
  }
  const auto normalize_owner_offsets = [&](std::vector<uint64_t> &offsets) {
    for (uint64_t &offset : offsets) {
      const auto pristine = pristine_owner_offsets.find(offset);
      if (pristine != pristine_owner_offsets.end())
        offset = pristine->second;
    }
  };
  for (ConSanCandidateResourcePlan &plan : instrument_result.resource_plans)
    normalize_owner_offsets(plan.owner_descriptor_file_offsets);
  for (ConSanPatchAbiEffects &patch : instrument_result.patches)
    normalize_owner_offsets(patch.owner_descriptor_file_offsets);
  instrument_result.warnings.insert(instrument_result.warnings.begin(),
                                    std::make_move_iterator(fault_result.warnings.begin()),
                                    std::make_move_iterator(fault_result.warnings.end()));
  instrument_result.mutation.fault = fault_result.mutation.fault;
  instrument_result.mutation.applied_fault_logical_identity =
      std::move(fault_result.mutation.applied_fault_logical_identity);
  switch (instrument_result.outcome) {
  case ConSanTransformOutcome::ModifiedValid:
    if (fault_result.outcome != ConSanTransformOutcome::ModifiedValid) {
      instrument_result.errors.emplace_back(
          "ConSan staged composition is missing an independently validated stage");
      instrument_result.outcome = ConSanTransformOutcome::Invalid;
      instrument_result.discard_candidate_modification();
      instrument_result.mutation.fault.applied = 0;
      instrument_result.mutation.applied_fault_logical_identity.reset();
      return instrument_result;
    }
    // Retain the exact applied mutation plan. The instrumentation pass
    // inventories the mutated image, so without this provenance a live
    // staged result loses which dry-run plan authorized the mutation.
    if (!retained_fault_plans.empty()) {
      instrument_result.mutation.fault.planned = retained_planned_faults;
      instrument_result.fault_plans = retained_fault_plans;
    }
    if (has_perturbation) {
      instrument_result.mutation.perturbation.requested =
          pristine_mutation_plan.mutation.perturbation.requested;
      instrument_result.mutation.perturbation.planned =
          pristine_mutation_plan.mutation.perturbation.planned;
    }
    instrument_result.patches.insert(instrument_result.patches.begin(),
                                     std::make_move_iterator(fault_result.patches.begin()),
                                     std::make_move_iterator(fault_result.patches.end()));
    return instrument_result;
  case ConSanTransformOutcome::Unchanged:
    // A no-target instrumentation stage must not discard an otherwise valid
    // mutation. Its inventory describes the staged image being committed.
    if (!retained_fault_plans.empty()) {
      instrument_result.mutation.fault.planned = retained_planned_faults;
      instrument_result.fault_plans = retained_fault_plans;
    }
    instrument_result.replacement = std::move(fault_result.replacement);
    instrument_result.patches = std::move(fault_result.patches);
    instrument_result.mark_modified();
    return instrument_result;
  case ConSanTransformOutcome::Unsupported:
    if (instrument_options.flavor == ConSanFlavor::SuperCollider && !instrument_options.probe_nop &&
        !instrument_options.probe_trampoline_nop && !instrument_options.probe_endpgm &&
        !instrument_options.probe_lds_endpgm && !instrument_options.probe_lds_check_trap &&
        !instrument_options.probe_flat_check_trap && !instrument_options.probe_flat_trap &&
        instrument_options.sc_perturb_kind == ConSanPerturbationKind::None) {
      // The second pass may classify the intentionally weakened image as a
      // preflight skip. With no instrumentation request, that is not a
      // reason to discard an independently validated mutation stage.
      instrument_result.outcome = ConSanTransformOutcome::Unchanged;
      if (!retained_fault_plans.empty()) {
        instrument_result.mutation.fault.planned = retained_planned_faults;
        instrument_result.fault_plans = retained_fault_plans;
      }
      instrument_result.replacement = std::move(fault_result.replacement);
      instrument_result.patches = std::move(fault_result.patches);
      instrument_result.mark_modified();
      return instrument_result;
    }
    instrument_result.warnings.emplace_back(
        "ConSan rolled back staged fault mutation because instrumentation is unsupported");
    instrument_result.mutation.fault.applied = 0;
    instrument_result.mutation.applied_fault_logical_identity.reset();
    break;
  case ConSanTransformOutcome::Invalid:
    instrument_result.warnings.emplace_back(
        "ConSan rolled back staged fault mutation because instrumentation is invalid");
    instrument_result.mutation.fault.applied = 0;
    instrument_result.mutation.applied_fault_logical_identity.reset();
    break;
  }
  instrument_result.discard_candidate_modification();
  return instrument_result;
}

[[nodiscard]] ConSanTransformArtifacts
lower_consan_stage(std::span<const uint8_t> code_object_bytes, const ConSanOptions &options,
                   const ConSanMoiOperatingPoint &initial_operating_point,
                   std::span<const ConSanPerturbationPlan> carried_plans,
                   ConSanPerturbationPlanningState *inspected_perturbation,
                   const ConSanPreappliedMutationLayout &preapplied_mutation, bool apply_fault_plan,
                   bool require_exactly_one_applied_fault, ConSanLoweringExtent extent,
                   const ConSanLoweringObservation *observation) {
  const major_image_ownership::ScopedPhase incremental_phase(
      major_image_ownership::Phase::IncrementalPatch, /*only_if_none=*/true);
  ConSanTransformArtifacts result;
  ConSanPerturbationPlanningState local_perturbation;
  ConSanPerturbationPlanningState &perturbation =
      inspected_perturbation == nullptr ? local_perturbation : *inspected_perturbation;
  perturbation = {};
  result.mutation.fault.requested = options.fault_mutation_count();
  ProgramInventoryBuilder inventory_builder(code_object_bytes, preapplied_mutation);
  result.program_inventory = inventory_builder.view();
  const bool has_perturbation = options.sc_perturb_kind != ConSanPerturbationKind::None;
  const bool has_atomic_fault = options.fault_atomic_wrong_address ||
                                options.fault_atomic_weaken_order ||
                                options.fault_atomic_weaken_scope;
  const bool has_ordinary_fault = options.fault_ordinary_wrong_address ||
                                  options.fault_ordinary_weaken_order ||
                                  options.fault_ordinary_weaken_scope;
  if (has_ordinary_fault && has_perturbation) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan ordinary fault/perturbation composition is not yet qualified");
    return result;
  }
  if (has_atomic_fault && has_perturbation &&
      (options.sc_perturb_kind != ConSanPerturbationKind::Atomic || options.fault_drop_barrier ||
       options.fault_move_barrier || options.fault_mutate_barrier_id_scope ||
       options.fault_mutate_barrier_participants)) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan atomic fault/perturbation composition requires the SC2D composite proof");
    return result;
  }
  const major_image_ownership::ScopedOwner result_owner(
      major_image_ownership::OwnerKind::ResultImage, result.replacement);

  if (options.flavor == ConSanFlavor::None)
    return result;

  std::unique_ptr<AmdGpuCodeObject> code_object;
  ConSanProgramAnalysisResult analysis;
  const bool analysis_succeeded = analyze_consan_program_inventory(
      code_object_bytes, options, code_object, inventory_builder, perturbation, analysis);
  result.program_inventory = std::move(analysis.program_inventory);
  result.fault_sites = std::move(analysis.fault_sites);
  result.barrier_move_destinations = std::move(analysis.barrier_move_destinations);
  result.outcome = analysis.outcome;
  result.warnings.insert(result.warnings.end(), std::make_move_iterator(analysis.warnings.begin()),
                         std::make_move_iterator(analysis.warnings.end()));
  result.errors.insert(result.errors.end(), std::make_move_iterator(analysis.errors.begin()),
                       std::make_move_iterator(analysis.errors.end()));
  if (!analysis_succeeded)
    return result;
  const rj_code_arch_t arch = result.program_inventory.arch();
  const ConSanTargetProfile *target_profile =
      consan_target_profile(result.program_inventory.target());
  if (extent == ConSanLoweringExtent::ThroughProgramInventory)
    return result;
  if (options.fault_dry_run) {
    resolve_consan_fault_mutations(*code_object, arch, options, apply_fault_plan, apply_fault_plan,
                                   require_exactly_one_applied_fault, result);
    build_perturbation_plan(result.program_inventory, options, perturbation,
                            result.mutation.perturbation, result.errors, carried_plans);
    return result;
  }
  if (has_perturbation) {
    build_perturbation_plan(result.program_inventory, options, perturbation,
                            result.mutation.perturbation, result.errors, carried_plans);
    if (!result.errors.empty())
      return result;
  }

  if (!initialize_consan_lowering_observation(options, result, observation))
    return result;

  if (options.flavor == ConSanFlavor::Moi) {
    // Incremental MOI passes reparse their current result. Release the
    // pristine inventory parser before entering them so its backing image,
    // section objects and names, and derived state cannot overlap every
    // stage's parser, patcher, text, and transaction.
    code_object.reset();
    return try_patch_consan_moi(std::move(result), options, initial_operating_point,
                                code_object_bytes, arch);
  }

  if (options.flavor != ConSanFlavor::SuperCollider) {
    result.errors.emplace_back("ConSan selected unknown flavor");
    return result;
  }

  const bool has_redundant_access = options.probe_lds_check_trap || options.probe_flat_check_trap;
  const uint32_t overall_patch_budget = std::max<uint32_t>(options.max_patches, 1u);
  if (perturbation.plans.size() > overall_patch_budget) {
    result.errors.emplace_back("ConSan SC perturbation exceeds the shared patch budget");
    return result;
  }
  ConSanOptions redundant_options = options;
  redundant_options.max_patches =
      overall_patch_budget - static_cast<uint32_t>(perturbation.plans.size());

  // LDS and FLAT stage site-local programs without choosing physical caves.
  // Perturbation composes against the still-source-coordinate image, and one
  // final relocation then carries every selected operation into active text.
  if (options.probe_lds_check_trap)
    try_apply_lds_load_check_trap_patch(*code_object, *target_profile, redundant_options, result);

  if (has_perturbation && result.errors.empty()) {
    try_apply_perturbation_patches(*code_object, arch, options, perturbation, result);
    if (result.mutation.perturbation.applied != perturbation.plans.size()) {
      if (result.outcome == ConSanTransformOutcome::ModifiedValid)
        result.outcome = result.errors.empty() ? ConSanTransformOutcome::Unchanged
                                               : ConSanTransformOutcome::Invalid;
      result.discard_candidate_modification();
    }
  }

  if (has_redundant_access) {
    if (options.probe_flat_check_trap && result.errors.empty()) {
      if (!result.modified()) {
        try_apply_flat_check_trap_patch(*code_object, *target_profile, redundant_options, result);
      } else {
        const uint32_t max_patches = redundant_options.max_patches;
        const uint32_t access_patches = static_cast<uint32_t>(std::ranges::count_if(
                                            result.patches,
                                            [](const ConSanPatchLoweringProduct &patch) {
                                              switch (patch.kind) {
                                              case ConSanPatchKind::LdsLoadCheckTrap:
                                              case ConSanPatchKind::LdsStoreCheckTrap:
                                              case ConSanPatchKind::FlatLoadCheckTrap:
                                              case ConSanPatchKind::FlatStoreCheckTrap:
                                                return true;
                                              default:
                                                return false;
                                              }
                                            })) +
                                        static_cast<uint32_t>(result.staged_text_fragments.size());
        if (access_patches >= max_patches && redundant_options.max_patches_is_expert_limit) {
          ConSanOptions flat_options = redundant_options;
          flat_options.max_patches = 0u;
          try_apply_flat_check_trap_patch(*code_object, *target_profile, flat_options, result);
        } else if (auto reserved_ranges =
                       reserved_ranges_for_existing_patches(*code_object, result.patches)) {
          ConSanOptions flat_options = redundant_options;
          if (flat_options.max_patches_is_expert_limit)
            flat_options.max_patches = max_patches - access_patches;
          try_apply_flat_check_trap_patch(*code_object, *target_profile, flat_options, result,
                                          *reserved_ranges);
        } else {
          ConSanOptions flat_options = redundant_options;
          flat_options.max_patches = 0u;
          flat_options.max_patches_is_expert_limit = true;
          try_apply_flat_check_trap_patch(*code_object, *target_profile, flat_options, result);
          result.warnings.emplace_back(
              "ConSan flat check/trap proof skipped after native LDS patching: "
              "existing patch ranges could not be mapped in the original code object");
        }
      }
    }
  } else if (!has_perturbation && options.probe_flat_trap)
    try_apply_flat_trap_patch(*code_object, *target_profile, options, result);
  else if (!has_perturbation && options.probe_lds_endpgm)
    try_apply_lds_endpgm_patch(*code_object, arch, result);
  else if (!has_perturbation && options.probe_endpgm)
    try_apply_proof_endpgm_patch(*code_object, arch, result);
  else if (!has_perturbation && (options.probe_nop || options.probe_trampoline_nop))
    try_apply_proof_nop_patch(*code_object, arch, options.probe_trampoline_nop, result);

  if (!result.staged_text_fragments.empty() && result.errors.empty()) {
    const std::string_view subject = options.probe_lds_check_trap && options.probe_flat_check_trap
                                         ? "SuperCollider access proof"
                                     : options.probe_lds_check_trap ? "LDS check/trap proof"
                                                                    : "flat check/trap proof";
    if (!finalize_consan_text_rewrites(result.replacement, arch, options, subject, result))
      result.discard_candidate_modification();
  }

  publish_unplaced_sc_access_rejections(result.coverage_ledger, result.errors);

  return result;
}

} // namespace
