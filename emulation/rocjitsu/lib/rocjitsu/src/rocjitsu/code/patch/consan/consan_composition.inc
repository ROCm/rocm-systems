namespace {

/// Translate physical descriptor locations between two inventory revisions.
/// Kernel arena IDs are assigned after the parser's deterministic name sort,
/// so the same ID and name identify one logical kernel even when an earlier
/// mutation moved its descriptor bytes.  Any mismatch is stale transaction
/// state and fails closed instead of preserving an ambiguous physical offset.
[[nodiscard]] bool translate_owner_descriptors(const ProgramInventory &source,
                                               const ProgramInventory &destination,
                                               std::vector<uint64_t> &offsets,
                                               uint64_t *stale_offset) {
  for (uint64_t &offset : offsets) {
    const ProgramContainer *source_kernel = source.find_kernel_by_descriptor(offset);
    if (source_kernel == nullptr) {
      // Retry-carried operating-point plans are already expressed in the
      // destination revision.  Accept only an exact destination descriptor;
      // an offset belonging to neither revision is stale.
      if (destination.find_kernel_by_descriptor(offset) == nullptr) {
        *stale_offset = offset;
        return false;
      }
      continue;
    }
    const ProgramContainer *destination_kernel = destination.container(source_kernel->id);
    if (destination_kernel == nullptr || !destination_kernel->is_kernel() ||
        destination_kernel->name != source_kernel->name) {
      *stale_offset = offset;
      return false;
    }
    offset = destination_kernel->descriptor_file_offset;
  }
  return true;
}

/// Install the transaction's immutable observation product into native
/// lowering state. The fallback exists for focused lowerer tests and legacy
/// mechanism entry points; it uses the same common assembly authority and is
/// deliberately outside both mode implementations.
[[nodiscard]] bool
initialize_lowering_observation(const Options &options, TransformArtifacts &result,
                                const LoweringObservation *prepared_observation) {
  if (prepared_observation != nullptr) {
    if (!prepared_observation->well_formed()) {
      result.errors.emplace_back("ConSan received an invalid prepared observation product");
      return false;
    }
    result.coverage_ledger = prepared_observation->initial_coverage;
    return true;
  }

  ObservationProduct observation =
      assemble_observation_product(result.program_inventory, static_cast<const Request &>(options),
                                   static_cast<const DebugOverrides &>(options));
  const bool valid = observation.valid();
  result.coverage_ledger = std::move(observation.initial_coverage);
  result.errors.insert(result.errors.end(),
                       std::make_move_iterator(observation.diagnostics.begin()),
                       std::make_move_iterator(observation.diagnostics.end()));
  return valid;
}

[[nodiscard]] std::optional<std::vector<SuperColliderPerturbationPlan>>
carry_supercollider_perturbation_plans_across_mutation(
    const ProgramInventory &pristine_inventory,
    const SuperColliderPerturbationPlanningState &planning, const SynchronizationMutation &mutation,
    std::string *error) {
  const SynchronizationInventoryView synchronization = pristine_inventory.sync();
  std::vector<SuperColliderPerturbationPlan> carried;
  carried.reserve(planning.plans.size());
  for (const SuperColliderPerturbationPlan &plan : planning.plans) {
    const SuperColliderPerturbationCandidate &candidate = plan.candidate;
    const SyncSequence *source_sequence = synchronization.find_sequence(candidate.sequence);
    const SyncEvent *source_anchor = synchronization.find_event(candidate.anchor_event);
    if (!candidate.eligible || source_sequence == nullptr || source_anchor == nullptr ||
        plan.sequence_identity.empty() || plan.container_name.empty() || plan.anchor_size == 0u) {
      *error = "selected pristine perturbation identity is incomplete or stale";
      return std::nullopt;
    }

    SuperColliderPerturbationPlan translated = plan;
    for (const SyncEventMutation &effect : mutation.events) {
      if (effect.source_event != candidate.anchor_event)
        continue;
      switch (effect.kind) {
      case SyncEventMutationKind::Destroyed:
        *error = "selected perturbation edge was destroyed by the fault mutation";
        return std::nullopt;
      case SyncEventMutationKind::Relocated:
        translated.anchor_text_offset = effect.relocated_text_offset;
        break;
      case SyncEventMutationKind::AtomicRewrite:
        translated.overlaps_atomic_mutation = true;
        break;
      case SyncEventMutationKind::AtomicBoundaryRemoval:
        translated.overlaps_atomic_mutation = true;
        translated.removed_cache_boundary = true;
        break;
      }
    }

    translated.source_sequence_identity = source_sequence->identity;
    translated.source_sequence_semantics_weakened =
        mutation.weakened_sequence == candidate.sequence;
    carried.push_back(std::move(translated));
  }
  return carried;
}

[[nodiscard]] TransformArtifacts rollback_composite_mutation(TransformArtifacts result,
                                                             std::string reason) {
  result.outcome = TransformOutcome::Unsupported;
  result.discard_candidate_modification();
  result.mutation.fault.applied = 0;
  result.mutation.supercollider_perturbation.applied = 0;
  result.warnings.emplace_back("ConSan rolled back staged fault/perturbation composition: " +
                               std::move(reason));
  return result;
}

[[nodiscard]] TransformArtifacts
lower_stage(std::span<const uint8_t> code_object_bytes, const Options &options,
            const OperatingPoint &initial_operating_point,
            std::span<const SuperColliderPerturbationPlan> carried_plans,
            SuperColliderPerturbationPlanningState *inspected_supercollider_perturbation,
            const PreappliedMutationLayout &preapplied_mutation, bool apply_fault_plan,
            bool require_exactly_one_applied_fault, LoweringExtent extent,
            const LoweringObservation *observation);

[[nodiscard]] TransformArtifacts
try_patch_impl(std::span<const uint8_t> code_object_bytes, const Options &options,
               const OperatingPoint &initial_operating_point,
               SuperColliderPerturbationPlanningState *inspected_supercollider_perturbation,
               const PreappliedMutationLayout &preapplied_mutation, LoweringExtent extent,
               const LoweringObservation *observation) {
  const bool has_fault = options.has_fault_mutation();
  const bool has_supercollider_perturbation =
      options.supercollider_perturb_kind != SuperColliderPerturbationKind::None;
  if (!has_fault || options.fault_dry_run) {
    return lower_stage(code_object_bytes, options, initial_operating_point, {},
                       inspected_supercollider_perturbation, preapplied_mutation, false, false,
                       extent, observation);
  }

  const major_image_ownership::ScopedPhase incremental_phase(
      major_image_ownership::Phase::IncrementalPatch, /*only_if_none=*/true);
  SuperColliderPerturbationPlanningState local_supercollider_perturbation;
  SuperColliderPerturbationPlanningState &supercollider_perturbation =
      inspected_supercollider_perturbation == nullptr ? local_supercollider_perturbation
                                                      : *inspected_supercollider_perturbation;
  Options planning_options = options;
  planning_options.fault_dry_run = true;
  if (!has_supercollider_perturbation)
    planning_options.fault_require_exactly_one = false;
  SuperColliderPerturbationPlanningState pristine_supercollider_perturbation;
  TransformArtifacts pristine_mutation_plan =
      lower_stage(code_object_bytes, planning_options, initial_operating_point, {},
                  &pristine_supercollider_perturbation, {}, true, options.fault_require_exactly_one,
                  LoweringExtent::Complete, nullptr);
  if (pristine_mutation_plan.outcome == TransformOutcome::Unsupported)
    return pristine_mutation_plan;
  if (has_supercollider_perturbation &&
      (!pristine_mutation_plan.errors.empty() || pristine_mutation_plan.fault_plans.empty() ||
       pristine_supercollider_perturbation.plans.empty())) {
    pristine_mutation_plan.outcome = TransformOutcome::Unsupported;
    pristine_mutation_plan.warnings.emplace_back(
        "ConSan composite planning did not select exact fault and perturbation plans");
    return pristine_mutation_plan;
  }
  std::vector<SuperColliderPerturbationPlan> staged_carried;
  if (has_supercollider_perturbation) {
    std::string carry_error;
    auto carried = carry_supercollider_perturbation_plans_across_mutation(
        pristine_mutation_plan.program_inventory, pristine_supercollider_perturbation,
        pristine_mutation_plan.synchronization_mutation, &carry_error);
    if (!carried)
      return rollback_composite_mutation(std::move(pristine_mutation_plan), std::move(carry_error));
    staged_carried = std::move(*carried);
  }
  const MutationTally retained_supercollider_perturbation_tally =
      pristine_mutation_plan.mutation.supercollider_perturbation;
  // The pristine mutation stage both resolved and applied the fault.
  // Its replacement must be instrumented so probes model the instruction
  // that the GPU will actually execute.
  TransformArtifacts fault_result = std::move(pristine_mutation_plan);
  fault_result = finalize_result(std::move(fault_result), code_object_bytes);
  if (fault_result.outcome != TransformOutcome::ModifiedValid)
    return fault_result;
  const major_image_ownership::ScopedOwner mutation_image_owner(
      major_image_ownership::OwnerKind::CompositeImage, fault_result.replacement);
  const major_image_ownership::ScopedPhase composite_phase(
      major_image_ownership::Phase::CompositeIncrementalPatch);

  Options instrument_options = options;
  PreappliedMutationLayout instrument_preapplied;
  clear_fault_mutations(instrument_options);
  // Cardinality was enforced on the mutation stage. The subsequent
  // instrumentation-only pass has no mutation of its own to count.
  instrument_options.fault_require_exactly_one = false;
  instrument_options.abort_unmatched_barrier_wait = false;
  for (const PatchMutationProduct &patch : fault_result.patches) {
    if (patch.original_size != 0) {
      instrument_preapplied.reserved_ranges.push_back(
          {.text_offset = patch.anchor_offset, .size = patch.original_size});
    }
    if (patch.trampoline_size != 0) {
      instrument_preapplied.reserved_ranges.push_back(
          {.text_offset = patch.trampoline_offset, .size = patch.trampoline_size});
    }
    if (patch.kind != PatchKind::InlineBarrierMoveTargetRewrite || patch.trampoline_size == 0)
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
  if (has_supercollider_perturbation) {
    instrument_options.supercollider_perturb_identity.clear();
    instrument_options.supercollider_perturb_index = 0;
  }
  const std::span<const SuperColliderPerturbationPlan> instrument_carried = staged_carried;
  TransformArtifacts instrument_result =
      lower_stage(fault_result.replacement, instrument_options, initial_operating_point,
                  instrument_carried, &supercollider_perturbation, instrument_preapplied, false,
                  false, LoweringExtent::Complete, nullptr);
  if (!instrument_result.errors.empty())
    instrument_result.outcome = TransformOutcome::Invalid;
  else if (instrument_result.outcome != TransformOutcome::ModifiedValid &&
           summarize_resource_plans(instrument_result.resource_plans).unsupported_plans != 0)
    instrument_result.outcome = TransformOutcome::Unsupported;
  const major_image_ownership::ScopedOwner instrument_result_owner(
      major_image_ownership::OwnerKind::ResultImage, instrument_result.replacement);
  // Patch ABI materializes descriptor locations in its current mutation image.
  // Resource plans retain stable kernel handles, so only byte-oriented patch
  // coordinates require translation before merging revision-local proof.
  uint64_t stale_owner_offset = 0;
  const auto translate_owners = [&](std::vector<uint64_t> &offsets) {
    return translate_owner_descriptors(instrument_result.program_inventory,
                                       fault_result.program_inventory, offsets,
                                       &stale_owner_offset);
  };
  const bool patch_owners_valid =
      std::ranges::all_of(instrument_result.patches, [&](PatchAbiEffects &patch) {
        return translate_owners(patch.owner_descriptor_file_offsets);
      });
  if (!patch_owners_valid) {
    instrument_result.errors.emplace_back(
        "ConSan staged composition found stale execution-owner identity at descriptor offset " +
        std::to_string(stale_owner_offset));
    instrument_result.outcome = TransformOutcome::Invalid;
  }
  instrument_result.warnings.insert(instrument_result.warnings.begin(),
                                    std::make_move_iterator(fault_result.warnings.begin()),
                                    std::make_move_iterator(fault_result.warnings.end()));
  instrument_result.mutation.fault = fault_result.mutation.fault;
  instrument_result.mutation.applied_fault_logical_identity =
      std::move(fault_result.mutation.applied_fault_logical_identity);
  switch (instrument_result.outcome) {
  case TransformOutcome::ModifiedValid:
    if (fault_result.outcome != TransformOutcome::ModifiedValid) {
      instrument_result.errors.emplace_back(
          "ConSan staged composition is missing an independently validated stage");
      instrument_result.outcome = TransformOutcome::Invalid;
      instrument_result.discard_candidate_modification();
      instrument_result.mutation.fault.applied = 0;
      instrument_result.mutation.applied_fault_logical_identity.reset();
      return instrument_result;
    }
    // Retain the exact applied mutation plan. The instrumentation pass
    // inventories the mutated image, so without this provenance a live
    // staged result loses which dry-run plan authorized the mutation.
    if (!fault_result.fault_plans.empty())
      instrument_result.fault_plans = std::move(fault_result.fault_plans);
    if (has_supercollider_perturbation) {
      instrument_result.mutation.supercollider_perturbation.requested =
          retained_supercollider_perturbation_tally.requested;
      instrument_result.mutation.supercollider_perturbation.planned =
          retained_supercollider_perturbation_tally.planned;
    }
    instrument_result.patches.insert(instrument_result.patches.begin(),
                                     std::make_move_iterator(fault_result.patches.begin()),
                                     std::make_move_iterator(fault_result.patches.end()));
    return instrument_result;
  case TransformOutcome::Unchanged:
    // A no-target instrumentation stage must not discard an otherwise valid
    // mutation. Its inventory describes the staged image being committed.
    if (!fault_result.fault_plans.empty())
      instrument_result.fault_plans = std::move(fault_result.fault_plans);
    instrument_result.replacement = std::move(fault_result.replacement);
    instrument_result.patches = std::move(fault_result.patches);
    instrument_result.mark_modified();
    return instrument_result;
  case TransformOutcome::Unsupported:
    if (instrument_options.mode == Mode::SuperCollider &&
        !instrument_options.probe_lds_check_trap && !instrument_options.probe_flat_check_trap &&
        instrument_options.supercollider_perturb_kind == SuperColliderPerturbationKind::None) {
      // The second pass may classify the intentionally weakened image as a
      // preflight skip. With no instrumentation request, that is not a
      // reason to discard an independently validated mutation stage.
      instrument_result.outcome = TransformOutcome::Unchanged;
      if (!fault_result.fault_plans.empty())
        instrument_result.fault_plans = std::move(fault_result.fault_plans);
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
  case TransformOutcome::Invalid:
    instrument_result.warnings.emplace_back(
        "ConSan rolled back staged fault mutation because instrumentation is invalid");
    instrument_result.mutation.fault.applied = 0;
    instrument_result.mutation.applied_fault_logical_identity.reset();
    break;
  }
  instrument_result.discard_candidate_modification();
  return instrument_result;
}

[[nodiscard]] TransformArtifacts
lower_stage(std::span<const uint8_t> code_object_bytes, const Options &options,
            const OperatingPoint &initial_operating_point,
            std::span<const SuperColliderPerturbationPlan> carried_plans,
            SuperColliderPerturbationPlanningState *inspected_supercollider_perturbation,
            const PreappliedMutationLayout &preapplied_mutation, bool apply_fault_plan,
            bool require_exactly_one_applied_fault, LoweringExtent extent,
            const LoweringObservation *observation) {
  const major_image_ownership::ScopedPhase incremental_phase(
      major_image_ownership::Phase::IncrementalPatch, /*only_if_none=*/true);
  TransformArtifacts result;
  SuperColliderPerturbationPlanningState local_supercollider_perturbation;
  SuperColliderPerturbationPlanningState &supercollider_perturbation =
      inspected_supercollider_perturbation == nullptr ? local_supercollider_perturbation
                                                      : *inspected_supercollider_perturbation;
  supercollider_perturbation = {};
  result.mutation.fault.requested = options.fault_mutation_count();
  ProgramInventoryBuilder inventory_builder(code_object_bytes, preapplied_mutation);
  result.program_inventory = inventory_builder.view();
  const bool has_supercollider_perturbation =
      options.supercollider_perturb_kind != SuperColliderPerturbationKind::None;
  const bool has_atomic_fault = options.fault_atomic_wrong_address ||
                                options.fault_atomic_weaken_order ||
                                options.fault_atomic_weaken_scope;
  const bool has_ordinary_fault = options.fault_ordinary_wrong_address ||
                                  options.fault_ordinary_weaken_order ||
                                  options.fault_ordinary_weaken_scope;
  if (has_ordinary_fault && has_supercollider_perturbation) {
    result.outcome = TransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan ordinary fault/perturbation composition is not yet qualified");
    return result;
  }
  if (has_atomic_fault && has_supercollider_perturbation &&
      (options.supercollider_perturb_kind != SuperColliderPerturbationKind::Atomic ||
       options.fault_drop_barrier || options.fault_move_barrier ||
       options.fault_mutate_barrier_id_scope || options.fault_mutate_barrier_participants)) {
    result.outcome = TransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan atomic fault/perturbation composition requires the SC2D composite proof");
    return result;
  }
  const major_image_ownership::ScopedOwner result_owner(
      major_image_ownership::OwnerKind::ResultImage, result.replacement);

  if (options.mode == Mode::None)
    return result;

  std::unique_ptr<AmdGpuCodeObject> code_object;
  ProgramAnalysisResult analysis;
  const bool analysis_succeeded = analyze_program_inventory(
      code_object_bytes, static_cast<const Request &>(options),
      static_cast<const DebugOverrides &>(options), static_cast<const MutationRequest &>(options),
      code_object, inventory_builder, supercollider_perturbation, analysis);
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
  const TargetProfile *target_profile = consan::target_profile(result.program_inventory.target());
  if (extent == LoweringExtent::ThroughProgramInventory)
    return result;
  if (options.fault_dry_run) {
    resolve_fault_mutations(*code_object, arch, static_cast<const MutationRequest &>(options),
                            static_cast<const DebugOverrides &>(options),
                            static_cast<const TransformPolicy &>(options), apply_fault_plan,
                            apply_fault_plan, require_exactly_one_applied_fault, result);
    build_supercollider_perturbation_plan(
        result.program_inventory, static_cast<const Request &>(options),
        static_cast<const MutationRequest &>(options), static_cast<const DebugOverrides &>(options),
        supercollider_perturbation, result.mutation.supercollider_perturbation, result.errors,
        carried_plans);
    return result;
  }
  if (has_supercollider_perturbation) {
    build_supercollider_perturbation_plan(
        result.program_inventory, static_cast<const Request &>(options),
        static_cast<const MutationRequest &>(options), static_cast<const DebugOverrides &>(options),
        supercollider_perturbation, result.mutation.supercollider_perturbation, result.errors,
        carried_plans);
    if (!result.errors.empty())
      return result;
  }

  if (!initialize_lowering_observation(options, result, observation))
    return result;

  if (options.mode == Mode::Default) {
    // Incremental ConSan passes reparse their current result. Release the
    // pristine inventory parser before entering them so its backing image,
    // section objects and names, and derived state cannot overlap every
    // stage's parser, patcher, text, and transaction.
    code_object.reset();
    return try_patch(std::move(result), options, initial_operating_point, code_object_bytes, arch);
  }

  if (options.mode != Mode::SuperCollider) {
    result.errors.emplace_back("ConSan selected unknown mode");
    return result;
  }

  const bool has_redundant_access = options.probe_lds_check_trap || options.probe_flat_check_trap;
  const uint32_t overall_patch_budget = std::max<uint32_t>(options.max_patches, 1u);
  if (supercollider_perturbation.plans.size() > overall_patch_budget) {
    result.errors.emplace_back("ConSan SuperCollider perturbation exceeds the shared patch budget");
    return result;
  }
  Options redundant_options = options;
  redundant_options.max_patches =
      overall_patch_budget - static_cast<uint32_t>(supercollider_perturbation.plans.size());

  if (has_supercollider_perturbation && result.errors.empty()) {
    const bool staged = stage_supercollider_perturbation_patches(
        *code_object, arch, supercollider_perturbation, result);
    const size_t staged_count = static_cast<size_t>(
        std::ranges::count_if(result.staged_text_fragments, [](const TextFragment &fragment) {
          return fragment.patch.kind == PatchKind::TrampolineSuperColliderPerturbation;
        }));
    if (!staged || staged_count != supercollider_perturbation.plans.size()) {
      if (result.outcome == TransformOutcome::ModifiedValid)
        result.outcome =
            result.errors.empty() ? TransformOutcome::Unchanged : TransformOutcome::Invalid;
      result.discard_candidate_modification();
    }
  }

  // Perturbation retains an independently validated exact trampoline. Commit
  // that small fragment transaction before access fragments can nest a larger
  // probe program around the same guest boundary.
  if (has_supercollider_perturbation && has_redundant_access &&
      !result.staged_text_fragments.empty() && result.errors.empty()) {
    const std::span<const uint8_t> descriptor_image =
        result.replacement.empty() ? code_object_bytes
                                   : std::span<const uint8_t>(result.replacement);
    if (!finalize_text_rewrites(descriptor_image, arch, options.patched_image_growth_limit,
                                "SC perturbation", result))
      result.discard_candidate_modification();
  }

  // LDS and FLAT access proofs then share one object-wide relocation over the
  // already committed perturbation image.
  if (options.probe_lds_check_trap && result.errors.empty())
    try_apply_lds_load_check_trap_patch(*code_object, *target_profile, redundant_options, result);

  if (has_redundant_access) {
    if (options.probe_flat_check_trap && result.errors.empty()) {
      if (!result.modified()) {
        try_apply_flat_check_trap_patch(*code_object, *target_profile, redundant_options, result);
      } else {
        const uint32_t max_patches = redundant_options.max_patches;
        const uint32_t access_patches =
            static_cast<uint32_t>(std::ranges::count_if(result.patches,
                                                        [](const PatchLoweringProduct &patch) {
                                                          switch (patch.kind) {
                                                          case PatchKind::LdsLoadCheckTrap:
                                                          case PatchKind::LdsStoreCheckTrap:
                                                          case PatchKind::FlatLoadCheckTrap:
                                                          case PatchKind::FlatStoreCheckTrap:
                                                            return true;
                                                          default:
                                                            return false;
                                                          }
                                                        })) +
            static_cast<uint32_t>(result.staged_text_fragments.size());
        if (access_patches >= max_patches && redundant_options.max_patches_is_expert_limit) {
          Options flat_options = redundant_options;
          flat_options.max_patches = 0u;
          try_apply_flat_check_trap_patch(*code_object, *target_profile, flat_options, result);
        } else {
          Options flat_options = redundant_options;
          if (flat_options.max_patches_is_expert_limit)
            flat_options.max_patches = max_patches - access_patches;
          try_apply_flat_check_trap_patch(*code_object, *target_profile, flat_options, result);
        }
      }
    }
  }

  if (!result.staged_text_fragments.empty() && result.errors.empty()) {
    const std::string_view subject = options.probe_lds_check_trap && options.probe_flat_check_trap
                                         ? "SuperCollider access proof"
                                     : options.probe_lds_check_trap  ? "LDS check/trap proof"
                                     : options.probe_flat_check_trap ? "flat check/trap proof"
                                                                     : "SC perturbation";
    const std::span<const uint8_t> descriptor_image =
        result.replacement.empty() ? code_object_bytes
                                   : std::span<const uint8_t>(result.replacement);
    if (!finalize_text_rewrites(descriptor_image, arch, options.patched_image_growth_limit, subject,
                                result))
      result.discard_candidate_modification();
  }
  if (has_supercollider_perturbation) {
    result.mutation.supercollider_perturbation.applied = static_cast<size_t>(std::ranges::count(
        result.patches, PatchKind::TrampolineSuperColliderPerturbation, &PatchInfo::kind));
  }

  publish_unplaced_supercollider_access_rejections(result.coverage_ledger, result.errors);

  return result;
}

} // namespace
