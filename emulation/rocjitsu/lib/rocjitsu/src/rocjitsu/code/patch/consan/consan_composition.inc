namespace {

struct ConSanTransformStage {
  std::vector<uint8_t> image;
  std::vector<ConSanPatchInfo> patches;
  std::vector<std::string> warnings;
  size_t requested_fault_mutations = 0;
  size_t planned_fault_mutations = 0;
  size_t applied_fault_mutations = 0;
  std::optional<std::string> applied_fault_logical_identity;
};

[[nodiscard]] ConSanResult finalize_consan_result(ConSanResult result,
                                                  std::span<const uint8_t> original_bytes);

template <typename Site>
void move_sites_in_range(std::vector<Site> &source, std::vector<Site> &destination, uint64_t begin,
                         uint32_t size) {
  const uint64_t end = begin + size;
  for (auto it = source.begin(); it != source.end();) {
    if (it->text_offset < begin || it->text_offset >= end) {
      ++it;
      continue;
    }
    destination.push_back(std::move(*it));
    it = source.erase(it);
  }
}

template <typename Site>
void append_unique_sites(std::vector<Site> &source, std::vector<Site> &destination) {
  for (Site &site : source) {
    if (std::ranges::none_of(destination, [&](const Site &existing) {
          return existing.text_offset == site.text_offset;
        })) {
      destination.push_back(std::move(site));
    }
  }
  std::ranges::sort(destination, {}, &Site::text_offset);
}

void reattribute_preapplied_code_ranges(std::span<const uint8_t> code_object_bytes,
                                        Decoder &decoder, rj_code_arch_t arch,
                                        const ConSanOptions &options, ConSanResult &result) {
  for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
    const auto kernel = std::ranges::find_if(result.kernels, [&](const ConSanKernelInfo &item) {
      return item.name == range.kernel_name;
    });
    if (kernel == result.kernels.end()) {
      result.errors.emplace_back("ConSan could not recover the owner of a preapplied fault cave");
      continue;
    }
    for (ConSanKernelInfo &other : result.kernels) {
      if (&other == &*kernel)
        continue;
      move_sites_in_range(other.lds_sites, kernel->lds_sites, range.text_offset, range.size);
      move_sites_in_range(other.flat_sites, kernel->flat_sites, range.text_offset, range.size);
      move_sites_in_range(other.ordinary_memory_sites, kernel->ordinary_memory_sites,
                          range.text_offset, range.size);
      move_sites_in_range(other.barrier_sites, kernel->barrier_sites, range.text_offset,
                          range.size);
      move_sites_in_range(other.fence_sites, kernel->fence_sites, range.text_offset, range.size);
      move_sites_in_range(other.atomic_sites, kernel->atomic_sites, range.text_offset, range.size);
    }
    for (ConSanFunctionInfo &function : result.functions) {
      move_sites_in_range(function.lds_sites, kernel->lds_sites, range.text_offset, range.size);
      move_sites_in_range(function.flat_sites, kernel->flat_sites, range.text_offset, range.size);
      move_sites_in_range(function.ordinary_memory_sites, kernel->ordinary_memory_sites,
                          range.text_offset, range.size);
      move_sites_in_range(function.barrier_sites, kernel->barrier_sites, range.text_offset,
                          range.size);
      move_sites_in_range(function.fence_sites, kernel->fence_sites, range.text_offset, range.size);
      move_sites_in_range(function.atomic_sites, kernel->atomic_sites, range.text_offset,
                          range.size);
    }

    ConSanKernelInfo decoded_range;
    decoded_range.name = kernel->name;
    decoded_range.entry_text_offset = range.text_offset;
    decoded_range.text_file_offset = kernel->text_file_offset;
    decoded_range.code_size = range.size;
    decoded_range.has_text_range = true;
    decode_kernel_stats(code_object_bytes, decoder, arch, decoded_range, result.warnings);
    append_unique_sites(decoded_range.lds_sites, kernel->lds_sites);
    append_unique_sites(decoded_range.flat_sites, kernel->flat_sites);
    append_unique_sites(decoded_range.ordinary_memory_sites, kernel->ordinary_memory_sites);
    append_unique_sites(decoded_range.barrier_sites, kernel->barrier_sites);
    append_unique_sites(decoded_range.fence_sites, kernel->fence_sites);
    append_unique_sites(decoded_range.atomic_sites, kernel->atomic_sites);
  }
}

[[nodiscard]] ConSanTransformStage take_modified_stage(ConSanResult &result) {
  return {std::move(result.elf_bytes),
          std::move(result.patches),
          std::move(result.warnings),
          result.requested_fault_mutations,
          result.planned_fault_mutations,
          result.applied_fault_mutations,
          std::move(result.applied_fault_logical_identity)};
}

void prepend_stage_diagnostics(ConSanResult &result, ConSanTransformStage &stage) {
  result.warnings.insert(result.warnings.begin(), std::make_move_iterator(stage.warnings.begin()),
                         std::make_move_iterator(stage.warnings.end()));
  result.requested_fault_mutations = stage.requested_fault_mutations;
  result.planned_fault_mutations = stage.planned_fault_mutations;
  result.applied_fault_mutations = stage.applied_fault_mutations;
  result.applied_fault_logical_identity = std::move(stage.applied_fault_logical_identity);
}

[[nodiscard]] bool ranges_overlap(uint64_t first_offset, uint32_t first_size,
                                  uint64_t second_offset, uint32_t second_size) {
  return first_offset < second_offset + second_size && second_offset < first_offset + first_size;
}

[[nodiscard]] std::optional<std::vector<CarriedPerturbationPlan>>
carry_perturbation_plans_across_mutation(const ConSanResult &pristine_plan,
                                         const ConSanTransformStage &mutation_stage,
                                         std::string *error) {
  std::vector<CarriedPerturbationPlan> carried;
  carried.reserve(pristine_plan.perturbation_plans.size());
  for (const ConSanPerturbationPlan &plan : pristine_plan.perturbation_plans) {
    const auto candidate =
        std::ranges::find(pristine_plan.perturbation_candidates, plan.candidate_identity,
                          &ConSanPerturbationCandidate::identity);
    if (candidate == pristine_plan.perturbation_candidates.end() || !candidate->eligible ||
        candidate->sequence_identity != plan.sequence_identity ||
        candidate->anchor_text_offset != plan.anchor_text_offset ||
        candidate->anchor_size != plan.anchor_size || candidate->container_name.empty()) {
      *error = "selected pristine perturbation identity is incomplete or stale";
      return std::nullopt;
    }

    uint64_t translated_offset = plan.anchor_text_offset;
    bool overlaps_atomic_mutation = false;
    bool removed_cache_boundary = false;
    bool sequence_semantics_weakened = false;
    std::optional<uint64_t> source_owner_descriptor_file_offset;
    if (candidate->in_kernel) {
      const auto owner = std::ranges::find(pristine_plan.kernels, candidate->container_name,
                                           &ConSanKernelInfo::name);
      if (owner == pristine_plan.kernels.end()) {
        *error = "selected perturbation edge lost its pristine kernel owner";
        return std::nullopt;
      }
      source_owner_descriptor_file_offset = owner->descriptor_file_offset;
    }
    const auto dropped =
        std::ranges::find_if(mutation_stage.patches, [&](const ConSanPatchInfo &patch) {
          return patch.kind == ConSanPatchKind::InlineBarrierNopRewrite &&
                 ranges_overlap(plan.anchor_text_offset, plan.anchor_size, patch.anchor_offset,
                                patch.original_size);
        });
    if (dropped != mutation_stage.patches.end()) {
      *error = "selected perturbation edge was destroyed by the barrier drop";
      return std::nullopt;
    }

    std::vector<const ConSanPatchInfo *> move_sources;
    const ConSanPatchInfo *move_target = nullptr;
    for (const ConSanPatchInfo &patch : mutation_stage.patches) {
      if (patch.kind == ConSanPatchKind::InlineBarrierMoveSourceRewrite)
        move_sources.push_back(&patch);
      else if (patch.kind == ConSanPatchKind::InlineBarrierMoveTargetRewrite)
        move_target = &patch;
    }
    const auto selected_source =
        std::ranges::find_if(move_sources, [&](const ConSanPatchInfo *patch) {
          return ranges_overlap(plan.anchor_text_offset, plan.anchor_size, patch->anchor_offset,
                                patch->original_size);
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

    for (const ConSanPatchInfo &patch : mutation_stage.patches) {
      if (patch.phase == ConSanPatchPhase::Mutation &&
          patch.kind == ConSanPatchKind::InlineAtomicScopeRewrite) {
        const auto event = std::ranges::find_if(pristine_plan.sync_events, [&](const auto &item) {
          return item.container_name == candidate->container_name &&
                 item.in_kernel == candidate->in_kernel &&
                 item.text_offset == patch.anchor_offset && item.size == patch.original_size;
        });
        if (event != pristine_plan.sync_events.end() &&
            std::ranges::find(candidate->ordered_member_identities, event->identity) !=
                candidate->ordered_member_identities.end()) {
          sequence_semantics_weakened = true;
        }
      }
      if (patch.phase != ConSanPatchPhase::Mutation || patch.original_size == 0u ||
          !ranges_overlap(plan.anchor_text_offset, plan.anchor_size, patch.anchor_offset,
                          patch.original_size))
        continue;
      const bool exact_alias =
          plan.anchor_text_offset == patch.anchor_offset && plan.anchor_size == patch.original_size;
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

    carried.push_back({.source_candidate_identity = candidate->identity,
                       .source_sequence_identity = candidate->sequence_identity,
                       .container_name = candidate->container_name,
                       .in_kernel = candidate->in_kernel,
                       .basic_block_index = candidate->basic_block_index,
                       .anchor_event_identity = candidate->anchor_event_identity,
                       .ordered_member_identities = candidate->ordered_member_identities,
                       .source_anchor_text_offset = candidate->anchor_text_offset,
                       .translated_anchor_text_offset = translated_offset,
                       .anchor_size = candidate->anchor_size,
                       .source_owner_descriptor_file_offset = source_owner_descriptor_file_offset,
                       .kind = candidate->kind,
                       .edge = candidate->edge,
                       .sleep_imm = plan.sleep_imm,
                       .overlaps_atomic_mutation = overlaps_atomic_mutation,
                       .removed_cache_boundary = removed_cache_boundary,
                       .sequence_semantics_weakened = sequence_semantics_weakened});
  }
  return carried;
}

[[nodiscard]] ConSanResult rollback_composite_mutation(ConSanResult result, std::string reason) {
  result.outcome = ConSanTransformOutcome::Unsupported;
  result.modified = false;
  result.final_validation_passed = false;
  result.staged_composition_validated = false;
  result.elf_bytes.clear();
  result.patches.clear();
  result.applied_fault_mutations = 0;
  result.applied_perturbations = 0;
  result.warnings.emplace_back("ConSan rolled back staged fault/perturbation composition: " +
                               std::move(reason));
  return result;
}

} // namespace

const ConSanSyncSequence *find_consan_sync_sequence_for_event(const ConSanResult &result,
                                                              std::string_view event_identity) {
  const ConSanSyncSequence *match = nullptr;
  for (const ConSanSyncSequence &sequence : result.sync_sequences) {
    if (std::ranges::find(sequence.member_event_identities, event_identity) ==
        sequence.member_event_identities.end()) {
      continue;
    }
    if (match != nullptr)
      return nullptr;
    match = &sequence;
  }
  return match;
}

[[nodiscard]] ConSanResult
try_patch_consan_impl(std::span<const uint8_t> code_object_bytes, const ConSanOptions &options,
                      std::span<const CarriedPerturbationPlan> carried_plans = {}) {
  const bool has_fault =
      options.fault_drop_barrier || options.fault_move_barrier ||
      options.fault_mutate_barrier_id_scope || options.fault_mutate_barrier_participants ||
      options.fault_atomic_wrong_address || options.fault_atomic_weaken_order ||
      options.fault_atomic_weaken_scope || options.fault_ordinary_wrong_address ||
      options.fault_ordinary_weaken_order || options.fault_ordinary_weaken_scope;
  const bool has_perturbation = options.sc_perturb_kind != ConSanPerturbationKind::None;
  const bool has_atomic_fault = options.fault_atomic_wrong_address ||
                                options.fault_atomic_weaken_order ||
                                options.fault_atomic_weaken_scope;
  const bool has_ordinary_fault = options.fault_ordinary_wrong_address ||
                                  options.fault_ordinary_weaken_order ||
                                  options.fault_ordinary_weaken_scope;
  if (has_ordinary_fault && has_perturbation) {
    ConSanResult result;
    result.visited_code_object = true;
    result.input_size = code_object_bytes.size();
    result.flavor = options.flavor;
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan ordinary fault/perturbation composition is not yet qualified");
    return result;
  }
  if (has_atomic_fault && has_perturbation &&
      (options.sc_perturb_kind != ConSanPerturbationKind::Atomic || options.fault_drop_barrier ||
       options.fault_move_barrier || options.fault_mutate_barrier_id_scope ||
       options.fault_mutate_barrier_participants)) {
    ConSanResult result;
    result.visited_code_object = true;
    result.input_size = code_object_bytes.size();
    result.flavor = options.flavor;
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan atomic fault/perturbation composition requires the SC2D composite proof");
    return result;
  }
  if (has_fault && !options.fault_dry_run && !options.faults_preapplied) {
    ConSanOptions planning_options = options;
    planning_options.fault_dry_run = true;
    planning_options.faults_preapplied = false;
    if (!has_perturbation)
      planning_options.fault_require_exactly_one = false;
    ConSanResult pristine_mutation_plan =
        try_patch_consan_impl(code_object_bytes, planning_options);
    if (has_perturbation &&
        (!pristine_mutation_plan.errors.empty() || pristine_mutation_plan.fault_plans.empty() ||
         pristine_mutation_plan.perturbation_plans.empty())) {
      pristine_mutation_plan.outcome = ConSanTransformOutcome::Unsupported;
      pristine_mutation_plan.warnings.emplace_back(
          has_perturbation
              ? "ConSan composite planning did not select exact fault and perturbation plans"
              : "ConSan mutation planning did not select an exact fault plan");
      return pristine_mutation_plan;
    }
    // Apply corruption to a pristine image first, then inventory and instrument
    // that image. This is essential for address/order faults: probes must model
    // the instruction the GPU will actually execute.
    ConSanOptions fault_options = options;
    fault_options.flavor = ConSanFlavor::SuperCollider;
    fault_options.probe_nop = false;
    fault_options.probe_trampoline_nop = false;
    fault_options.probe_endpgm = false;
    fault_options.probe_lds_endpgm = false;
    fault_options.probe_lds_check_trap = false;
    fault_options.probe_flat_check_trap = false;
    fault_options.probe_flat_trap = false;
    // An explicitly authorized mutation must survive into the instrumentation
    // stage. Do not turn a deliberately dropped barrier half into the
    // ordinary-user malformed-wait abort.
    fault_options.abort_unmatched_barrier_wait = false;
    fault_options.sc_perturb_kind = ConSanPerturbationKind::None;
    fault_options.faults_preapplied = true;
    ConSanResult fault_result = try_patch_consan(code_object_bytes, fault_options);
    if (fault_result.outcome != ConSanTransformOutcome::ModifiedValid)
      return fault_result;
    ConSanTransformStage mutation_stage = take_modified_stage(fault_result);

    ConSanOptions instrument_options = options;
    instrument_options.fault_drop_barrier = false;
    instrument_options.fault_move_barrier = false;
    instrument_options.fault_mutate_barrier_id_scope = false;
    instrument_options.fault_mutate_barrier_participants = false;
    instrument_options.fault_atomic_wrong_address = false;
    instrument_options.fault_atomic_weaken_order = false;
    instrument_options.fault_atomic_weaken_scope = false;
    instrument_options.fault_ordinary_wrong_address = false;
    instrument_options.fault_ordinary_weaken_order = false;
    instrument_options.fault_ordinary_weaken_scope = false;
    // Cardinality was enforced on the mutation stage. The subsequent
    // instrumentation-only pass has no mutation of its own to count.
    instrument_options.fault_require_exactly_one = false;
    instrument_options.faults_preapplied = true;
    instrument_options.abort_unmatched_barrier_wait = false;
    for (const ConSanPatchInfo &patch : mutation_stage.patches) {
      if (patch.original_size != 0) {
        instrument_options.preapplied_reserved_ranges.push_back(
            {.text_offset = patch.anchor_offset, .size = patch.original_size});
      }
      if (patch.trampoline_size != 0) {
        instrument_options.preapplied_reserved_ranges.push_back(
            {.text_offset = patch.trampoline_offset, .size = patch.trampoline_size});
      }
      if (patch.kind != ConSanPatchKind::InlineBarrierMoveTargetRewrite ||
          patch.trampoline_size == 0)
        continue;
      const auto owner = std::ranges::find_if(fault_result.kernels, [&](const auto &kernel) {
        return kernel.has_text_range && patch.anchor_offset >= kernel.entry_text_offset &&
               patch.anchor_offset - kernel.entry_text_offset < kernel.code_size;
      });
      if (owner == fault_result.kernels.end()) {
        fault_result.errors.emplace_back(
            "ConSan markerless barrier move could not retain fault-cave kernel ownership");
        return fault_result;
      }
      instrument_options.preapplied_code_ranges.push_back(
          {.text_offset = patch.trampoline_offset,
           .size = patch.trampoline_size,
           .kernel_name = owner->name,
           .continuation_text_offset = patch.anchor_offset + patch.original_size});
    }
    std::vector<CarriedPerturbationPlan> staged_carried;
    if (has_perturbation) {
      std::string carry_error;
      auto carried = carry_perturbation_plans_across_mutation(pristine_mutation_plan,
                                                              mutation_stage, &carry_error);
      if (!carried)
        return rollback_composite_mutation(std::move(fault_result), std::move(carry_error));
      instrument_options.sc_perturb_identity.clear();
      instrument_options.sc_perturb_index = 0;
      staged_carried = std::move(*carried);
    }
    const std::span<const CarriedPerturbationPlan> instrument_carried = staged_carried;
    ConSanResult instrument_result = finalize_consan_result(
        try_patch_consan_impl(mutation_stage.image, instrument_options, instrument_carried),
        mutation_stage.image);
    // Resource ownership is a logical kernel identity, but instrumentation of
    // a staged mutation records descriptor offsets in the mutation image.
    // Normalize those offsets before merging stages: an intermediate offset
    // can otherwise collide with a different pristine kernel descriptor.
    AmdGpuCodeObject pristine_code_object(code_object_bytes.data(), code_object_bytes.size());
    AmdGpuCodeObject staged_code_object(mutation_stage.image.data(), mutation_stage.image.size());
    std::unordered_map<uint64_t, uint64_t> pristine_owner_offsets;
    for (const AmdGpuKernelInfo &staged_kernel : staged_code_object.kernels()) {
      const auto pristine_kernel = std::ranges::find(pristine_code_object.kernels(),
                                                     staged_kernel.name, &AmdGpuKernelInfo::name);
      if (pristine_kernel != pristine_code_object.kernels().end()) {
        pristine_owner_offsets.emplace(staged_kernel.descriptor_file_offset,
                                       pristine_kernel->descriptor_file_offset);
      }
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
    for (ConSanPatchInfo &patch : instrument_result.patches)
      normalize_owner_offsets(patch.owner_descriptor_file_offsets);
    prepend_stage_diagnostics(instrument_result, mutation_stage);
    switch (instrument_result.outcome) {
    case ConSanTransformOutcome::ModifiedValid:
      if (!fault_result.final_validation_passed || !instrument_result.final_validation_passed) {
        instrument_result.errors.emplace_back(
            "ConSan staged composition is missing an independently validated stage");
        instrument_result.outcome = ConSanTransformOutcome::Invalid;
        instrument_result.modified = false;
        instrument_result.elf_bytes.clear();
        instrument_result.patches.clear();
        instrument_result.applied_fault_mutations = 0;
        instrument_result.applied_fault_logical_identity.reset();
        return instrument_result;
      }
      instrument_result.staged_composition_validated = true;
      // Retain the exact applied mutation plan. The instrumentation pass
      // inventories the mutated image, so without this provenance a live
      // staged result loses which dry-run plan authorized the mutation.
      if (!pristine_mutation_plan.fault_plans.empty()) {
        instrument_result.planned_fault_mutations = pristine_mutation_plan.planned_fault_mutations;
        instrument_result.fault_plans = pristine_mutation_plan.fault_plans;
      }
      if (has_perturbation) {
        instrument_result.requested_perturbations = pristine_mutation_plan.requested_perturbations;
        instrument_result.planned_perturbations = pristine_mutation_plan.planned_perturbations;
      }
      instrument_result.patches.insert(instrument_result.patches.begin(),
                                       std::make_move_iterator(mutation_stage.patches.begin()),
                                       std::make_move_iterator(mutation_stage.patches.end()));
      return instrument_result;
    case ConSanTransformOutcome::Unchanged:
      // A no-target instrumentation stage must not discard an otherwise valid
      // mutation. Its inventory describes the staged image being committed.
      if (!pristine_mutation_plan.fault_plans.empty()) {
        instrument_result.planned_fault_mutations = pristine_mutation_plan.planned_fault_mutations;
        instrument_result.fault_plans = pristine_mutation_plan.fault_plans;
      }
      instrument_result.elf_bytes = std::move(mutation_stage.image);
      instrument_result.patches = std::move(mutation_stage.patches);
      instrument_result.modified = true;
      return instrument_result;
    case ConSanTransformOutcome::Unsupported:
      if (instrument_options.flavor == ConSanFlavor::SuperCollider &&
          !instrument_options.probe_nop && !instrument_options.probe_trampoline_nop &&
          !instrument_options.probe_endpgm && !instrument_options.probe_lds_endpgm &&
          !instrument_options.probe_lds_check_trap && !instrument_options.probe_flat_check_trap &&
          !instrument_options.probe_flat_trap &&
          instrument_options.sc_perturb_kind == ConSanPerturbationKind::None) {
        // The second pass may classify the intentionally weakened image as a
        // preflight skip. With no instrumentation request, that is not a
        // reason to discard an independently validated mutation stage.
        instrument_result.outcome = ConSanTransformOutcome::Unchanged;
        if (!pristine_mutation_plan.fault_plans.empty()) {
          instrument_result.planned_fault_mutations =
              pristine_mutation_plan.planned_fault_mutations;
          instrument_result.fault_plans = pristine_mutation_plan.fault_plans;
        }
        instrument_result.elf_bytes = std::move(mutation_stage.image);
        instrument_result.patches = std::move(mutation_stage.patches);
        instrument_result.modified = true;
        return instrument_result;
      }
      instrument_result.warnings.emplace_back(
          "ConSan rolled back staged fault mutation because instrumentation is unsupported");
      instrument_result.applied_fault_mutations = 0;
      instrument_result.applied_fault_logical_identity.reset();
      break;
    case ConSanTransformOutcome::Invalid:
      instrument_result.warnings.emplace_back(
          "ConSan rolled back staged fault mutation because instrumentation is invalid");
      instrument_result.applied_fault_mutations = 0;
      instrument_result.applied_fault_logical_identity.reset();
      break;
    }
    instrument_result.elf_bytes.clear();
    instrument_result.patches.clear();
    instrument_result.modified = false;
    return instrument_result;
  }

  ConSanResult result;
  result.visited_code_object = true;
  result.flavor = options.flavor;
  result.moi_engine = options.moi_engine;
  result.carried_composite_instrumentation_stage = !carried_plans.empty();
  result.input_size = code_object_bytes.size();
  result.requested_fault_mutations = requested_fault_mutation_count(options);

  if (options.flavor == ConSanFlavor::None)
    return result;

  if (code_object_bytes.empty()) {
    result.errors.emplace_back("ConSan received an empty code object");
    return result;
  }

  AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
  if (!code_object.is_valid()) {
    result.errors.emplace_back("ConSan could not parse AMDGPU code object");
    return result;
  }
  result.errors = validate_consan_input_layout(code_object);
  if (!result.errors.empty())
    return result;

  result.target_name = target_name(code_object.target_id());
  result.arch_name = arch_name(code_object.target_id());

  for (const Section *section : code_object.text_sections()) {
    ConSanTextSection info;
    info.name = section->name();
    info.file_offset = section->sectionOffset();
    info.virtual_address = section->vaddr();
    info.size = section->size();
    result.text_sections.push_back(std::move(info));
  }

  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    ConSanKernelInfo info;
    info.name = kernel.name;
    info.descriptor_file_offset = kernel.descriptor_file_offset;
    info.entry_text_offset = kernel.entry_text_offset;
    info.text_file_offset = kernel.text_file_offset;
    info.code_size = kernel.code_size;
    info.code_size_inferred_from_zero = kernel.code_size_inferred_from_zero;
    info.has_text_range = kernel.has_text_range;
    info.has_dynamic_lds = kernel.has_dynamic_lds;
    info.uses_dynamic_stack = kernel.uses_dynamic_stack;
    info.sgpr_count = kernel.sgpr_count;
    info.required_workgroup_size = kernel.required_workgroup_size;
    result.kernels.push_back(std::move(info));
  }

  std::unordered_set<std::string> kernel_names;
  kernel_names.reserve(code_object.kernels().size());
  for (const AmdGpuKernelInfo &kernel : code_object.kernels())
    kernel_names.insert(kernel.name);

  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    if (kernel_names.contains(function.name))
      continue;
    ConSanFunctionInfo info;
    info.name = function.name;
    info.entry_text_offset = function.entry_text_offset;
    info.text_file_offset = function.text_file_offset;
    info.code_size = function.code_size;
    info.code_size_inferred_from_zero = function.code_size_inferred_from_zero;
    result.functions.push_back(std::move(info));
  }

  if (result.text_sections.empty())
    result.warnings.emplace_back("ConSan found no .text sections");
  if (result.kernels.empty())
    result.warnings.emplace_back("ConSan found no kernel descriptor symbols");

  const rj_code_arch_t arch = arch_for_target(code_object.target_id());
  if (arch == ROCJITSU_CODE_ARCH_INVALID) {
    result.warnings.emplace_back("ConSan does not support target '" + result.target_name + "'");
    result.outcome = ConSanTransformOutcome::Unsupported;
    return result;
  }

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    result.warnings.emplace_back("ConSan could not create decoder for arch '" + result.arch_name +
                                 "'");
    result.outcome = ConSanTransformOutcome::Unsupported;
    return result;
  }

  for (ConSanKernelInfo &kernel : result.kernels) {
    decode_kernel_stats(code_object_bytes, *decoder, arch, kernel, result.warnings);
  }
  for (ConSanFunctionInfo &function : result.functions)
    decode_function_stats(code_object_bytes, *decoder, arch, function, result.warnings);
  const bool has_decode_error =
      std::ranges::any_of(
          result.kernels,
          [](const ConSanKernelInfo &kernel) { return kernel.stats.decode_error_count != 0u; }) ||
      std::ranges::any_of(result.functions, [](const ConSanFunctionInfo &function) {
        return function.stats.decode_error_count != 0u;
      });
  if (has_decode_error) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan stopped before CFG analysis because an instruction could not be decoded");
    return result;
  }
  relay_flat_pointer_provenance_across_calls(code_object_bytes, code_object, arch, result.kernels,
                                             result.functions, result.warnings);
  if (options.flavor == ConSanFlavor::SuperCollider && !options.fault_dry_run &&
      options.sc_perturb_kind == ConSanPerturbationKind::None) {
    for (ConSanKernelInfo &kernel : result.kernels)
      preflight_kernel(kernel, options, result.errors, result.warnings);
  }
  reattribute_preapplied_code_ranges(code_object_bytes, *decoder, arch, options, result);
  if (!result.errors.empty())
    return result;
  prune_unreachable_inferred_ranges(code_object, *decoder, arch, options, result);
  const bool needs_semantic_inventory = options.flavor == ConSanFlavor::Moi ||
                                        options.fault_dry_run || has_fault || has_perturbation ||
                                        options.abort_unmatched_barrier_wait ||
                                        options.collect_barrier_move_destinations;
  if (needs_semantic_inventory) {
    build_fault_site_inventory(code_object_bytes, result);
    build_sync_event_inventory(code_object_bytes, result);
    build_singleton_sync_sequences(code_object, *decoder, arch, result);
    const bool allow_extended_barrier_pairs =
        options.fault_drop_barrier || options.fault_move_barrier ||
        options.fault_mutate_barrier_id_scope || options.fault_mutate_barrier_participants ||
        options.abort_unmatched_barrier_wait ||
        (options.moi_engine == ConSanMoiEngine::Sampled && options.moi_track_barriers);
    associate_barrier_sync_sequences(code_object, *decoder, arch, allow_extended_barrier_pairs,
                                     result);
    associate_atomic_sync_sequences(code_object, *decoder, arch, result);
    build_barrier_lifecycle_group_inventory(result);
    if (options.collect_barrier_move_destinations)
      build_barrier_move_destination_inventory(code_object, *decoder, arch, result);
    annotate_execution_owners(code_object, *decoder, arch, options, result);
    associate_ordinary_release_sync_sequences(code_object, *decoder, arch, result);
    associate_ordinary_acquire_sync_sequences(code_object, *decoder, arch, result);
    build_communication_address_recipes(code_object, *decoder, arch, result);
    build_moi_fence_candidate_inventory(result);
    build_perturbation_candidate_inventory(result);
    annotate_fault_sites_with_sync_sequences(result);
  } else if (std::ranges::any_of(result.functions, [](const ConSanFunctionInfo &function) {
               return !function.flat_sites.empty();
             })) {
    // SuperCollider's ordinary clean transform does not consume synchronization
    // semantics. Shared-function flat sites still need kernel ownership, while
    // kernel-local sites carry their descriptor directly.
    annotate_execution_owners(code_object, *decoder, arch, options, result);
  }
  if (options.fault_dry_run) {
    build_fault_mutation_plan(options, result);
    build_perturbation_plan(options, result, carried_plans);
    return result;
  }
  if (has_perturbation) {
    build_perturbation_plan(options, result, carried_plans);
    if (!result.errors.empty())
      return result;
  }

  if (options.flavor == ConSanFlavor::Moi)
    return try_patch_consan_moi(std::move(result), options, code_object_bytes, arch);

  if (options.flavor != ConSanFlavor::SuperCollider) {
    result.errors.emplace_back("ConSan selected unknown flavor");
    return result;
  }

  const bool has_redundant_access = options.probe_lds_check_trap || options.probe_flat_check_trap;
  const uint32_t overall_patch_budget = std::max<uint32_t>(options.max_patches, 1u);
  if (result.perturbation_plans.size() > overall_patch_budget) {
    result.errors.emplace_back("ConSan SC perturbation exceeds the shared patch budget");
    return result;
  }
  ConSanOptions redundant_options = options;
  redundant_options.max_patches =
      overall_patch_budget - static_cast<uint32_t>(result.perturbation_plans.size());

  if (has_redundant_access && redundant_options.max_patches != 0u) {
    if (options.probe_lds_check_trap)
      try_apply_lds_load_check_trap_patch(code_object, arch, redundant_options, result);
    if (options.probe_flat_check_trap && result.errors.empty()) {
      if (!result.modified) {
        try_apply_flat_check_trap_patch(code_object, arch, redundant_options, result);
      } else {
        const uint32_t max_patches = redundant_options.max_patches;
        const uint32_t access_patches = static_cast<uint32_t>(
            std::ranges::count_if(result.patches, [](const ConSanPatchInfo &patch) {
              switch (patch.kind) {
              case ConSanPatchKind::InlineLdsLoadCheckTrap:
              case ConSanPatchKind::InlineLdsStoreCheckTrap:
              case ConSanPatchKind::LocalCaveLdsLoadCheckTrap:
              case ConSanPatchKind::LocalCaveLdsStoreCheckTrap:
              case ConSanPatchKind::InlineFlatLoadCheckTrap:
              case ConSanPatchKind::InlineFlatStoreCheckTrap:
              case ConSanPatchKind::LocalCaveFlatLoadCheckTrap:
              case ConSanPatchKind::LocalCaveFlatStoreCheckTrap:
                return true;
              default:
                return false;
              }
            }));
        if (access_patches >= max_patches && redundant_options.max_patches_is_expert_limit) {
          result.warnings.emplace_back(
              "ConSan flat check/trap proof skipped after native LDS patching: "
              "patch budget already consumed");
        } else if (auto reserved_ranges =
                       reserved_ranges_for_existing_patches(code_object, result)) {
          ConSanOptions flat_options = redundant_options;
          if (flat_options.max_patches_is_expert_limit)
            flat_options.max_patches = max_patches - access_patches;
          try_apply_flat_check_trap_patch(code_object, arch, flat_options, result,
                                          *reserved_ranges);
        } else {
          result.warnings.emplace_back(
              "ConSan flat check/trap proof skipped after native LDS patching: "
              "existing patch ranges could not be mapped in the original code object");
        }
      }
    }
  } else if (has_redundant_access) {
    result.warnings.emplace_back(
        "ConSan redundant-access proof skipped: shared patch budget consumed by perturbation");
  } else if (!has_perturbation && options.probe_flat_trap)
    try_apply_flat_trap_patch(code_object, arch, options, result);
  else if (!has_perturbation && options.probe_lds_endpgm)
    try_apply_lds_endpgm_patch(code_object, arch, result);
  else if (!has_perturbation && options.probe_endpgm)
    try_apply_proof_endpgm_patch(code_object, arch, result);
  else if (!has_perturbation && (options.probe_nop || options.probe_trampoline_nop))
    try_apply_proof_nop_patch(code_object, arch, options.probe_trampoline_nop, result);

  if (has_perturbation && result.errors.empty()) {
    try_apply_perturbation_patches(code_object, arch, result);
    if (result.applied_perturbations != result.perturbation_plans.size()) {
      result.modified = false;
      result.elf_bytes.clear();
      result.patches.clear();
    }
  }

  if (options.fault_drop_barrier)
    try_apply_barrier_drop_fault_patch(code_object, arch, options, result);
  if (options.fault_move_barrier && result.errors.empty())
    try_apply_barrier_move_fault_patch(code_object, arch, options, result);
  if (options.fault_mutate_barrier_id_scope && result.errors.empty())
    try_apply_barrier_id_scope_fault_patch(code_object, options, result);
  if (options.fault_mutate_barrier_participants && result.errors.empty() &&
      result.outcome != ConSanTransformOutcome::Unsupported)
    try_apply_barrier_participant_fault_patch(code_object, options, result);
  if ((options.fault_atomic_wrong_address || options.fault_atomic_weaken_order ||
       options.fault_atomic_weaken_scope) &&
      result.errors.empty())
    try_apply_atomic_fault_patch(code_object, options, result);
  if ((options.fault_ordinary_wrong_address || options.fault_ordinary_weaken_order ||
       options.fault_ordinary_weaken_scope) &&
      result.errors.empty())
    try_apply_ordinary_fault_patch(code_object, options, result);
  result.applied_fault_mutations = applied_fault_mutation_count(result);
  if (options.fault_require_exactly_one && result.outcome != ConSanTransformOutcome::Unsupported &&
      result.applied_fault_mutations != 1u) {
    result.errors.emplace_back(
        "ConSan fault injection required exactly one applied mutation, got " +
        std::to_string(result.applied_fault_mutations));
  }

  return result;
}
