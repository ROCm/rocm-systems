namespace {

/// Owns the independently validated fault-corruption image while the ordinary
/// instrumentation pass analyzes and lowers against that image. This stage is
/// intentionally fault-only: perturbation is planned against the pristine
/// image and applied by the subsequent instrumentation pass, so carrying a
/// complete `ConSanMutationOutcome` here would give this stage false ownership
/// of perturbation facts and permit it to erase the later pass's outcome.
struct ConSanFaultTransformStage {
  std::vector<uint8_t> image;
  std::vector<ConSanPatchInfo> patches;
  std::vector<std::string> warnings;
  ConSanMutationTally fault;
  std::optional<std::string> logical_identity;
};

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

[[nodiscard]] ConSanResult finalize_consan_result(ConSanResult result,
                                                  std::span<const uint8_t> original_bytes,
                                                  uint64_t expected_moi_report_dispatch_id = 0,
                                                  bool validating_carried_composite_stage = false);

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
                                        const ConSanOptions &options,
                                        ProgramInventoryBuilder &inventory, ConSanResult &result) {
  std::span<ConSanKernelInfo> kernels = inventory.kernels();
  std::span<ConSanFunctionInfo> functions = inventory.functions();
  for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
    const auto kernel = std::ranges::find_if(
        kernels, [&](const ConSanKernelInfo &item) { return item.name == range.kernel_name; });
    if (kernel == kernels.end()) {
      result.errors.emplace_back("ConSan could not recover the owner of a preapplied fault cave");
      continue;
    }
    for (ConSanKernelInfo &other : kernels) {
      if (&other == &*kernel)
        continue;
      move_sites_in_range(other.ordinary_memory_sites, kernel->ordinary_memory_sites,
                          range.text_offset, range.size);
      move_sites_in_range(other.barrier_sites, kernel->barrier_sites, range.text_offset,
                          range.size);
      move_sites_in_range(other.fence_sites, kernel->fence_sites, range.text_offset, range.size);
      move_sites_in_range(other.atomic_sites, kernel->atomic_sites, range.text_offset, range.size);
    }
    for (ConSanFunctionInfo &function : functions) {
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
    std::vector<ConSanAccessInventorySite> decoded_accesses;
    decode_kernel_stats(code_object_bytes, decoder, arch, decoded_range, decoded_accesses,
                        result.warnings);
    inventory.reattribute_access_range(range.text_offset, range.size, *kernel, decoded_accesses,
                                       code_object_bytes);
    append_unique_sites(decoded_range.ordinary_memory_sites, kernel->ordinary_memory_sites);
    append_unique_sites(decoded_range.barrier_sites, kernel->barrier_sites);
    append_unique_sites(decoded_range.fence_sites, kernel->fence_sites);
    append_unique_sites(decoded_range.atomic_sites, kernel->atomic_sites);
  }
}

[[nodiscard]] ConSanFaultTransformStage take_modified_stage(ConSanTransformArtifacts &result) {
  return {std::move(result.replacement), std::move(result.patches), std::move(result.warnings),
          result.mutation.fault, std::move(result.mutation.applied_fault_logical_identity)};
}

[[nodiscard]] ConSanResult resume_composite_lowering(ConSanTransformArtifacts artifacts) {
  ConSanResult result;
  static_cast<ConSanTransformArtifacts &>(result) = std::move(artifacts);
  return result;
}

void prepend_stage_diagnostics(ConSanResult &result, ConSanFaultTransformStage &stage) {
  result.warnings.insert(result.warnings.begin(), std::make_move_iterator(stage.warnings.begin()),
                         std::make_move_iterator(stage.warnings.end()));
  result.mutation.fault = stage.fault;
  result.mutation.applied_fault_logical_identity = std::move(stage.logical_identity);
}

[[nodiscard]] bool ranges_overlap(uint64_t first_offset, uint32_t first_size,
                                  uint64_t second_offset, uint32_t second_size) {
  return first_offset < second_offset + second_size && second_offset < first_offset + first_size;
}

[[nodiscard]] std::optional<std::vector<CarriedPerturbationPlan>>
carry_perturbation_plans_across_mutation(const ConSanResult &pristine_plan,
                                         const ConSanFaultTransformStage &mutation_stage,
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
      const ConSanKernelInfo *owner =
          pristine_plan.program_inventory.find_kernel_by_name(candidate->container_name);
      if (owner == nullptr) {
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
        const auto event = std::ranges::find_if(
            pristine_plan.program_inventory.sync().sync_events, [&](const auto &item) {
              return item.container_name == candidate->container_name &&
                     item.in_kernel == candidate->in_kernel &&
                     item.text_offset == patch.anchor_offset && item.size == patch.original_size;
            });
        if (event != pristine_plan.program_inventory.sync().sync_events.end() &&
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
  result.discard_candidate_modification();
  result.mutation.fault.applied = 0;
  result.mutation.perturbation.applied = 0;
  result.warnings.emplace_back("ConSan rolled back staged fault/perturbation composition: " +
                               std::move(reason));
  return result;
}

} // namespace

namespace {

void reset_inventory_for_fault_planning(const ConSanOptions &options, ConSanResult &result) {
  // The mutation stage must observe the reused result exactly as if semantic
  // analysis and fault planning had run, without inheriting a MOI lowering
  // outcome or any mutation-stage state from the sizing pass.
  result.warnings.clear();
  result.outcome = ConSanTransformOutcome::Unchanged;
  result.mutation.fault.requested = requested_fault_mutation_count(options);
  result.mutation.fault.planned = 0;
  result.mutation.fault.applied = 0;
  result.mutation.applied_fault_logical_identity.reset();
  result.fault_plans.clear();
}

void apply_fault_mutation_to_inventory(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                       const ConSanOptions &options, ConSanResult &result) {
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
    try_apply_atomic_fault_patch(code_object, arch, options, result);
  if (options.fault_lds_wrong_address && result.errors.empty())
    try_apply_lds_fault_patch(code_object, arch, options, result);
  if ((options.fault_ordinary_wrong_address || options.fault_ordinary_weaken_order ||
       options.fault_ordinary_weaken_scope) &&
      result.errors.empty())
    try_apply_ordinary_fault_patch(code_object, options, result);
  result.mutation.fault.applied = applied_fault_mutation_count(result);
  if (options.fault_require_exactly_one && result.outcome != ConSanTransformOutcome::Unsupported &&
      result.mutation.fault.applied != 1u) {
    result.errors.emplace_back(
        "ConSan fault injection required exactly one applied mutation, got " +
        std::to_string(result.mutation.fault.applied));
  }
}

} // namespace

[[nodiscard]] ConSanResult
try_patch_consan_impl(std::span<const uint8_t> code_object_bytes, const ConSanOptions &options,
                      std::span<const CarriedPerturbationPlan> carried_plans = {},
                      std::optional<ConSanResult> pristine_inventory = std::nullopt) {
  const major_image_ownership::ScopedPhase incremental_phase(
      major_image_ownership::Phase::IncrementalPatch, /*only_if_none=*/true);
  ConSanResult result;
  result.mutation.fault.requested = requested_fault_mutation_count(options);
  ProgramInventoryBuilder inventory_builder(code_object_bytes);
  result.program_inventory = inventory_builder.view();
  const bool has_fault = options.fault_drop_barrier || options.fault_move_barrier ||
                         options.fault_mutate_barrier_id_scope ||
                         options.fault_mutate_barrier_participants ||
                         options.fault_atomic_wrong_address || options.fault_atomic_weaken_order ||
                         options.fault_atomic_weaken_scope || options.fault_lds_wrong_address ||
                         options.fault_ordinary_wrong_address ||
                         options.fault_ordinary_weaken_order || options.fault_ordinary_weaken_scope;
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
  if (has_fault && !options.fault_dry_run && !options.faults_preapplied) {
    ConSanOptions planning_options = options;
    planning_options.fault_dry_run = true;
    planning_options.faults_preapplied = false;
    if (!has_perturbation)
      planning_options.fault_require_exactly_one = false;
    ConSanResult pristine_mutation_plan;
    if (pristine_inventory) {
      pristine_mutation_plan = std::move(*pristine_inventory);
      reset_inventory_for_fault_planning(planning_options, pristine_mutation_plan);
      build_fault_mutation_plan(planning_options, pristine_mutation_plan.program_inventory.arch(),
                                pristine_mutation_plan);
    } else {
      pristine_mutation_plan = try_patch_consan_impl(code_object_bytes, planning_options);
    }
    const size_t retained_planned_faults = pristine_mutation_plan.mutation.fault.planned;
    const std::vector<ConSanFaultMutationPlan> retained_fault_plans =
        pristine_mutation_plan.fault_plans;
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
    ConSanResult fault_result;
    if (has_perturbation) {
      fault_result = resume_composite_lowering(lower_consan(code_object_bytes, fault_options));
    } else {
      // Planning already built the complete pristine semantic inventory.
      // Reuse it to apply the selected mutation instead of decoding and
      // analyzing a large code object a second time. Final validation below
      // still re-derives mutation semantics from pristine executable bytes,
      // and the instrumentation stage independently analyzes the mutated
      // image.
      fault_result = std::move(pristine_mutation_plan);
      AmdGpuCodeObject code_object(code_object_bytes.data(), code_object_bytes.size());
      const rj_code_arch_t arch = consan_arch_for_target(code_object.target_id());
      apply_fault_mutation_to_inventory(code_object, arch, fault_options, fault_result);
      fault_result = resume_composite_lowering(finalize_consan_result(
          std::move(fault_result), code_object_bytes, fault_options.moi_report_dispatch_id));
    }
    if (fault_result.outcome != ConSanTransformOutcome::ModifiedValid)
      return fault_result;
    ConSanFaultTransformStage mutation_stage = take_modified_stage(fault_result);
    const major_image_ownership::ScopedOwner mutation_image_owner(
        major_image_ownership::OwnerKind::CompositeImage, mutation_stage.image);
    const major_image_ownership::ScopedPhase composite_phase(
        major_image_ownership::Phase::CompositeIncrementalPatch);

    ConSanOptions instrument_options = options;
    instrument_options.fault_drop_barrier = false;
    instrument_options.fault_move_barrier = false;
    instrument_options.fault_mutate_barrier_id_scope = false;
    instrument_options.fault_mutate_barrier_participants = false;
    instrument_options.fault_atomic_wrong_address = false;
    instrument_options.fault_atomic_weaken_order = false;
    instrument_options.fault_atomic_weaken_scope = false;
    instrument_options.fault_lds_wrong_address = false;
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
        mutation_stage.image, instrument_options.moi_report_dispatch_id,
        /*validating_carried_composite_stage=*/!instrument_carried.empty());
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
      AmdGpuCodeObject staged_code_object(mutation_stage.image.data(), mutation_stage.image.size());
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
    for (ConSanPatchInfo &patch : instrument_result.patches)
      normalize_owner_offsets(patch.owner_descriptor_file_offsets);
    prepend_stage_diagnostics(instrument_result, mutation_stage);
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
                                       std::make_move_iterator(mutation_stage.patches.begin()),
                                       std::make_move_iterator(mutation_stage.patches.end()));
      return instrument_result;
    case ConSanTransformOutcome::Unchanged:
      // A no-target instrumentation stage must not discard an otherwise valid
      // mutation. Its inventory describes the staged image being committed.
      if (!retained_fault_plans.empty()) {
        instrument_result.mutation.fault.planned = retained_planned_faults;
        instrument_result.fault_plans = retained_fault_plans;
      }
      instrument_result.replacement = std::move(mutation_stage.image);
      instrument_result.patches = std::move(mutation_stage.patches);
      instrument_result.mark_modified();
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
        if (!retained_fault_plans.empty()) {
          instrument_result.mutation.fault.planned = retained_planned_faults;
          instrument_result.fault_plans = retained_fault_plans;
        }
        instrument_result.replacement = std::move(mutation_stage.image);
        instrument_result.patches = std::move(mutation_stage.patches);
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

  const major_image_ownership::ScopedOwner result_owner(
      major_image_ownership::OwnerKind::ResultImage, result.replacement);

  if (options.flavor == ConSanFlavor::None)
    return result;

  if (code_object_bytes.empty()) {
    result.errors.emplace_back("ConSan received an empty code object");
    return result;
  }
  auto code_object =
      std::make_unique<AmdGpuCodeObject>(code_object_bytes.data(), code_object_bytes.size());
  if (!code_object->is_valid()) {
    result.errors.emplace_back("ConSan could not parse AMDGPU code object");
    return result;
  }
  const bool kernel_metadata_trustworthy = code_object->kernel_metadata_is_trustworthy();
  const size_t malformed_kernel_metadata_note_count =
      code_object->malformed_kernel_metadata_note_count();
  const rj_code_target_id_t target = code_object->target_id();
  const rj_code_arch_t arch = consan_arch_for_target(target);
  inventory_builder.set_code_object_facts(kernel_metadata_trustworthy,
                                          malformed_kernel_metadata_note_count, arch, target);
  result.errors = validate_consan_input_layout(*code_object);
  if (!result.errors.empty())
    return result;

  for (const Section *section : code_object->text_sections()) {
    ConSanTextSection info;
    info.name = section->name();
    info.file_offset = section->sectionOffset();
    info.virtual_address = section->vaddr();
    info.size = section->size();
    inventory_builder.text_sections().push_back(std::move(info));
  }

  for (const AmdGpuKernelInfo &kernel : code_object->kernels()) {
    ConSanKernelInfo info;
    info.name = kernel.name;
    info.descriptor_file_offset = kernel.descriptor_file_offset;
    if (kernel.descriptor_file_offset <= code_object_bytes.size() &&
        sizeof(rocr::llvm::amdhsa::kernel_descriptor_t) <=
            code_object_bytes.size() - kernel.descriptor_file_offset) {
      rocr::llvm::amdhsa::kernel_descriptor_t descriptor{};
      std::memcpy(&descriptor, code_object_bytes.data() + kernel.descriptor_file_offset,
                  sizeof(descriptor));
      info.declared_group_segment_bytes = descriptor.group_segment_fixed_size;
    }
    info.entry_text_offset = kernel.entry_text_offset;
    info.text_file_offset = kernel.text_file_offset;
    info.code_size = kernel.code_size;
    info.code_size_inferred_from_zero = kernel.code_size_inferred_from_zero;
    info.has_text_range = kernel.has_text_range;
    info.has_dynamic_lds = kernel.has_dynamic_lds;
    info.uses_dynamic_stack = kernel.uses_dynamic_stack;
    if (kernel_metadata_trustworthy) {
      info.vgpr_count = kernel.vgpr_count;
      info.agpr_count = kernel.agpr_count;
    }
    info.sgpr_count = kernel.sgpr_count;
    info.required_workgroup_size = kernel.required_workgroup_size;
    inventory_builder.kernels().push_back(std::move(info));
  }

  std::unordered_set<std::string> kernel_names;
  kernel_names.reserve(code_object->kernels().size());
  for (const AmdGpuKernelInfo &kernel : code_object->kernels())
    kernel_names.insert(kernel.name);

  for (const AmdGpuFunctionInfo &function : code_object->functions()) {
    if (kernel_names.contains(function.name))
      continue;
    ConSanFunctionInfo info;
    info.name = function.name;
    info.entry_text_offset = function.entry_text_offset;
    info.text_file_offset = function.text_file_offset;
    info.code_size = function.code_size;
    info.code_size_inferred_from_zero = function.code_size_inferred_from_zero;
    inventory_builder.functions().push_back(std::move(info));
  }

  result.program_inventory = inventory_builder.view();
  const auto publish_access_inventory = [&] {
    inventory_builder.publish_decoded_accesses(code_object_bytes);
    result.program_inventory = inventory_builder.view();
  };

  if (result.program_inventory.text_sections().empty())
    result.warnings.emplace_back("ConSan found no .text sections");
  if (result.program_inventory.kernels().empty())
    result.warnings.emplace_back("ConSan found no kernel descriptor symbols");

  if (arch == ROCJITSU_CODE_ARCH_INVALID) {
    publish_access_inventory();
    result.warnings.emplace_back("ConSan does not support target '" +
                                 std::string(rj_code_target_name(target)) + "'");
    result.outcome = ConSanTransformOutcome::Unsupported;
    return result;
  }
  inventory_builder.set_semantic_arch_required(true);

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    publish_access_inventory();
    result.warnings.emplace_back("ConSan could not create decoder for arch '" +
                                 std::string(rj_code_arch_name(arch)) + "'");
    result.outcome = ConSanTransformOutcome::Unsupported;
    return result;
  }

  for (ConSanKernelInfo &kernel : inventory_builder.kernels()) {
    decode_kernel_stats(code_object_bytes, *decoder, arch, kernel, inventory_builder.access_sites(),
                        result.warnings);
  }
  std::vector<ConSanAccessInventorySite> function_accesses;
  for (ConSanFunctionInfo &function : inventory_builder.functions())
    decode_function_stats(code_object_bytes, *decoder, arch, function, function_accesses,
                          result.warnings);
  const auto append_function_accesses = [&] {
    inventory_builder.access_sites().insert(inventory_builder.access_sites().end(),
                                            std::make_move_iterator(function_accesses.begin()),
                                            std::make_move_iterator(function_accesses.end()));
    function_accesses.clear();
  };
  const bool has_decode_error =
      std::ranges::any_of(
          result.program_inventory.kernels(),
          [](const ConSanKernelInfo &kernel) { return kernel.stats.decode_error_count != 0u; }) ||
      std::ranges::any_of(result.program_inventory.functions(),
                          [](const ConSanFunctionInfo &function) {
                            return function.stats.decode_error_count != 0u;
                          });
  if (has_decode_error) {
    append_function_accesses();
    publish_access_inventory();
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan stopped before CFG analysis because an instruction could not be decoded");
    return result;
  }
  relay_flat_pointer_provenance_across_calls(
      code_object_bytes, *code_object, arch, inventory_builder.kernels(),
      inventory_builder.functions(), function_accesses, result.warnings);
  append_function_accesses();
  publish_access_inventory();
  if (options.flavor == ConSanFlavor::SuperCollider && !options.fault_dry_run &&
      options.sc_perturb_kind == ConSanPerturbationKind::None) {
    for (ConSanKernelInfo &kernel : inventory_builder.kernels())
      preflight_kernel(kernel, result.warnings);
  }
  reattribute_preapplied_code_ranges(code_object_bytes, *decoder, arch, options, inventory_builder,
                                     result);
  if (!result.errors.empty()) {
    return result;
  }
  prune_unreachable_inferred_ranges(*code_object, *decoder, arch, options,
                                    inventory_builder.kernels(), inventory_builder.functions(),
                                    inventory_builder.access_sites());
  const bool needs_semantic_inventory = options.flavor == ConSanFlavor::Moi ||
                                        options.fault_dry_run || has_fault || has_perturbation ||
                                        options.abort_unmatched_barrier_wait ||
                                        options.collect_barrier_move_destinations;
  SynchronizationInventoryBuildView synchronization_inventory = inventory_builder.synchronization();
  if (needs_semantic_inventory) {
    build_fault_site_inventory(code_object_bytes, arch, result);
    build_sync_event_inventory(code_object_bytes, synchronization_inventory, result);
    if (options.flavor == ConSanFlavor::Moi &&
        !canonicalize_sync_events_by_physical_site(synchronization_inventory, result)) {
      result.program_inventory = inventory_builder.view();
      return result;
    }
    const SyncEventIdentityIndex sync_events =
        build_sync_event_identity_index(synchronization_inventory.sync_events);
    const auto sync_blocks = build_sync_basic_blocks(*code_object, *decoder, arch, result);
    build_singleton_sync_sequences(sync_blocks, synchronization_inventory);
    associate_barrier_sync_sequences(sync_blocks, consan_qualifies_extended_barrier_pairs(options),
                                     synchronization_inventory);
    associate_atomic_sync_sequences(sync_blocks, arch, sync_events, synchronization_inventory,
                                    result);
    build_barrier_lifecycle_group_inventory(synchronization_inventory);
    if (options.collect_barrier_move_destinations)
      build_barrier_move_destination_inventory(*code_object, sync_blocks, sync_events, arch,
                                               synchronization_inventory.sync_sequences, result);
    annotate_execution_owners(*code_object, *decoder, arch, options, &sync_blocks,
                              inventory_builder.access_sites(), synchronization_inventory, result);
    associate_ordinary_release_sync_sequences(sync_blocks, arch, sync_events,
                                              synchronization_inventory);
    associate_ordinary_acquire_sync_sequences(sync_blocks, sync_events, arch,
                                              synchronization_inventory, result);
    build_moi_fence_candidate_inventory(synchronization_inventory);
    result.program_inventory = inventory_builder.view();
    build_perturbation_candidate_inventory(result);
    annotate_fault_sites_with_sync_sequences(result);
  } else if (std::ranges::any_of(result.program_inventory.access_sites(),
                                 [](const ConSanAccessInventorySite &site) {
                                   return site.container.kind ==
                                          ConSanProgramContainerKind::Function;
                                 })) {
    // SuperCollider's ordinary clean transform does not consume synchronization
    // semantics. Shared-function memory sites still need kernel ownership,
    // while kernel-local sites carry their descriptor directly.
    annotate_execution_owners(*code_object, *decoder, arch, options, nullptr,
                              inventory_builder.access_sites(), synchronization_inventory, result);
  }
  result.program_inventory = inventory_builder.view();
  if (options.fault_dry_run) {
    build_fault_mutation_plan(options, arch, result);
    build_perturbation_plan(options, result, carried_plans);
    return result;
  }
  if (has_perturbation) {
    build_perturbation_plan(options, result, carried_plans);
    if (!result.errors.empty())
      return result;
  }

  if (options.flavor == ConSanFlavor::Moi) {
    // Incremental MOI passes reparse their current result. Release the
    // pristine inventory parser before entering them so its backing image,
    // section objects and names, and derived state cannot overlap every
    // stage's parser, patcher, text, and transaction.
    code_object.reset();
    return try_patch_consan_moi(std::move(result), options, code_object_bytes, arch);
  }

  if (options.flavor != ConSanFlavor::SuperCollider) {
    result.errors.emplace_back("ConSan selected unknown flavor");
    return result;
  }

  if (!initialize_sc_access_coverage(options, result))
    return result;

  const bool has_redundant_access = options.probe_lds_check_trap || options.probe_flat_check_trap;
  const uint32_t overall_patch_budget = std::max<uint32_t>(options.max_patches, 1u);
  if (result.perturbation_plans.size() > overall_patch_budget) {
    result.errors.emplace_back("ConSan SC perturbation exceeds the shared patch budget");
    return result;
  }
  ConSanOptions redundant_options = options;
  redundant_options.max_patches =
      overall_patch_budget - static_cast<uint32_t>(result.perturbation_plans.size());

  if (has_redundant_access) {
    if (options.probe_lds_check_trap)
      try_apply_lds_load_check_trap_patch(*code_object, arch, redundant_options, result);
    if (options.probe_flat_check_trap && result.errors.empty()) {
      if (!result.modified()) {
        try_apply_flat_check_trap_patch(*code_object, arch, redundant_options, result);
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
          ConSanOptions flat_options = redundant_options;
          flat_options.max_patches = 0u;
          try_apply_flat_check_trap_patch(*code_object, arch, flat_options, result);
        } else if (auto reserved_ranges =
                       reserved_ranges_for_existing_patches(*code_object, result)) {
          ConSanOptions flat_options = redundant_options;
          if (flat_options.max_patches_is_expert_limit)
            flat_options.max_patches = max_patches - access_patches;
          try_apply_flat_check_trap_patch(*code_object, arch, flat_options, result,
                                          *reserved_ranges);
        } else {
          ConSanOptions flat_options = redundant_options;
          flat_options.max_patches = 0u;
          flat_options.max_patches_is_expert_limit = true;
          try_apply_flat_check_trap_patch(*code_object, arch, flat_options, result);
          result.warnings.emplace_back(
              "ConSan flat check/trap proof skipped after native LDS patching: "
              "existing patch ranges could not be mapped in the original code object");
        }
      }
    }
  } else if (!has_perturbation && options.probe_flat_trap)
    try_apply_flat_trap_patch(*code_object, arch, options, result);
  else if (!has_perturbation && options.probe_lds_endpgm)
    try_apply_lds_endpgm_patch(*code_object, arch, result);
  else if (!has_perturbation && options.probe_endpgm)
    try_apply_proof_endpgm_patch(*code_object, arch, result);
  else if (!has_perturbation && (options.probe_nop || options.probe_trampoline_nop))
    try_apply_proof_nop_patch(*code_object, arch, options.probe_trampoline_nop, result);

  if (has_perturbation && result.errors.empty()) {
    try_apply_perturbation_patches(*code_object, arch, options, result);
    if (result.mutation.perturbation.applied != result.perturbation_plans.size()) {
      if (result.outcome == ConSanTransformOutcome::ModifiedValid)
        result.outcome = result.errors.empty() ? ConSanTransformOutcome::Unchanged
                                               : ConSanTransformOutcome::Invalid;
      result.discard_candidate_modification();
    }
  }

  finalize_sc_access_coverage_ledger(result);
  apply_fault_mutation_to_inventory(*code_object, arch, options, result);

  return result;
}
