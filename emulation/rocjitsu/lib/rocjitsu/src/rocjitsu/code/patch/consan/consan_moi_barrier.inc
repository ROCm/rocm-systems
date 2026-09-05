/// Project admitted barrier evidence directly into the common MOI lowering
/// contract shared by Record/Replay, Sampled, and InlineShadow.
///
/// Policy may map several barrier-event decisions to one completing intent.
/// This join therefore starts from intent IDs cited by admitted decisions,
/// deduplicates coalesced sequences, and resolves exactly one graph event and
/// decoded instruction per insertion site. Redundant barriers, runtime
/// kernels, container filters, and engine-specific signal/wait rules have
/// already been decided by policy and are never rediscovered here.
[[nodiscard]] std::vector<MoiBarrierEvidenceSitePlan> build_moi_barrier_evidence_site_plans(
    const ProgramInventory &inventory, const ConSanObservationPlan &observation,
    ConSanProbeIntentKind evidence_kind, std::vector<std::string> &errors) {
  const SynchronizationInventoryView graph = inventory.sync();
  std::vector<ConSanProbeIntentId> admitted_intents;
  for (const ConSanBarrierSiteDecision &decision : observation.barrier_site_decisions) {
    if (decision.kind != ConSanSiteDecisionKind::Admitted)
      continue;
    for (ConSanProbeIntentId id : decision.intent_ids) {
      const ConSanProbeIntent *intent = observation.intent(id);
      if (intent != nullptr && intent->kind == evidence_kind &&
          std::ranges::find(admitted_intents, id) == admitted_intents.end()) {
        admitted_intents.push_back(id);
      }
    }
  }

  std::vector<MoiBarrierEvidenceSitePlan> plans;
  plans.reserve(admitted_intents.size());
  for (ConSanProbeIntentId id : admitted_intents) {
    const ConSanProbeIntent *intent = observation.intent(id);
    const ConSanSyncEvent *event =
        intent == nullptr
            ? nullptr
            : graph.find_unique_event(ConSanSyncEventKind::Barrier, intent->physical_site);
    const ConSanSyncSequence *sequence =
        event == nullptr ? nullptr : graph.find_unique_sequence_containing(event->semantic_id);
    if (intent == nullptr || event == nullptr || sequence == nullptr ||
        intent->position != ConSanProbePosition::After) {
      errors.emplace_back("ConSan MOI admitted barrier evidence lost its synchronization graph");
      return {};
    }
    bool coverage_is_exact = !intent->covered_semantic_sites.empty();
    for (const SemanticSiteId &covered : intent->covered_semantic_sites) {
      const ConSanSyncEvent *member = graph.find_event(covered);
      SemanticSiteId member_id = covered;
      member_id.domain = ConSanSemanticSiteDomain::SynchronizationSequenceMember;
      coverage_is_exact &= member != nullptr && member->kind == ConSanSyncEventKind::Barrier &&
                           std::ranges::find(sequence->member_semantic_ids, member_id) !=
                               sequence->member_semantic_ids.end();
    }
    if (!coverage_is_exact) {
      errors.emplace_back(
          "ConSan MOI admitted barrier evidence has inconsistent sequence coverage");
      return {};
    }

    MoiBarrierEvidenceSitePlan plan;
    plan.semantic_site = event->semantic_id;
    plan.association = {.value = sequence->identity};
    plan.evidence_intent = id;
    plan.in_kernel = event->in_kernel;
    auto container = resolve_moi_evidence_container(inventory, event->in_kernel,
                                                    event->container_name, event->execution_owners);
    const ConSanBarrierSite *site = inventory.program_site<ConSanBarrierSite>(event->source_site);
    if (site == nullptr) {
      errors.emplace_back("ConSan MOI admitted barrier evidence lost its decoded lowering site");
      return {};
    }
    plan.container_name = std::move(container->qualified_name);
    plan.site = *site;
    plan.kernel_descriptor_file_offset = container->kernel_descriptor_file_offset;
    plan.container_entry_text_offset = container->entry_text_offset;
    plan.container_code_size = container->code_size;
    plan.text_file_offset = container->text_file_offset;
    if (!plan.is_well_formed()) {
      errors.emplace_back("ConSan MOI admitted barrier evidence lost its decoded lowering site");
      return {};
    }
    plans.push_back(std::move(plan));
  }

  std::ranges::sort(plans, {},
                    [](const MoiBarrierEvidenceSitePlan &plan) { return plan.site.text_offset; });
  return plans;
}

class MoiPatchAnchorRangeIndex {
public:
  MoiPatchAnchorRangeIndex(std::span<const ConSanPatchInfo> patches,
                           std::span<const ConSanTextFragment> staged_fragments) {
    for (const ConSanPatchLoweringProduct &patch : patches)
      add(patch);
    for (const ConSanTextFragment &fragment : staged_fragments)
      add(fragment.patch);
    std::ranges::sort(ranges_);
    std::vector<std::pair<uint64_t, uint64_t>> merged;
    for (const auto &range : ranges_) {
      if (merged.empty() || merged.back().second < range.first)
        merged.push_back(range);
      else
        merged.back().second = std::max(merged.back().second, range.second);
    }
    ranges_ = std::move(merged);
  }

  [[nodiscard]] bool overlaps(uint64_t begin, uint64_t size) const {
    if (size == 0u || begin > std::numeric_limits<uint64_t>::max() - size)
      return false;
    const auto it =
        std::ranges::lower_bound(ranges_, begin + size, {}, &std::pair<uint64_t, uint64_t>::first);
    return it != ranges_.begin() && std::prev(it)->second > begin;
  }

private:
  void add(const ConSanPatchLoweringProduct &patch) {
    if (patch.original_size != 0u)
      ranges_.emplace_back(patch.anchor_offset, patch.anchor_offset + patch.original_size);
  }

  std::vector<std::pair<uint64_t, uint64_t>> ranges_;
};

class MoiBarrierResourcePlanIndex {
public:
  explicit MoiBarrierResourcePlanIndex(std::span<const ConSanCandidateResourcePlan> plans) {
    for (const ConSanCandidateResourcePlan &plan : plans) {
      if (plan.site_kind == ConSanResourceSiteKind::Barrier)
        plans_by_offset_.try_emplace(plan.text_offset, &plan);
    }
  }

  [[nodiscard]] const ConSanCandidateResourcePlan *find(uint64_t text_offset) const {
    const auto found = plans_by_offset_.find(text_offset);
    return found == plans_by_offset_.end() ? nullptr : found->second;
  }

private:
  std::unordered_map<uint64_t, const ConSanCandidateResourcePlan *> plans_by_offset_;
};

class MoiBarrierPatchOwnerIndex {
public:
  MoiBarrierPatchOwnerIndex(std::span<const ConSanPatchInfo> patches,
                            std::span<const ConSanTextFragment> staged_fragments) {
    for (const ConSanPatchLoweringProduct &patch : patches)
      add(patch);
    for (const ConSanTextFragment &fragment : staged_fragments)
      add(fragment.patch);
  }

  [[nodiscard]] const ConSanPatchLoweringProduct *
  persistent_access_patch_for(std::span<const uint64_t> owners) const {
    if (owners.empty())
      return nullptr;
    const auto candidates = persistent_access_patches_by_owner_.find(owners.front());
    if (candidates == persistent_access_patches_by_owner_.end())
      return nullptr;
    const auto found = std::ranges::find_if(candidates->second, [&](const auto *patch) {
      return std::ranges::all_of(owners, [&](uint64_t owner) {
        return std::ranges::find(patch->owner_descriptor_file_offsets, owner) !=
               patch->owner_descriptor_file_offsets.end();
      });
    });
    return found == candidates->second.end() ? nullptr : *found;
  }

private:
  void add(const ConSanPatchLoweringProduct &patch) {
    if (!is_persistent_access_patch(patch))
      return;
    for (uint64_t owner : patch.owner_descriptor_file_offsets)
      persistent_access_patches_by_owner_[owner].push_back(&patch);
  }

  [[nodiscard]] static bool is_persistent_access_patch(const ConSanPatchLoweringProduct &patch) {
    return (patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
            patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore ||
            patch.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
            patch.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) &&
           patch.scratch_vgpr && patch.private_state_layout;
  }

  std::unordered_map<uint64_t, std::vector<const ConSanPatchLoweringProduct *>>
      persistent_access_patches_by_owner_;
};

using MoiBarrierBodyPlan =
    std::variant<MoiRecordEventEmissionPlan, MoiInlineBarrierEpochEmissionPlan,
                 MoiInlinePrivateEpochBarrierEmissionPlan>;

using MoiBarrierBodyPlanner = std::optional<MoiBarrierBodyPlan> (*)(
    const ConSanOptions &, ConSanMoiOperatingPoint &, const MoiScalarAbiPlan &,
    std::optional<uint16_t>, std::span<const uint64_t>, rj_code_arch_t, MoiOwnerAssignments,
    std::vector<std::string> &, std::vector<std::string> &);

struct PlannedBarrierLowering {
  MoiBarrierEvidenceSitePlan candidate;
  MoiBarrierBodyPlan body;
  std::optional<VgprSpillSequence> spill;
  std::optional<SgprSpillSequence> scalar_spill;
  ConSanPatchAbiEffects patch_abi;
};

struct MoiPrivateBarrierPlanningContext {
  std::span<const uint8_t> bytes;
  const ConSanOptions &options;
  ConSanMoiOperatingPoint point;
  const MoiBarrierEvidenceSitePlan &candidate;
  const ConSanCandidateResourcePlan &resource_plan;
  const ConSanPatchLoweringProduct &access_patch;
  const MoiObjectModeSemantics &semantics;
  MoiSpillManagers &spill_managers;
  rj_code_arch_t arch;
  const ProgramInventory &inventory;
  MoiOwnerAssignments assignments;
  std::vector<std::string> &warnings;
};

using MoiPrivateBarrierPlanner =
    std::optional<PlannedBarrierLowering> (*)(MoiPrivateBarrierPlanningContext);

[[nodiscard]] ConSanPatchAbiEffects
plan_barrier_patch_abi(std::span<const uint64_t> owners, std::optional<uint16_t> scratch_vgpr,
                       std::optional<ConSanMoiPrivateStateLayout> private_state_layout,
                       const VgprSpillSequence *spill, uint32_t required_private_bytes) {
  ConSanPatchAbiEffects abi;
  abi.scratch_vgpr = scratch_vgpr;
  abi.private_state_layout = std::move(private_state_layout);
  abi.required_private_segment_size = required_private_bytes;
  abi.owner_descriptor_file_offsets.assign(owners.begin(), owners.end());
  if (spill) {
    abi.spilled_vgpr_count = spill->vgpr_count;
    note_dynamic_stack_private_requirement(abi, spill);
  }
  return abi;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_barrier_epoch_cave_words(
    const MoiInlineBarrierEpochEmissionPlan &plan, rj_code_arch_t arch,
    const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    uint32_t original_barrier_word, std::optional<uint16_t> selectable_vgpr_bank_mode,
    std::vector<std::string> &errors, uint32_t *guest_instruction_offset = nullptr);

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_inline_shadow_private_epoch_barrier_cave_words(
    const MoiInlinePrivateEpochBarrierEmissionPlan &plan, const VgprSpillSequence &spill,
    const SgprSpillSequence *scalar_spill, rj_code_arch_t arch, uint32_t original_barrier_word,
    std::optional<uint16_t> selectable_vgpr_bank_mode, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr);
void try_apply_shared_barrier_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                    ConSanProbeIntentKind evidence_kind,
                                    MoiBarrierBodyPlanner plan_body,
                                    MoiPrivateBarrierPlanner plan_private_body,
                                    const ConSanMoiOperatingPoint &operating_point,
                                    rj_code_arch_t arch, const MoiObjectModeSemantics &semantics,
                                    ConSanTransformArtifacts &result) {
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch does not support this architecture");
    return;
  }
  if (!operating_point.moi_owner_epoch_vgprs.epoch() &&
      result.moi_operating_point.owner_persistent_vgprs.empty() &&
      !operating_point.moi_persistent_sgprs.epoch() &&
      !operating_point.automatic_moi_private_epoch) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch has no persistent epoch state");
    return;
  }
  if (operating_point.moi_owner_epoch_vgprs.epoch() &&
      static_cast<uint32_t>(operating_point.moi_owner_epoch_vgprs->epoch) + 1u > kMaxVgprs) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch needs one epoch VGPR");
    return;
  }

  const MoiModeOperations &mode = moi_mode_operations(options.moi_engine);
  const bool emits_barrier_record = evidence_kind == ConSanProbeIntentKind::BarrierRecord;
  const bool updates_exact_epoch = evidence_kind == ConSanProbeIntentKind::ExactBarrierEpoch;
  if (!emits_barrier_record && !updates_exact_epoch) {
    result.errors.emplace_back(
        "ConSan MOI exact barrier lowering received an incompatible evidence operation");
    return;
  }
  const ConSanPatchKind barrier_patch_kind = emits_barrier_record
                                                 ? ConSanPatchKind::TrampolineMoiBarrierRecord
                                                 : ConSanPatchKind::TrampolineMoiInlineEpochBarrier;
  std::vector<MoiBarrierEvidenceSitePlan> candidates = build_moi_barrier_evidence_site_plans(
      result.program_inventory, result.observation_plan(), evidence_kind, result.errors);
  if (!result.errors.empty())
    return;
  if (candidates.empty()) {
    result.warnings.emplace_back("ConSan MOI inline-shadow barrier epoch patch found no barriers");
    return;
  }

  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no .text section");
    return;
  }
  std::vector<MoiBarrierEvidenceSitePlan> selected_candidates;
  selected_candidates.reserve(options.max_patches);
  std::unordered_set<uint64_t> selected_offsets;
  const MoiPatchAnchorRangeIndex occupied_anchors(result.patches, result.staged_text_fragments);
  for (const MoiBarrierEvidenceSitePlan &candidate : candidates) {
    if (selected_candidates.size() == options.max_patches)
      break;
    const ConSanBarrierSite &site = candidate.site;
    if (site.size != sizeof(uint32_t)) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch skipped non-32-bit barrier " +
          site.mnemonic + " in " + candidate.container_name);
      continue;
    }
    if (site.text_offset > old_text.size() || site.size > old_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch site is outside .text");
      return;
    }
    if (occupied_anchors.overlaps(site.text_offset, site.size) ||
        !selected_offsets.insert(site.text_offset).second) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow barrier epoch skipped an overlapping barrier patch");
      continue;
    }
    selected_candidates.push_back(candidate);
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no patchable barriers");
    return;
  }

  std::vector<PlannedBarrierLowering> planned_candidates;
  MoiDescriptorPrivateRequirements private_requirements;
  MoiSpillManagers visible_evidence_spill_managers;
  MoiDescriptorVgprRequirements visible_evidence_vgpr_requirements;
  MoiDescriptorSgprRequirements visible_evidence_sgpr_requirements;
  MoiDescriptorPrivateRequirements visible_evidence_private_requirements;
  const MoiBarrierResourcePlanIndex resource_plan_index(result.resource_plans);
  const MoiBarrierPatchOwnerIndex patch_owner_index(result.patches, result.staged_text_fragments);
  if (operating_point.automatic_moi_private_epoch) {
    for (const MoiBarrierEvidenceSitePlan &candidate : selected_candidates) {
      const ConSanCandidateResourcePlan *plan =
          resource_plan_index.find(candidate.site.text_offset);
      if (plan == nullptr || plan->owner_descriptor_file_offsets.empty()) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier has no owned resource scope in " +
            candidate.container_name);
        continue;
      }
      const std::vector<uint64_t> &owners = plan->owner_descriptor_file_offsets;
      ConSanMoiOperatingPoint owner_point = operating_point;
      if (!apply_moi_persistent_vgpr_assignment(owner_point, result.moi_operating_point, owners)) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier lost its owner-local persistent assignment in " +
            candidate.container_name);
        continue;
      }
      if (!owner_point.automatic_moi_private_epoch)
        continue;
      if (!apply_moi_transient_sgpr_assignment(options, owner_point, result.moi_operating_point,
                                               owners)) {
        result.warnings.emplace_back("ConSan MOI private epoch barrier lost its transient state");
        continue;
      }
      const ConSanPatchLoweringProduct *access_patch =
          patch_owner_index.persistent_access_patch_for(owners);
      if (access_patch == nullptr) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier has no common access-probe state in " +
            candidate.container_name);
        continue;
      }
      auto planned = plan_private_body(MoiPrivateBarrierPlanningContext{
          .bytes = bytes,
          .options = options,
          .point = std::move(owner_point),
          .candidate = candidate,
          .resource_plan = *plan,
          .access_patch = *access_patch,
          .semantics = semantics,
          .spill_managers = visible_evidence_spill_managers,
          .arch = arch,
          .inventory = result.program_inventory,
          .assignments = result.moi_operating_point,
          .warnings = result.warnings,
      });
      if (planned) {
        note_maximum_descriptor_extents(private_requirements,
                                        planned->patch_abi.owner_descriptor_file_offsets,
                                        planned->patch_abi.required_private_segment_size);
        planned_candidates.push_back(std::move(*planned));
      }
    }
  }
  for (const MoiBarrierEvidenceSitePlan &candidate : selected_candidates) {
    std::vector<uint64_t> owners;
    const ConSanCandidateResourcePlan *plan = resource_plan_index.find(candidate.site.text_offset);
    if (plan) {
      owners = plan->owner_descriptor_file_offsets;
    }
    if (owners.empty() && candidate.kernel_descriptor_file_offset)
      owners.push_back(*candidate.kernel_descriptor_file_offset);
    if ((operating_point.automatic_moi_persistent_vgprs ||
         operating_point.moi_persistent_sgprs.complete() ||
         operating_point.automatic_moi_private_epoch) &&
        owners.empty())
      continue;
    ConSanMoiOperatingPoint candidate_point = operating_point;
    if (!apply_moi_transient_sgpr_assignment(options, candidate_point, result.moi_operating_point,
                                             owners) ||
        !apply_moi_persistent_vgpr_assignment(candidate_point, result.moi_operating_point,
                                              owners)) {
      continue;
    }
    if (candidate_point.automatic_moi_private_epoch)
      continue;
    std::optional<uint16_t> scratch_vgpr;
    std::optional<VgprSpillSequence> spill;
    std::optional<SgprSpillSequence> scalar_spill;
    std::optional<MoiBarrierBodyPlan> body;
    uint32_t required_private_bytes = 0u;
    const uint16_t scratch_count = mode.barrier_scratch_vgpr_count(
        project_moi_barrier_scratch_facts(options, candidate_point, semantics));
    const bool needs_record_replay_scalar_spill = candidate_point.has_compact_moi_scalar_spill();
    const bool needs_scalar_spill =
        needs_record_replay_scalar_spill ||
        (scratch_count > 1u && candidate_point.has_inline_moi_scalar_spill());
    if (scratch_count > 1u || needs_scalar_spill) {
      if (plan == nullptr) {
        result.warnings.emplace_back(
            "ConSan MOI inline barrier visible evidence has no resource plan in " +
            candidate.container_name);
        continue;
      }
      const auto resources = resolve_moi_scratch_plan(*plan, candidate_point,
                                                      result.moi_operating_point, scratch_count);
      if (!resources) {
        result.warnings.emplace_back(
            "ConSan MOI inline barrier visible evidence has no usable resource plan in " +
            candidate.container_name);
        continue;
      }
      // A one-VGPR Record/Replay resource can be used solely as the scalar
      // spill transfer register. It is not Inline visible-evidence scratch:
      // passing it as such would select Inline's much wider +12/+14 SGPR
      // layout and reject an otherwise valid compact Record/Replay probe.
      if (scratch_count > 1u)
        scratch_vgpr = resources->base;
      owners = resources->owner_descriptor_file_offsets;
      if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
          moi_scalar_spill_requires_dynamic_vgpr_frame(result.program_inventory, *resources,
                                                       candidate_point)) {
        spill = build_moi_spill_sequence(result.program_inventory, *resources, options, options,
                                         candidate_point, semantics,
                                         visible_evidence_spill_managers, arch, result.warnings);
        if (!spill)
          continue;
        required_private_bytes = spill->total_private_bytes;
        note_spill_descriptor_requirements(visible_evidence_private_requirements, *resources,
                                           *spill);
      }
      if (needs_scalar_spill) {
        scalar_spill = build_moi_sgpr_spill_sequence(
            result.program_inventory, *resources, options, options, candidate_point, semantics,
            visible_evidence_spill_managers, arch, result.warnings, std::nullopt,
            spill ? &*spill : nullptr);
        if (!scalar_spill)
          continue;
        required_private_bytes =
            std::max(required_private_bytes, scalar_spill->total_private_bytes);
        note_maximum_descriptor_extents(visible_evidence_private_requirements,
                                        resources->owner_descriptor_file_offsets,
                                        scalar_spill->total_private_bytes);
      }
      note_descriptor_requirements(visible_evidence_vgpr_requirements, *resources);
      note_moi_sgpr_requirements(visible_evidence_sgpr_requirements, *resources, options, options,
                                 candidate_point, semantics, arch);
    }
    const MoiScalarAbiPlan scalar_abi =
        plan_moi_scalar_abi(options.moi_engine, moi_scalar_preservation_state(candidate_point));
    body = plan_body(options, candidate_point, scalar_abi, scratch_vgpr, owners, arch,
                     result.moi_operating_point, result.warnings, result.errors);
    if (!body) {
      if (!result.errors.empty())
        return;
      continue;
    }
    ConSanPatchAbiEffects patch_abi = plan_barrier_patch_abi(
        owners, scratch_vgpr, std::nullopt, spill ? &*spill : nullptr, required_private_bytes);
    planned_candidates.push_back(PlannedBarrierLowering{
        .candidate = candidate,
        .body = std::move(*body),
        .spill = std::move(spill),
        .scalar_spill = std::move(scalar_spill),
        .patch_abi = std::move(patch_abi),
    });
  }
  if (planned_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no compatible persistent state");
    return;
  }
  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(planned_candidates.size());
  const ConSanMoiReportBufferLayout &record_layout = semantics.report_layout;
  for (const PlannedBarrierLowering &planned : planned_candidates) {
    const MoiBarrierEvidenceSitePlan &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    uint32_t original_barrier_word = 0u;
    std::memcpy(&original_barrier_word, old_text.data() + site.text_offset,
                sizeof(original_barrier_word));
    const std::optional<uint16_t> selectable_vgpr_bank_mode = consan_selectable_vgpr_bank_mode_at(
        arch, active_bytes, candidate.text_file_offset, candidate.container_entry_text_offset,
        site.file_offset);
    uint32_t guest_instruction_offset = 0u;
    std::optional<std::vector<uint32_t>> words;
    if (const auto *record = std::get_if<MoiRecordEventEmissionPlan>(&planned.body)) {
      words = build_barrier_record_cave_words(
          active_bytes, candidate, *record, planned.spill ? &*planned.spill : nullptr,
          planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch,
          record_layout.barrier_record_capacity, record_layout.barrier_records_offset,
          original_barrier_word, result.errors, &guest_instruction_offset);
    } else if (const auto *private_epoch =
                   std::get_if<MoiInlinePrivateEpochBarrierEmissionPlan>(&planned.body);
               private_epoch && planned.spill) {
      words = build_inline_shadow_private_epoch_barrier_cave_words(
          *private_epoch, *planned.spill, planned.scalar_spill ? &*planned.scalar_spill : nullptr,
          arch, original_barrier_word, selectable_vgpr_bank_mode, result.errors,
          &guest_instruction_offset);
    } else if (const auto *inline_epoch =
                   std::get_if<MoiInlineBarrierEpochEmissionPlan>(&planned.body)) {
      words = build_inline_shadow_barrier_epoch_cave_words(
          *inline_epoch, arch, planned.spill ? &*planned.spill : nullptr,
          planned.scalar_spill ? &*planned.scalar_spill : nullptr, original_barrier_word,
          selectable_vgpr_bank_mode, result.errors, &guest_instruction_offset);
    } else {
      result.errors.emplace_back("ConSan MOI barrier lost its planned emission state");
      return;
    }
    if (!words)
      return;

    ConSanPatchInfo patch;
    patch.kind = barrier_patch_kind;
    patch.anchor_offset = site.text_offset;
    patch.original_size = site.size;
    static_cast<ConSanPatchAbiEffects &>(patch) = planned.patch_abi;
    auto fragment = make_consan_around_text_fragment(
        std::move(*words), guest_instruction_offset, site.size, candidate.intent_ids(), {},
        std::move(patch), result.errors, "MOI inline-shadow barrier epoch patch");
    if (!fragment)
      return;
    fragments.push_back(std::move(*fragment));
  }

  std::unordered_set<uint64_t> emitted_owner_descriptors;
  bool has_unscoped_emitted_patch = false;
  for (const PlannedBarrierLowering &planned : planned_candidates) {
    has_unscoped_emitted_patch |= planned.patch_abi.owner_descriptor_file_offsets.empty();
    emitted_owner_descriptors.insert(planned.patch_abi.owner_descriptor_file_offsets.begin(),
                                     planned.patch_abi.owner_descriptor_file_offsets.end());
  }
  if (!operating_point.moi_persistent_sgprs.complete()) {
    if (operating_point.automatic_moi_persistent_vgprs) {
      for (uint64_t owner : emitted_owner_descriptors) {
        const auto *assignment = moi_persistent_vgpr_assignment(result.moi_operating_point, owner);
        if (assignment)
          note_maximum_descriptor_extent(
              visible_evidence_vgpr_requirements, owner,
              static_cast<uint16_t>(assignment->owner_epoch_vgprs.epoch + 1u));
      }
    } else if (!operating_point.automatic_moi_private_epoch) {
      const uint16_t required_vgprs = operating_point.moi_owner_epoch_vgprs->epoch + 1u;
      for (const ConSanProgramContainer &kernel : result.program_inventory.kernels())
        note_maximum_descriptor_extent(visible_evidence_vgpr_requirements,
                                       kernel.descriptor_file_offset, required_vgprs);
    }
  }
  merge_maximum_descriptor_extents(visible_evidence_private_requirements, private_requirements);
  const auto retain_emitted_owners = [&](auto &requirements) {
    if (has_unscoped_emitted_patch)
      return;
    std::erase_if(requirements, [&](const auto &entry) {
      return !emitted_owner_descriptors.contains(entry.first);
    });
  };
  retain_emitted_owners(visible_evidence_vgpr_requirements);
  retain_emitted_owners(visible_evidence_private_requirements);
  retain_emitted_owners(visible_evidence_sgpr_requirements);

  const MoiDescriptorLdsRequirements no_lds;
  (void)stage_consan_text_rewrites(
      code_object, arch,
      {visible_evidence_vgpr_requirements, visible_evidence_sgpr_requirements,
       visible_evidence_private_requirements, no_lds},
      moi_descriptor_mutation_policy(nullptr, arch), "MOI inline-shadow barrier epoch patch",
      std::move(fragments), result);
}
