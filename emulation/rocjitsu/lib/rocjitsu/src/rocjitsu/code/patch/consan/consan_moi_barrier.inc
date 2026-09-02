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
    const ConSanBarrierSite *site =
        container ? find_unique_moi_evidence_site(container->barrier_sites, event->text_offset)
                  : nullptr;
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
  explicit MoiPatchAnchorRangeIndex(std::span<const ConSanPatchInfo> patches) {
    for (const ConSanPatchLoweringProduct &patch : patches) {
      if (patch.original_size != 0u)
        ranges_.emplace_back(patch.anchor_offset, patch.anchor_offset + patch.original_size);
    }
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
  template <typename PatchRange> explicit MoiBarrierPatchOwnerIndex(const PatchRange &patches) {
    for (const ConSanPatchLoweringProduct &patch : patches) {
      if (!is_persistent_access_patch(patch))
        continue;
      for (uint64_t owner : patch.owner_descriptor_file_offsets)
        persistent_access_patches_by_owner_[owner].push_back(&patch);
    }
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

void note_relocated_barrier_instruction(ConSanPatchInfo &patch,
                                        std::span<const uint32_t> body_words,
                                        uint32_t original_barrier_word, uint64_t body_offset,
                                        uint64_t exact_epoch_relocated_offset,
                                        bool emits_barrier_record) {
  if (!emits_barrier_record) {
    patch.relocated_guest_instruction_offset = exact_epoch_relocated_offset;
    return;
  }
  const auto guest =
      std::ranges::find(body_words.rbegin(), body_words.rend(), original_barrier_word);
  if (guest != body_words.rend()) {
    patch.relocated_guest_instruction_offset =
        body_offset +
        static_cast<uint64_t>(std::distance(guest, body_words.rend()) - 1u) * sizeof(uint32_t);
  }
}

struct MoiInlineBarrierVisibleEvidencePlan {
  uint64_t counter_address = 0;
  uint16_t result_vgpr = 0;
  uint16_t address_vgpr = 0;
  uint16_t exec_save_sgpr = 0;
};

/// Exact InlineShadow state consumed by one ordinary barrier-epoch body.
/// Candidate planning resolves the persistent representation and scalar ABI;
/// native emission cannot reopen request or operating-point policy.
struct MoiInlineBarrierEpochEmissionPlan {
  std::optional<uint16_t> persistent_epoch_sgpr;
  std::optional<uint16_t> epoch_vgpr;
  std::optional<MoiInlineBarrierVisibleEvidencePlan> visible_evidence;
  std::optional<MoiSpecialStateSgprs> special_state;
  std::optional<ConSanIndirectJumpSgprs> indirect_jump;
};

/// Exact private-epoch state consumed by one InlineShadow barrier body.
struct MoiInlinePrivateEpochBarrierEmissionPlan {
  uint16_t scratch_vgpr = 0;
  uint32_t epoch_offset = 0;
  std::optional<ConSanIndirectJumpSgprs> indirect_jump;
};

using MoiBarrierBodyPlan =
    std::variant<MoiRecordEventEmissionPlan, MoiInlineBarrierEpochEmissionPlan,
                 MoiInlinePrivateEpochBarrierEmissionPlan>;

[[nodiscard]] std::optional<ConSanIndirectJumpSgprs>
moi_barrier_body_indirect_jump(const MoiBarrierBodyPlan &body) {
  return std::visit([](const auto &plan) { return plan.indirect_jump; }, body);
}

[[nodiscard]] ConSanPatchAbiEffects
plan_inline_barrier_patch_abi(std::span<const uint64_t> owners,
                              std::optional<uint16_t> scratch_vgpr,
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

[[nodiscard]] std::optional<MoiInlineBarrierEpochEmissionPlan>
plan_inline_shadow_barrier_epoch_emission(const BoundRuntimeResources &bound_resources,
                                          const ConSanMoiOperatingPoint &point,
                                          const MoiScalarAbiPlan &scalar_abi,
                                          std::optional<uint16_t> visible_evidence_scratch_vgpr,
                                          rj_code_arch_t arch, std::vector<std::string> &errors) {
  MoiInlineBarrierEpochEmissionPlan plan{
      .persistent_epoch_sgpr = point.moi_persistent_sgprs.epoch(),
      .epoch_vgpr = point.moi_owner_epoch_vgprs.epoch(),
      .visible_evidence = std::nullopt,
      .special_state = scalar_abi.special_state,
      .indirect_jump = scalar_abi.indirect_jump,
  };
  if (visible_evidence_scratch_vgpr) {
    const ConSanTargetProfile *target = consan_target_profile(arch);
    if (!bound_resources.moi_report_buffer_address || !point.moi_exec_save_sgpr ||
        target == nullptr || *visible_evidence_scratch_vgpr > 253u) {
      errors.emplace_back(
          "ConSan MOI inline barrier visible evidence requires report and scalar state");
      return std::nullopt;
    }
    const uint16_t scratch = *visible_evidence_scratch_vgpr;
    const bool address_first = target->moi_visible_evidence_address_precedes_result;
    plan.visible_evidence = MoiInlineBarrierVisibleEvidencePlan{
        .counter_address = *bound_resources.moi_report_buffer_address +
                           offsetof(ConSanMoiReportHeader, event_counter),
        .result_vgpr = static_cast<uint16_t>(scratch + (address_first ? 2u : 0u)),
        .address_vgpr = static_cast<uint16_t>(scratch + (address_first ? 0u : 1u)),
        .exec_save_sgpr = *point.moi_exec_save_sgpr,
    };
  }
  if ((plan.persistent_epoch_sgpr || plan.visible_evidence) && !plan.special_state) {
    errors.emplace_back("ConSan MOI scalar inline barrier requires scalar special-state storage");
    return std::nullopt;
  }
  return plan;
}

[[nodiscard]] bool
append_inline_barrier_return(std::vector<uint32_t> &words, uint64_t cave_text_offset,
                             uint64_t return_text_offset, std::optional<uint16_t> call_return_sgpr,
                             const std::optional<ConSanIndirectJumpSgprs> &indirect_jump,
                             rj_code_arch_t arch) {
  const uint64_t branch_pc =
      cave_text_offset + static_cast<uint64_t>(words.size()) * sizeof(uint32_t);
  if (call_return_sgpr) {
    words.push_back(build_s_setpc_b64(*call_return_sgpr, arch));
    return true;
  }
  if (const auto direct = compute_sopp_branch_simm16(branch_pc, return_text_offset)) {
    words.push_back(build_s_branch(*direct, arch));
    return true;
  }
  return indirect_jump && append_moi_scc_preserving_indirect_jump(
                              words, cave_text_offset, return_text_offset, indirect_jump->pc_sgpr,
                              indirect_jump->scc_save_sgpr,
                              /*capture_scc=*/true, arch);
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_barrier_epoch_cave_words(
    const ConSanBarrierSite &site, const MoiInlineBarrierEpochEmissionPlan &plan,
    rj_code_arch_t arch, const VgprSpillSequence *spill, const SgprSpillSequence *scalar_spill,
    uint32_t original_barrier_word, uint64_t cave_text_offset, uint64_t return_text_offset,
    std::optional<uint16_t> call_return_sgpr, std::optional<uint16_t> selectable_vgpr_bank_mode,
    std::vector<std::string> &errors);

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_inline_shadow_private_epoch_barrier_cave_words(
    const ConSanBarrierSite &site, const MoiInlinePrivateEpochBarrierEmissionPlan &plan,
    const VgprSpillSequence &spill, const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    uint32_t original_barrier_word, uint64_t cave_text_offset, uint64_t return_text_offset,
    std::optional<uint16_t> call_return_sgpr, std::optional<uint16_t> selectable_vgpr_bank_mode,
    std::vector<std::string> &errors);
[[nodiscard]] std::optional<MoiBarrierIslandReservation>
resolve_reserved_barrier_island_layout(
    std::span<const uint8_t> text, std::span<const ConSanPatchInfo> patches, rj_code_arch_t arch,
    const std::optional<MoiBarrierIslandReservation> &explicit_layout, size_t barrier_count,
    uint64_t reserved_barrier_count) {
  if (barrier_count == 0u || reserved_barrier_count < barrier_count)
    return std::nullopt;

  std::unordered_set<uint64_t> access_anchors;
  uint64_t first_access_body = std::numeric_limits<uint64_t>::max();
  for (const ConSanPatchLoweringProduct &patch : patches) {
    if (patch.kind != ConSanPatchKind::InlineMoiExactShadowStore &&
        patch.kind != ConSanPatchKind::TrampolineMoiExactShadowStore &&
        patch.kind != ConSanPatchKind::InlineMoiAccessRecordStore &&
        patch.kind != ConSanPatchKind::TrampolineMoiAccessRecordStore) {
      continue;
    }
    access_anchors.insert(patch.anchor_offset);
    if (patch.trampoline_size != 0u)
      first_access_body = std::min(first_access_body, patch.trampoline_offset);
  }

  uint64_t access_island_begin = std::numeric_limits<uint64_t>::max();
  uint64_t access_island_end = 0u;
  for (const ConSanPatchLoweringProduct &patch : patches) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
        !access_anchors.contains(patch.anchor_offset)) {
      continue;
    }
    access_island_begin = std::min(access_island_begin, patch.trampoline_offset);
    access_island_end = std::max(
        access_island_end, patch.trampoline_offset + static_cast<uint64_t>(patch.trampoline_size));
  }
  if (explicit_layout && explicit_layout->reserved_slot_count == reserved_barrier_count &&
      (explicit_layout->slot_words == kMoiRecordReplayIndirectIslandWords ||
       explicit_layout->slot_words == kMoiLocalIndirectIslandWords ||
       explicit_layout->slot_words == kMoiRecordReplayBarrierRelaySlotWords) &&
      reserved_barrier_count <=
          std::numeric_limits<uint64_t>::max() / (explicit_layout->slot_words * sizeof(uint32_t))) {
    const uint64_t reservation_bytes =
        reserved_barrier_count * explicit_layout->slot_words * sizeof(uint32_t);
    if (explicit_layout->begin <= std::numeric_limits<uint64_t>::max() - reservation_bytes) {
      access_island_begin = explicit_layout->begin + reservation_bytes;
    }
  }

  std::unordered_set<uint64_t> generated_branch_relays;
  for (const ConSanPatchLoweringProduct &patch : patches) {
    if (patch.kind == ConSanPatchKind::TrampolineBranchRelayReservoir &&
        patch.original_size == 0u && patch.trampoline_size == sizeof(uint32_t)) {
      generated_branch_relays.insert(patch.trampoline_offset);
    }
  }
  const auto checked_reserved_bytes = [&](uint32_t slot_words) -> std::optional<uint64_t> {
    constexpr uint64_t kWordBytes = sizeof(uint32_t);
    if (reserved_barrier_count > std::numeric_limits<uint64_t>::max() / (slot_words * kWordBytes)) {
      return std::nullopt;
    }
    return reserved_barrier_count * slot_words * kWordBytes;
  };
  const auto word_is_nop = [&](uint64_t offset) {
    if (offset > text.size() || sizeof(uint32_t) > text.size() - offset)
      return false;
    uint32_t word = 0u;
    std::memcpy(&word, text.data() + offset, sizeof(word));
    return word == build_s_nop(0, arch);
  };
  const auto word_is_branch = [&](uint64_t offset) {
    if (offset > text.size() || sizeof(uint32_t) > text.size() - offset)
      return false;
    uint32_t word = 0u;
    std::memcpy(&word, text.data() + offset, sizeof(word));
    return (word & 0xffff0000u) == (build_s_branch(0, arch) & 0xffff0000u);
  };
  const auto is_reserved_nop_range = [&](uint64_t candidate_begin, uint64_t required_bytes) {
    bool all_reserved_nops =
        candidate_begin <= text.size() && required_bytes <= text.size() - candidate_begin;
    for (uint64_t offset = candidate_begin;
         all_reserved_nops && offset < candidate_begin + required_bytes;
         offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, text.data() + offset, sizeof(word));
      all_reserved_nops = word == build_s_nop(0, arch);
    }
    return all_reserved_nops;
  };

  // A branch-only Record/Replay access pass may retain a dedicated relay pair
  // after each seven-word barrier island. Claimed spine words already
  // contain emitted branches, so recognize them through their patch
  // provenance while requiring every barrier-owned word to remain pristine.
  const auto find_spined_prefix = [&]() -> std::optional<MoiBarrierIslandReservation> {
    const std::optional<uint64_t> required_bytes =
        checked_reserved_bytes(kMoiRecordReplayBarrierRelaySlotWords);
    if (!required_bytes || access_island_begin == std::numeric_limits<uint64_t>::max() ||
        access_island_begin < *required_bytes) {
      return std::nullopt;
    }
    const uint64_t candidate_begin = access_island_begin - *required_bytes;
    const bool has_explicit_spined_layout =
        explicit_layout && explicit_layout->begin == candidate_begin &&
        explicit_layout->slot_words == kMoiRecordReplayBarrierRelaySlotWords &&
        explicit_layout->reserved_slot_count == reserved_barrier_count;
    bool has_claimed_spine_word = false;
    for (uint64_t slot = 0u; slot < reserved_barrier_count; ++slot) {
      const uint64_t slot_begin =
          candidate_begin + slot * kMoiRecordReplayBarrierRelaySlotWords * sizeof(uint32_t);
      for (uint32_t word = 0u; word < kMoiRecordReplayIndirectIslandWords; ++word) {
        if (!word_is_nop(slot_begin + word * sizeof(uint32_t)))
          return std::nullopt;
      }
      for (uint32_t word = kMoiRecordReplayIndirectIslandWords;
           word < kMoiRecordReplayBarrierRelaySlotWords; ++word) {
        const uint64_t spine_offset = slot_begin + word * sizeof(uint32_t);
        const bool claimed = generated_branch_relays.contains(spine_offset) ||
                             (has_explicit_spined_layout && word_is_branch(spine_offset));
        if (!claimed && !word_is_nop(spine_offset))
          return std::nullopt;
        has_claimed_spine_word = has_claimed_spine_word || claimed;
      }
    }
    if (!has_claimed_spine_word && !has_explicit_spined_layout)
      return std::nullopt;
    return MoiBarrierIslandReservation{
        .begin = candidate_begin,
        .slot_words = kMoiRecordReplayBarrierRelaySlotWords,
        .reserved_slot_count = reserved_barrier_count,
    };
  };
  if (const auto spined = find_spined_prefix())
    return spined;

  const uint32_t plain_island_words =
      explicit_layout && explicit_layout->slot_words == kMoiLocalIndirectIslandWords
          ? kMoiLocalIndirectIslandWords
          : kMoiRecordReplayIndirectIslandWords;
  const std::optional<uint64_t> required_island_bytes = checked_reserved_bytes(plain_island_words);
  if (!required_island_bytes)
    return std::nullopt;

  // Current access placement puts barrier islands first: early
  // synchronization sites are more likely to reach the start of a large
  // generated object, while sorted access relays follow them.
  if (access_island_begin != std::numeric_limits<uint64_t>::max() &&
      access_island_begin >= *required_island_bytes) {
    const uint64_t candidate_begin = access_island_begin - *required_island_bytes;
    if (is_reserved_nop_range(candidate_begin, *required_island_bytes)) {
      return MoiBarrierIslandReservation{
          .begin = candidate_begin,
          .slot_words = plain_island_words,
          .reserved_slot_count = reserved_barrier_count,
      };
    }
  }

  // Accept the older trailing reservation layout as well so independently
  // composed exact-shadow producers remain compatible.
  if (first_access_body != std::numeric_limits<uint64_t>::max() &&
      first_access_body >= *required_island_bytes &&
      first_access_body - *required_island_bytes >= access_island_end) {
    const uint64_t candidate_begin = first_access_body - *required_island_bytes;
    if (is_reserved_nop_range(candidate_begin, *required_island_bytes)) {
      return MoiBarrierIslandReservation{
          .begin = candidate_begin,
          .slot_words = plain_island_words,
          .reserved_slot_count = reserved_barrier_count,
      };
    }
  }
  return std::nullopt;
}

void try_apply_shared_barrier_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
    MoiResourcePlanningState &resource_state,
    const std::optional<MoiBarrierIslandReservation> &reserved_sync_islands,
    const MoiObjectModeSemantics &semantics, ConSanTransformArtifacts &result) {
  if (!options.moi_track_barriers)
    return;
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
  const ConSanEngineProbeVocabulary &evidence =
      *consan_engine_probe_vocabulary(result.observation_plan().engine);
  const ConSanProbeIntentKind evidence_kind = evidence.barrier;
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
  const MoiPatchAnchorRangeIndex occupied_anchors(result.patches);
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

  struct PlannedInlineEpochBarrier {
    MoiBarrierEvidenceSitePlan candidate;
    std::optional<MoiWorkitemOwnerDerivationPlan> derived_owner;
    MoiBarrierBodyPlan body;
    bool branch_only_spill = false;
    std::optional<VgprSpillSequence> spill;
    std::optional<SgprSpillSequence> scalar_spill;
    ConSanPatchAbiEffects patch_abi;

    [[nodiscard]] bool uses_scalar_epoch() const {
      const auto *inline_epoch = std::get_if<MoiInlineBarrierEpochEmissionPlan>(&body);
      return inline_epoch != nullptr && inline_epoch->persistent_epoch_sgpr.has_value();
    }
  };
  std::vector<PlannedInlineEpochBarrier> planned_candidates;
  MoiDescriptorPrivateRequirements private_requirements;
  MoiDescriptorVgprRequirements persistent_epoch_vgpr_requirements;
  MoiSpillManagers visible_evidence_spill_managers;
  MoiDescriptorVgprRequirements visible_evidence_vgpr_requirements;
  MoiDescriptorSgprRequirements visible_evidence_sgpr_requirements;
  MoiDescriptorPrivateRequirements visible_evidence_private_requirements;
  const MoiBarrierResourcePlanIndex resource_plan_index(result.resource_plans);
  const MoiBarrierPatchOwnerIndex patch_owner_index(result.patches);
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
      const ConSanPatchLoweringProduct *access_patch =
          patch_owner_index.persistent_access_patch_for(owners);
      if (access_patch == nullptr) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier has no common access-probe state in " +
            candidate.container_name);
        continue;
      }
      const ConSanMoiPrivateStateLayout &private_state_layout = *access_patch->private_state_layout;
      if (emits_barrier_record) {
        const auto resources =
            resolve_moi_scratch_plan(*plan, owner_point, result.moi_operating_point,
                                     /*expected_count=*/6u);
        if (!resources || !private_state_layout.record_replay_workgroup_offsets.complete()) {
          result.warnings.emplace_back(
              "ConSan MOI Record/Replay dense barrier has no complete private exact-workgroup "
              "layout in " +
              candidate.container_name);
          continue;
        }
        ConSanMoiOperatingPoint candidate_point = std::move(owner_point);
        if (!apply_moi_transient_sgpr_assignment(options, candidate_point,
                                                 result.moi_operating_point, owners) ||
            !apply_record_replay_entry_workgroup_assignment(options, candidate_point,
                                                            result.moi_operating_point, owners,
                                                            &private_state_layout)) {
          result.warnings.emplace_back(
              "ConSan MOI Record/Replay dense barrier lost its private owner-local assignment");
          continue;
        }
        const MoiScalarAbiPlan scalar_abi =
            plan_moi_scalar_abi(options.moi_engine, moi_scalar_routing_state(candidate_point));
        const auto record_emission =
            resolve_moi_record_event_emission_plan(options, options, candidate_point, scalar_abi,
                                                   resources->base, arch, &private_state_layout);
        if (!record_emission) {
          result.warnings.emplace_back(
              "ConSan MOI Record/Replay dense barrier has an incomplete emission plan");
          continue;
        }
        std::optional<MoiWorkitemOwnerDerivationPlan> derived_owner;
        if (!candidate_point.moi_owner_epoch_vgprs.owner() &&
            !candidate_point.moi_persistent_sgprs.owner()) {
          const auto owner_shift =
              common_moi_workitem_owner_shift(bytes, *resources, arch, result.warnings);
          if (!owner_shift)
            continue;
          derived_owner = MoiWorkitemOwnerDerivationPlan{
              .entry_workitem_x_private_offset = private_state_layout.owner_offset,
              .wave_size_shift = *owner_shift,
          };
        }
        const uint32_t ephemeral_base = private_state_layout.ephemeral_base;
        std::optional<VgprSpillSequence> spill;
        if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
            moi_scalar_spill_requires_dynamic_vgpr_frame(result.program_inventory, *resources,
                                                         candidate_point)) {
          spill = build_moi_spill_sequence(
              result.program_inventory, *resources, options, options, candidate_point, semantics,
              visible_evidence_spill_managers, arch, result.warnings, ephemeral_base);
          if (!spill)
            continue;
        }
        std::optional<SgprSpillSequence> scalar_spill;
        if (candidate_point.has_compact_moi_scalar_spill()) {
          scalar_spill = build_moi_sgpr_spill_sequence(
              result.program_inventory, *resources, options, options, candidate_point, semantics,
              visible_evidence_spill_managers, arch, result.warnings, ephemeral_base,
              spill ? &*spill : nullptr);
          if (!scalar_spill)
            continue;
        }
        uint32_t required_private_bytes = ephemeral_base;
        if (spill)
          required_private_bytes = std::max(required_private_bytes, spill->total_private_bytes);
        if (scalar_spill) {
          required_private_bytes =
              std::max(required_private_bytes, scalar_spill->total_private_bytes);
        }
        note_maximum_descriptor_extents(private_requirements, owners, required_private_bytes);
        ConSanPatchAbiEffects patch_abi =
            plan_inline_barrier_patch_abi(owners, resources->base, private_state_layout,
                                          spill ? &*spill : nullptr, required_private_bytes);
        planned_candidates.push_back(PlannedInlineEpochBarrier{
            .candidate = candidate,
            .derived_owner = derived_owner,
            .body = *record_emission,
            .branch_only_spill = candidate_point.moi_branch_only_spill.has_value(),
            .spill = std::move(spill),
            .scalar_spill = std::move(scalar_spill),
            .patch_abi = std::move(patch_abi),
        });
        continue;
      }
      const uint32_t ephemeral_base = private_state_layout.ephemeral_base;
      const auto private_limit = consan_address_free_private_limit(arch);
      if (!private_limit) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier has no address-free scratch support");
        continue;
      }
      SpillManager manager(ephemeral_base, *private_limit);
      auto spill = build_vgpr_spill_sequence(manager, *access_patch->scratch_vgpr,
                                             /*vgpr_count=*/1, arch);
      if (!spill) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier could not preserve its temporary VGPR");
        continue;
      }
      if (!apply_moi_transient_sgpr_assignment(options, owner_point, result.moi_operating_point,
                                               owners)) {
        result.warnings.emplace_back(
            "ConSan MOI private epoch barrier lost its transient scalar assignment");
        continue;
      }
      const uint32_t required_private_bytes =
          std::max(access_patch->required_private_segment_size, spill->total_private_bytes);
      note_maximum_descriptor_extents(private_requirements, owners, required_private_bytes);
      ConSanPatchAbiEffects patch_abi =
          plan_inline_barrier_patch_abi(owners, access_patch->scratch_vgpr, private_state_layout,
                                        &*spill, required_private_bytes);
      const MoiScalarAbiPlan scalar_abi =
          plan_moi_scalar_abi(options.moi_engine, moi_scalar_routing_state(owner_point));
      planned_candidates.push_back(PlannedInlineEpochBarrier{
          .candidate = candidate,
          .derived_owner = std::nullopt,
          .body =
              MoiInlinePrivateEpochBarrierEmissionPlan{
                  .scratch_vgpr = *access_patch->scratch_vgpr,
                  .epoch_offset = private_state_layout.epoch_offset,
                  .indirect_jump = scalar_abi.indirect_jump,
              },
          .branch_only_spill = owner_point.moi_branch_only_spill.has_value(),
          .spill = std::move(spill),
          .scalar_spill = std::nullopt,
          .patch_abi = std::move(patch_abi),
      });
    }
  }
  if (operating_point.moi_persistent_sgprs.complete()) {
    // The owner-kernel entry prologue grows each descriptor for this scalar
    // tuple after all emitted patch ownership is known.
  } else if (operating_point.automatic_moi_persistent_vgprs) {
    for (const MoiBarrierEvidenceSitePlan &candidate : selected_candidates) {
      const ConSanCandidateResourcePlan *plan =
          resource_plan_index.find(candidate.site.text_offset);
      if (plan == nullptr || plan->owner_descriptor_file_offsets.empty()) {
        result.warnings.emplace_back(
            "ConSan MOI inline barrier has no owned automatic persistent-state scope");
        continue;
      }
      for (uint64_t descriptor_offset : plan->owner_descriptor_file_offsets) {
        const auto *assignment =
            moi_persistent_vgpr_assignment(result.moi_operating_point, descriptor_offset);
        if (assignment)
          persistent_epoch_vgpr_requirements[descriptor_offset] =
              static_cast<uint16_t>(assignment->owner_epoch_vgprs.epoch + 1u);
      }
    }
  } else if (!operating_point.automatic_moi_private_epoch) {
    for (const ConSanKernelInfo &kernel : result.program_inventory.kernels()) {
      persistent_epoch_vgpr_requirements.emplace(
          kernel.descriptor_file_offset,
          static_cast<uint32_t>(operating_point.moi_owner_epoch_vgprs->epoch) + 1u);
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
    if (emits_barrier_record) {
      if (!scratch_vgpr) {
        result.warnings.emplace_back(
            "ConSan MOI Record/Replay dense barrier fallback has no scratch VGPR state");
        continue;
      }
      if (!apply_record_replay_entry_workgroup_assignment(options, candidate_point,
                                                          result.moi_operating_point, owners)) {
        result.warnings.emplace_back(
            "ConSan MOI Record/Replay dense barrier fallback lost its entry workgroup assignment");
        continue;
      }
    }
    const MoiScalarAbiPlan scalar_abi =
        plan_moi_scalar_abi(options.moi_engine, moi_scalar_routing_state(candidate_point));
    if (emits_barrier_record) {
      auto record = resolve_moi_record_event_emission_plan(options, options, candidate_point,
                                                           scalar_abi, *scratch_vgpr, arch);
      if (!record) {
        result.warnings.emplace_back(
            "ConSan MOI Record/Replay dense barrier fallback has an incomplete emission plan");
        continue;
      }
      body = std::move(*record);
    } else {
      auto inline_epoch = plan_inline_shadow_barrier_epoch_emission(
          options, candidate_point, scalar_abi, scratch_vgpr, arch, result.errors);
      if (!inline_epoch)
        return;
      body = std::move(*inline_epoch);
    }
    ConSanPatchAbiEffects patch_abi = plan_inline_barrier_patch_abi(
        owners, scratch_vgpr, std::nullopt, spill ? &*spill : nullptr, required_private_bytes);
    planned_candidates.push_back(PlannedInlineEpochBarrier{
        .candidate = candidate,
        .derived_owner = std::nullopt,
        .body = std::move(*body),
        .branch_only_spill = candidate_point.moi_branch_only_spill.has_value(),
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
  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  DbiPatchPlacementPlanner placement_planner(arch, old_text.size());
  MoiLocalNopIslandAllocator local_indirect_allocator(
      old_text, result.program_inventory, result.patches, arch, kMoiLocalIndirectIslandWords);
  std::vector<ConSanPatchInfo> patches;
  std::vector<ConSanCommittedLowering> lowering_commits;
  patches.reserve(planned_candidates.size() * 2u);
  lowering_commits.reserve(planned_candidates.size());
  constexpr uint64_t kIndirectIslandWords = kMoiRecordReplayIndirectIslandWords;
  const uint64_t indirect_island_bytes = kIndirectIslandWords * sizeof(uint32_t);
  const std::optional<MoiBarrierIslandReservation> barrier_island_layout =
      operating_point.moi_exec_save_sgpr
          ? resolve_reserved_barrier_island_layout(
                old_text, result.patches, arch, reserved_sync_islands, planned_candidates.size(),
                moi_reserved_access_sync_island_count(result.program_inventory,
                                                      result.observation_plan(), options, options,
                                                      semantics))
          : std::nullopt;
  if (barrier_island_layout) {
    const uint64_t reserved_words =
        barrier_island_layout->reserved_slot_count * barrier_island_layout->slot_words;
    if (reserved_words > std::numeric_limits<uint32_t>::max()) {
      result.errors.emplace_back("ConSan MOI reserved barrier-island layout is too large");
      return;
    }
    local_indirect_allocator.reserve(barrier_island_layout->begin,
                                     static_cast<uint32_t>(reserved_words));
  }
  std::vector<size_t> dense_fallback_indices;
  const ConSanMoiReportBufferLayout &record_layout = semantics.report_layout;
  const auto build_engine_barrier_body =
      [&](const PlannedInlineEpochBarrier &planned, uint32_t original_barrier_word,
          uint64_t body_offset, uint64_t return_offset, std::optional<uint16_t> call_return_sgpr,
          std::optional<uint16_t> selectable_vgpr_bank_mode)
      -> std::optional<std::vector<uint32_t>> {
    if (const auto *record = std::get_if<MoiRecordEventEmissionPlan>(&planned.body)) {
      return build_barrier_record_cave_words(
          active_bytes, planned.candidate, *record, planned.spill ? &*planned.spill : nullptr,
          planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch,
          record_layout.barrier_record_capacity, record_layout.barrier_records_offset,
          original_barrier_word, body_offset, return_offset, planned.derived_owner, result.errors);
    }
    if (const auto *private_epoch =
            std::get_if<MoiInlinePrivateEpochBarrierEmissionPlan>(&planned.body);
        private_epoch && planned.spill) {
      return build_inline_shadow_private_epoch_barrier_cave_words(
          planned.candidate.site, *private_epoch, *planned.spill,
          planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, original_barrier_word,
          body_offset, return_offset, call_return_sgpr, selectable_vgpr_bank_mode, result.errors);
    }
    const auto *inline_epoch = std::get_if<MoiInlineBarrierEpochEmissionPlan>(&planned.body);
    if (!inline_epoch) {
      result.errors.emplace_back("ConSan MOI inline barrier lost its planned epoch emission state");
      return std::nullopt;
    }
    return build_inline_shadow_barrier_epoch_cave_words(
        planned.candidate.site, *inline_epoch, arch, planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, original_barrier_word, body_offset,
        return_offset, call_return_sgpr, selectable_vgpr_bank_mode, result.errors);
  };

  for (size_t planned_index = 0; planned_index < planned_candidates.size(); ++planned_index) {
    const PlannedInlineEpochBarrier &planned = planned_candidates[planned_index];
    const MoiBarrierEvidenceSitePlan &candidate = planned.candidate;
    const ConSanBarrierSite &site = candidate.site;
    const auto indirect_jump = moi_barrier_body_indirect_jump(planned.body);
    uint32_t original_barrier_word = 0;
    std::memcpy(&original_barrier_word, new_text.data() + site.text_offset,
                sizeof(original_barrier_word));
    const std::optional<uint16_t> selectable_vgpr_bank_mode = consan_selectable_vgpr_bank_mode_at(
        arch, active_bytes, candidate.text_file_offset, candidate.container_entry_text_offset,
        site.file_offset);
    std::optional<uint64_t> oversized_local_body_island;

    // The persistent-VGPR epoch update is small enough to live entirely in a
    // local NOP island. Prefer that layout over an indirect island followed by
    // an appended body: besides being smaller, it avoids perturbing SCC and
    // consuming temporary SGPR state on both the entry and return paths. The
    // private-epoch form is larger and continues to use the general appended
    // placement below.
    if (!std::holds_alternative<MoiInlinePrivateEpochBarrierEmissionPlan>(planned.body) &&
        !planned.uses_scalar_epoch()) {
      if (const auto local_body_offset =
              local_indirect_allocator.claim_reachable(site.text_offset)) {
        const uint64_t return_text_offset = site.text_offset + site.size;
        auto local_words =
            build_engine_barrier_body(planned, original_barrier_word, *local_body_offset,
                                      return_text_offset, std::nullopt, selectable_vgpr_bank_mode);
        if (!local_words)
          return;
        if (local_words->size() <= kMoiLocalIndirectIslandWords &&
            *local_body_offset + local_words->size() * sizeof(uint32_t) <= new_text.size()) {
          const auto fwd = compute_sopp_branch_simm16(site.text_offset, *local_body_offset);
          if (!fwd || !write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow barrier epoch could not enter its local body");
            return;
          }
          std::memcpy(new_text.data() + *local_body_offset, local_words->data(),
                      local_words->size() * sizeof(uint32_t));

          ConSanPatchInfo info;
          info.kind = barrier_patch_kind;
          info.anchor_offset = site.text_offset;
          info.trampoline_offset = *local_body_offset;
          info.original_size = site.size;
          info.trampoline_size = static_cast<uint32_t>(local_words->size() * sizeof(uint32_t));
          info.relocated_guest_instruction_offset =
              *local_body_offset + (planned.uses_scalar_epoch() ? sizeof(uint32_t) : 0u);
          info.owner_descriptor_file_offsets = planned.patch_abi.owner_descriptor_file_offsets;
          if (!append_moi_sync_lowering_commit(result.observation_plan(), result.errors, candidate,
                                               info, "barrier epoch patch", lowering_commits))
            return;
          patches.push_back(std::move(info));
          continue;
        }
        // A selectable-bank target may need setup around this otherwise-small body.
        // A claimed short island is therefore only an optimization; if the
        // final body does not fit, retain the island as the indirect entry to
        // the general appended body below.
        oversized_local_body_island = *local_body_offset;
      }
    }

    const uint64_t cave_text_offset = placement_planner.appended_end();
    const uint64_t return_text_offset = site.text_offset + site.size;
    // A branch-only scalar-spill owner has deliberately reserved no SGPRs for
    // indirect control flow.  If even a branch at the start of the appended
    // body could not return to the guest continuation, a real body (whose
    // return is later still) cannot be encoded.  Classify that site as a
    // placement omission before body construction: the builder reports
    // malformed requested encodings as errors, whereas lack of a route for
    // one otherwise-supported site must not abort the whole code object.
    if (cave_text_offset >= return_text_offset &&
        !compute_sopp_branch_simm16(cave_text_offset, return_text_offset) && !indirect_jump) {
      result.warnings.emplace_back(
          planned.branch_only_spill
              ? "ConSan MOI branch-only scalar-spill barrier has no encodable far return in " +
                    candidate.container_name
              : "ConSan MOI inline-shadow barrier has no encodable far return in " +
                    candidate.container_name);
      continue;
    }
    std::optional<std::vector<uint32_t>> cave_words =
        build_engine_barrier_body(planned, original_barrier_word, cave_text_offset,
                                  return_text_offset, std::nullopt, selectable_vgpr_bank_mode);
    if (!cave_words)
      return;

    std::optional<uint64_t> island_offset =
        barrier_island_layout
            ? std::optional<uint64_t>(barrier_island_layout->begin +
                                      planned_index * barrier_island_layout->slot_words *
                                          sizeof(uint32_t))
            : std::nullopt;
    if (island_offset && !compute_sopp_branch_simm16(site.text_offset, *island_offset))
      island_offset.reset();
    if (!island_offset &&
        !compute_sopp_branch_simm16(site.text_offset, cave_text_offset).has_value()) {
      island_offset = oversized_local_body_island
                          ? oversized_local_body_island
                          : local_indirect_allocator.claim_reachable(site.text_offset);
      if (!island_offset) {
        const bool has_dense_router =
            consan_arch_has_selectable_vgpr_bank(arch) ||
            (consan_is_capability_arch(arch) &&
             ((updates_exact_epoch && !consan_uses_gfx11_encoding(arch)) ||
              (emits_barrier_record && semantics.dense_barrier_router)));
        if (has_dense_router && indirect_jump &&
            !planned.patch_abi.owner_descriptor_file_offsets.empty()) {
          dense_fallback_indices.push_back(planned_index);
        } else {
          result.warnings.emplace_back(
              "ConSan MOI inline-shadow barrier epoch found no reachable indirect entry island "
              "in " +
              candidate.container_name);
        }
        continue;
      }
    }
    std::optional<DbiPatchPlacement> placement;
    if (island_offset) {
      std::string placement_error;
      placement = placement_planner.plan_indirect_appended(
          site.text_offset, site.size, cave_words->size() * sizeof(uint32_t), &placement_error);
      if (!placement)
        result.errors.emplace_back("ConSan MOI inline-shadow barrier epoch placement failed: " +
                                   placement_error);
    } else {
      placement = plan_prebuilt_appended_cave(placement_planner, site.text_offset, site.size,
                                              *cave_words, result.errors,
                                              "ConSan MOI inline-shadow barrier epoch patch");
    }
    if (!placement || placement->body_offset != cave_text_offset ||
        new_text.size() != placement->body_offset) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch has a stale cave mapping");
      return;
    }

    const uint64_t entry_target = island_offset.value_or(placement->body_offset);
    const auto fwd = compute_sopp_branch_simm16(placement->anchor_offset, entry_target);
    if (!fwd) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch forward branch is out of range");
      return;
    }
    if (!write_word_bytes(new_text, site.text_offset, build_s_branch(*fwd, arch))) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow barrier epoch patch could not rewrite barrier slot");
      return;
    }
    if (island_offset) {
      std::vector<uint32_t> island_words;
      if (!indirect_jump ||
          !append_moi_scc_preserving_indirect_jump(island_words, *island_offset,
                                                   placement->body_offset, indirect_jump->pc_sgpr,
                                                   indirect_jump->scc_save_sgpr,
                                                   /*capture_scc=*/true, arch) ||
          island_words.size() != kIndirectIslandWords ||
          *island_offset + indirect_island_bytes > new_text.size()) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow barrier epoch could not encode its entry island");
        return;
      }
      std::memcpy(new_text.data() + *island_offset, island_words.data(), indirect_island_bytes);
      ConSanPatchInfo island_info;
      island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      island_info.anchor_offset = site.text_offset;
      island_info.trampoline_offset = *island_offset;
      island_info.trampoline_size = static_cast<uint32_t>(indirect_island_bytes);
      island_info.owner_descriptor_file_offsets = planned.patch_abi.owner_descriptor_file_offsets;
      patches.push_back(std::move(island_info));
    }
    append_words_bytes(new_text, *cave_words);

    ConSanPatchInfo info;
    info.kind = barrier_patch_kind;
    info.anchor_offset = site.text_offset;
    info.trampoline_offset = cave_text_offset;
    info.original_size = site.size;
    info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));
    note_relocated_barrier_instruction(info, *cave_words, original_barrier_word, cave_text_offset,
                                       cave_text_offset +
                                           (planned.uses_scalar_epoch() ? sizeof(uint32_t) : 0u),
                                       emits_barrier_record);
    static_cast<ConSanPatchAbiEffects &>(info) = planned.patch_abi;
    if (!append_moi_sync_lowering_commit(result.observation_plan(), result.errors, candidate, info,
                                         "barrier epoch patch", lowering_commits))
      return;
    patches.push_back(info);
  }

  if (!dense_fallback_indices.empty()) {
    constexpr uint64_t kDispatcherWordsPerSite = 20u;

    std::vector<consan_detail::MoiDenseRouteSite> route_sites;
    route_sites.reserve(dense_fallback_indices.size());
    for (size_t index : dense_fallback_indices) {
      const MoiBarrierEvidenceSitePlan &candidate = planned_candidates[index].candidate;
      route_sites.push_back({
          .owner = {candidate.in_kernel ? ConSanProgramContainerKind::Kernel
                                        : ConSanProgramContainerKind::Function,
                    candidate.container_name},
          .anchor = candidate.site.text_offset,
          .source_index = index,
      });
    }
    const auto groups = consan_detail::partition_moi_dense_route_sites(route_sites, 0u);
    std::vector<std::pair<uint64_t, uint64_t>> occupied_ranges;
    occupied_ranges.reserve(result.patches.size() + patches.size() + selected_candidates.size());
    const auto append_occupied_range = [&](uint64_t begin, uint64_t size) {
      if (size != 0u && begin <= std::numeric_limits<uint64_t>::max() - size)
        occupied_ranges.emplace_back(begin, begin + size);
    };
    for (const ConSanCommittedPatchGeometry &patch : result.patches)
      append_occupied_range(patch.anchor_offset, patch.original_size);
    for (const ConSanCommittedPatchGeometry &patch : patches)
      append_occupied_range(patch.anchor_offset, patch.original_size);
    for (const MoiBarrierEvidenceSitePlan &candidate : selected_candidates)
      append_occupied_range(candidate.site.text_offset, candidate.site.size);
    const consan_detail::MoiOccupiedTextRanges occupied(std::move(occupied_ranges));
    std::map<uint64_t, uint64_t> claimed_hosts;
    const auto overlaps_existing = [&](uint64_t begin, uint64_t end) {
      if (occupied.overlaps(begin, end))
        return true;
      const auto claimed_after = claimed_hosts.lower_bound(end);
      return claimed_after != claimed_hosts.begin() && std::prev(claimed_after)->second > begin;
    };

    struct ReusableDenseAccessRelay {
      uint64_t island_offset = 0;
      size_t dispatcher_patch_index = 0;
      uint64_t reservation_end = 0;
    };
    std::map<std::vector<uint64_t>, std::vector<ReusableDenseAccessRelay>> reusable_access_relays;
    if (target->direct_call_form == ConSanDirectCallForm::SCallI64) {
      for (size_t patch_index = 0; patch_index < result.patches.size(); ++patch_index) {
        const ConSanPatchLoweringProduct &dispatcher = result.patches[patch_index];
        if (dispatcher.kind != ConSanPatchKind::TrampolineMoiIndirectBranchIsland ||
            dispatcher.original_size != 0u || !dispatcher.moi_dense_entry_island_offset ||
            dispatcher.owner_descriptor_file_offsets.empty()) {
          continue;
        }
        // Local barrier bodies selected earlier in this pass are not in
        // result.patches until the pass commits. They still bound a reused
        // access dispatcher; growing through one would overwrite an emitted
        // body and make both inventories describe the same bytes.
        const auto boundary = consan_detail::next_moi_trampoline_boundary(
            dispatcher.trampoline_offset, result.patches, patches);
        if (!boundary || dispatcher.trampoline_offset + dispatcher.trampoline_size > *boundary) {
          continue;
        }
        reusable_access_relays[dispatcher.owner_descriptor_file_offsets].push_back(
            {.island_offset = *dispatcher.moi_dense_entry_island_offset,
             .dispatcher_patch_index = patch_index,
             .reservation_end = *boundary});
      }
    }
    const auto find_reusable_access_relay =
        [&](const std::vector<size_t> &indices,
            const std::vector<uint64_t> &owners) -> std::optional<ReusableDenseAccessRelay> {
      // S_CALL_I64 access relays already derive their dispatch key from the
      // call return PC. Explicit-key relays must retain their separate
      // one-word barrier router.
      if (target->direct_call_form != ConSanDirectCallForm::SCallI64)
        return std::nullopt;
      const auto found = reusable_access_relays.find(owners);
      if (found == reusable_access_relays.end())
        return std::nullopt;
      for (const ReusableDenseAccessRelay &relay : found->second) {
        if (!std::ranges::all_of(indices, [&](size_t index) {
              return compute_sopp_branch_simm16(
                         planned_candidates[index].candidate.site.text_offset, relay.island_offset)
                  .has_value();
            })) {
          continue;
        }
        return relay;
      }
      return std::nullopt;
    };

    enum class ReusedDenseRelayResult : uint8_t {
      Unavailable,
      Applied,
      Failed,
    };
    const auto apply_reused_access_relay =
        [&](const std::vector<size_t> &indices, const std::vector<uint64_t> &owners,
            const ConSanMoiOperatingPoint &group_point) -> ReusedDenseRelayResult {
      const auto relay = find_reusable_access_relay(indices, owners);
      if (!relay)
        return ReusedDenseRelayResult::Unavailable;

      const auto dense_router =
          plan_moi_dense_router(options.moi_engine, moi_scalar_routing_state(group_point), arch);
      if (!dense_router || !dense_router->call_return_sgpr) {
        result.errors.emplace_back("ConSan MOI reused barrier relay has no compatible scalar ABI");
        return ReusedDenseRelayResult::Failed;
      }
      const ConSanIndirectJumpSgprs indirect_state = dense_router->indirect_jump;
      const uint16_t call_return_sgpr = *dense_router->call_return_sgpr;
      const auto build_direct_call = [&](int16_t simm16) -> std::optional<uint32_t> {
        return instrumentation::build_s_call_i64(call_return_sgpr, simm16, arch);
      };

      ConSanPatchLoweringProduct &dispatcher = result.patches[relay->dispatcher_patch_index];
      if (dispatcher.trampoline_size < sizeof(uint32_t) ||
          dispatcher.trampoline_offset + dispatcher.trampoline_size > new_text.size()) {
        result.errors.emplace_back(
            "ConSan MOI reused barrier relay found a stale access dispatcher");
        return ReusedDenseRelayResult::Failed;
      }
      std::vector<uint32_t> dispatcher_words(dispatcher.trampoline_size / sizeof(uint32_t));
      std::memcpy(dispatcher_words.data(), new_text.data() + dispatcher.trampoline_offset,
                  dispatcher.trampoline_size);
      if (dispatcher_words.empty() || dispatcher_words.back() != build_s_endpgm(arch)) {
        result.errors.emplace_back(
            "ConSan MOI reused barrier relay found an unterminated access dispatcher");
        return ReusedDenseRelayResult::Failed;
      }
      dispatcher_words.pop_back();

      struct StagedReusedBarrier {
        size_t planned_index = 0;
        uint64_t site_offset = 0;
        uint32_t call_word = 0;
        std::vector<uint32_t> body_words;
        ConSanPatchInfo patch;
      };
      std::vector<StagedReusedBarrier> staged_barriers;
      staged_barriers.reserve(indices.size());
      uint64_t staged_body_bytes = 0u;
      for (size_t index : indices) {
        const PlannedInlineEpochBarrier &planned = planned_candidates[index];
        const MoiBarrierEvidenceSitePlan &candidate = planned.candidate;
        const ConSanBarrierSite &site = candidate.site;
        uint32_t original_barrier_word = 0;
        std::memcpy(&original_barrier_word, old_text.data() + site.text_offset,
                    sizeof(original_barrier_word));
        const std::optional<uint16_t> selectable_vgpr_bank_mode =
            consan_selectable_vgpr_bank_mode_at(arch, active_bytes, candidate.text_file_offset,
                                                candidate.container_entry_text_offset,
                                                site.file_offset);
        if (staged_body_bytes > std::numeric_limits<uint64_t>::max() - new_text.size()) {
          result.errors.emplace_back("ConSan MOI reused barrier relay body offsets overflowed");
          return ReusedDenseRelayResult::Failed;
        }
        const uint64_t body_offset = new_text.size() + staged_body_bytes;
        std::optional<std::vector<uint32_t>> body_words = build_engine_barrier_body(
            planned, original_barrier_word, body_offset, site.text_offset + site.size,
            call_return_sgpr, selectable_vgpr_bank_mode);
        if (!body_words)
          return ReusedDenseRelayResult::Failed;
        const uint64_t body_bytes = body_words->size() * sizeof(uint32_t);
        if (body_bytes > std::numeric_limits<uint64_t>::max() - staged_body_bytes) {
          result.errors.emplace_back("ConSan MOI reused barrier relay body size overflowed");
          return ReusedDenseRelayResult::Failed;
        }
        staged_body_bytes += body_bytes;

        if (!append_moi_call_return_match(dispatcher_words, dispatcher.trampoline_offset,
                                          site.text_offset + sizeof(uint32_t),
                                          indirect_state.pc_sgpr, call_return_sgpr, arch)) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay could not encode its relocatable caller match");
          return ReusedDenseRelayResult::Failed;
        }
        std::vector<uint32_t> matched_words;
        const uint64_t matched_offset =
            dispatcher.trampoline_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
        if (!append_moi_scc_preserving_indirect_jump(
                matched_words, matched_offset, body_offset, indirect_state.pc_sgpr,
                indirect_state.scc_save_sgpr, /*capture_scc=*/false, arch)) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay could not encode a dispatch target");
          return ReusedDenseRelayResult::Failed;
        }
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay could not encode a dispatch branch");
          return ReusedDenseRelayResult::Failed;
        }
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());

        const auto call_delta = compute_sopp_branch_simm16(site.text_offset, relay->island_offset);
        const auto call = call_delta ? build_direct_call(*call_delta) : std::nullopt;
        if (!call) {
          result.errors.emplace_back(
              "ConSan MOI reused barrier relay could not encode a barrier call");
          return ReusedDenseRelayResult::Failed;
        }

        ConSanPatchInfo info;
        info.kind = barrier_patch_kind;
        info.anchor_offset = site.text_offset;
        info.trampoline_offset = body_offset;
        info.original_size = site.size;
        info.trampoline_size = static_cast<uint32_t>(body_words->size() * sizeof(uint32_t));
        note_relocated_barrier_instruction(
            info, *body_words, original_barrier_word, body_offset,
            body_offset + (planned.uses_scalar_epoch() ? sizeof(uint32_t) : 0u),
            emits_barrier_record);
        static_cast<ConSanPatchAbiEffects &>(info) = planned.patch_abi;
        staged_barriers.push_back({.planned_index = index,
                                   .site_offset = site.text_offset,
                                   .call_word = *call,
                                   .body_words = std::move(*body_words),
                                   .patch = std::move(info)});
      }

      dispatcher_words.push_back(build_s_endpgm(arch));
      const uint64_t dispatcher_bytes = dispatcher_words.size() * sizeof(uint32_t);
      if (dispatcher.trampoline_offset + dispatcher_bytes > relay->reservation_end) {
        return ReusedDenseRelayResult::Unavailable;
      }

      // Everything above is side-effect-free with respect to the transformed
      // image. Commit only after the exact dispatcher encoding is known to fit
      // its access-relay reservation.
      new_text.reserve(new_text.size() + staged_body_bytes);
      for (StagedReusedBarrier &staged : staged_barriers) {
        append_words_bytes(new_text, staged.body_words);
        std::memcpy(new_text.data() + staged.site_offset, &staged.call_word, sizeof(uint32_t));
        if (!append_moi_sync_lowering_commit(result.observation_plan(), result.errors,
                                             planned_candidates[staged.planned_index].candidate,
                                             staged.patch, "barrier epoch patch", lowering_commits))
          return ReusedDenseRelayResult::Failed;
        patches.push_back(std::move(staged.patch));
      }
      std::memcpy(new_text.data() + dispatcher.trampoline_offset, dispatcher_words.data(),
                  dispatcher_bytes);
      dispatcher.trampoline_size = static_cast<uint32_t>(dispatcher_bytes);
      return ReusedDenseRelayResult::Applied;
    };

    for (const auto &group : groups) {
      const std::vector<size_t> &indices = group.source_indices;
      if (indices.empty())
        continue;
      const MoiBarrierEvidenceSitePlan &first_candidate =
          planned_candidates[indices.front()].candidate;
      if (first_candidate.container_code_size == 0u)
        continue;
      const uint64_t owner_begin = first_candidate.container_entry_text_offset;
      const uint64_t owner_end = owner_begin + first_candidate.container_code_size;
      const std::string &owner_name = first_candidate.container_name;
      const std::vector<uint64_t> &group_owner =
          planned_candidates[indices.front()].patch_abi.owner_descriptor_file_offsets;
      if (group_owner.empty() || std::ranges::any_of(indices, [&](size_t index) {
            return planned_candidates[index].patch_abi.owner_descriptor_file_offsets != group_owner;
          })) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier relay has inconsistent dispatch ownership");
        return;
      }
      ConSanMoiOperatingPoint group_point = operating_point;
      if (!apply_moi_transient_sgpr_assignment(options, group_point, result.moi_operating_point,
                                               group_owner)) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier relay lost its transient scalar assignment");
        return;
      }
      switch (apply_reused_access_relay(indices, group_owner, group_point)) {
      case ReusedDenseRelayResult::Applied:
        continue;
      case ReusedDenseRelayResult::Failed:
        return;
      case ReusedDenseRelayResult::Unavailable:
        break;
      }
      const MoiScalarRoutingState routing_state = moi_scalar_routing_state(group_point);
      const auto dense_router = plan_moi_dense_router(options.moi_engine, routing_state, arch);
      if (!dense_router) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier relay requires scalar special-state storage");
        return;
      }
      const MoiDenseBarrierRouterPlan &barrier_router = dense_router->barrier;
      const uint16_t jump_pc_sgpr = barrier_router.indirect_jump.pc_sgpr;
      const uint16_t saved_scc_sgpr = barrier_router.indirect_jump.scc_save_sgpr;
      const uint16_t key_sgpr = barrier_router.dispatch_key_sgpr;
      const uint16_t call_return_sgpr = barrier_router.call_return_sgpr;
      const bool derive_key_at_entry = barrier_router.derive_key_at_entry;
      const bool key_encodes_scc = derive_key_at_entry && key_sgpr == saved_scc_sgpr;
      const auto build_direct_call = [&](int16_t simm16) -> std::optional<uint32_t> {
        if (target->direct_call_form == ConSanDirectCallForm::SCallI64)
          return instrumentation::build_s_call_i64(call_return_sgpr, simm16, arch);
        if (target->direct_call_form == ConSanDirectCallForm::SCallB64)
          return build_s_call_b64(call_return_sgpr, simm16, arch);
        return std::nullopt;
      };
      const uint64_t island_words_count =
          moi_dense_entry_island_words(derive_key_at_entry, key_encodes_scc);
      const uint64_t host_words_count = island_words_count + 1u;
      const uint64_t host_bytes = host_words_count * sizeof(uint32_t);
      std::vector<uint64_t> anchors;
      anchors.reserve(indices.size());
      for (size_t index : indices)
        anchors.push_back(planned_candidates[index].candidate.site.text_offset);
      const ConSanCandidateResourcePlan *group_resource =
          resource_plan_index.find(planned_candidates[indices.front()].candidate.site.text_offset);
      if (group_resource == nullptr) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow barrier epoch has no dense relay resource plan in " +
            owner_name);
        continue;
      }
      const std::vector<MoiSgprRange> host_bootstrap_ranges = moi_relocatable_host_scalar_ranges(
          options, group_point, MoiRelocatableHostScalarState::BarrierRouter, arch);
      auto dense_host = find_moi_dense_relay_host_for_owners(
          resource_state, old_text,
          {
              .owner_begin = owner_begin,
              .owner_end = owner_end,
              .host_word_count = static_cast<size_t>(host_words_count),
              .anchors = anchors,
              .preapplied_reserved_ranges = moi_resource_reserved_ranges(resource_state),
              .owner_descriptor_file_offsets = group_resource->owner_descriptor_file_offsets,
              .bootstrap_ranges = host_bootstrap_ranges,
              .claimed_host_ranges = {},
          },
          overlaps_existing);
      if (!dense_host) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow barrier epoch found no relocatable dense relay host in " +
            owner_name);
        continue;
      }
      const uint64_t host_offset = dense_host->host_offset;
      std::vector<uint32_t> displaced_words = std::move(dense_host->displaced_words);
      claimed_hosts.emplace(host_offset, host_offset + host_bytes);

      const uint64_t dispatcher_offset = new_text.size();
      const uint64_t dispatcher_reservation =
          (indices.size() + 1u) * kDispatcherWordsPerSite * sizeof(uint32_t);
      while (new_text.size() < dispatcher_offset + dispatcher_reservation)
        append_word_bytes(new_text, build_s_nop(0, arch));

      struct DenseBarrierBody {
        size_t planned_index = 0;
        uint64_t offset = 0;
        uint64_t relocated_guest_instruction_offset = 0;
        std::vector<uint32_t> words;
      };
      std::vector<DenseBarrierBody> bodies;
      bodies.reserve(indices.size());
      for (size_t index : indices) {
        const PlannedInlineEpochBarrier &planned = planned_candidates[index];
        const ConSanBarrierSite &site = planned.candidate.site;
        uint32_t original_barrier_word = 0;
        std::memcpy(&original_barrier_word, old_text.data() + site.text_offset,
                    sizeof(original_barrier_word));
        const MoiBarrierEvidenceSitePlan &candidate = planned.candidate;
        const std::optional<uint16_t> selectable_vgpr_bank_mode =
            consan_selectable_vgpr_bank_mode_at(arch, active_bytes, candidate.text_file_offset,
                                                candidate.container_entry_text_offset,
                                                site.file_offset);
        const uint64_t body_offset = new_text.size();
        std::optional<std::vector<uint32_t>> body_words = build_engine_barrier_body(
            planned, original_barrier_word, body_offset, site.text_offset + site.size,
            derive_key_at_entry ? std::nullopt : std::optional<uint16_t>(call_return_sgpr),
            selectable_vgpr_bank_mode);
        if (!body_words)
          return;
        append_words_bytes(new_text, *body_words);
        bodies.push_back({.planned_index = index,
                          .offset = body_offset,
                          .relocated_guest_instruction_offset =
                              body_offset + (planned.uses_scalar_epoch() ? sizeof(uint32_t) : 0u),
                          .words = std::move(*body_words)});
      }

      const uint64_t host_body_offset = new_text.size();
      std::vector<uint32_t> host_words = displaced_words;
      if (!append_moi_scc_preserving_indirect_jump(host_words, host_body_offset,
                                                   host_offset + host_bytes, jump_pc_sgpr,
                                                   saved_scc_sgpr, /*capture_scc=*/true, arch)) {
        result.errors.emplace_back("ConSan MOI dense barrier relay could not return from its host");
        return;
      }
      append_words_bytes(new_text, host_words);

      const uint64_t island_offset = host_offset + sizeof(uint32_t);
      const auto island_words =
          build_moi_dense_barrier_entry_island(island_offset, dispatcher_offset,
                                               {.jump_pc_sgpr = jump_pc_sgpr,
                                                .saved_scc_sgpr = saved_scc_sgpr,
                                                .key_sgpr = key_sgpr,
                                                .call_return_sgpr = call_return_sgpr},
                                               derive_key_at_entry, arch);
      if (!island_words || island_words->size() != island_words_count) {
        result.errors.emplace_back("ConSan MOI dense barrier relay could not encode its island");
        return;
      }
      std::memcpy(new_text.data() + island_offset, island_words->data(),
                  island_words->size() * sizeof(uint32_t));
      const auto host_call = build_direct_call(/*next instruction=*/0);
      if (!host_call) {
        result.errors.emplace_back("ConSan MOI dense barrier relay could not encode its host call");
        return;
      }
      std::memcpy(new_text.data() + host_offset, &*host_call, sizeof(uint32_t));

      std::vector<uint32_t> dispatcher_words;
      if (key_encodes_scc) {
        dispatcher_words.push_back(
            build_s_lshr_b32(jump_pc_sgpr, key_sgpr, scalar_positive_inline_u32(1), arch));
      }
      if (!derive_key_at_entry && target->direct_call_form != ConSanDirectCallForm::SCallI64) {
        const auto subtract_pc =
            instrumentation::build_s_sub_u32(key_sgpr, call_return_sgpr, jump_pc_sgpr, arch);
        if (!subtract_pc) {
          result.errors.emplace_back("ConSan MOI dense barrier relay could not encode its key");
          return;
        }
        dispatcher_words.push_back(*subtract_pc);
        dispatcher_words.push_back(build_s_add_u32(key_sgpr, key_sgpr, /*literal=*/255u, arch));
        dispatcher_words.push_back(static_cast<uint32_t>(dispatcher_offset - island_offset));
      }

      const auto append_dispatch_target = [&](uint64_t caller_return,
                                              uint64_t destination) -> bool {
        if (target->direct_call_form == ConSanDirectCallForm::SCallI64 && !derive_key_at_entry) {
          if (!append_moi_call_return_match(dispatcher_words, dispatcher_offset, caller_return,
                                            jump_pc_sgpr, call_return_sgpr, arch))
            return false;
        } else {
          const auto compare = instrumentation::build_s_cmp_eq_u32(
              key_encodes_scc ? jump_pc_sgpr : key_sgpr, /*literal=*/255u, arch);
          if (!compare)
            return false;
          dispatcher_words.push_back(*compare);
          dispatcher_words.push_back(static_cast<uint32_t>(caller_return - island_offset) /
                                     (key_encodes_scc ? 2u : 1u));
        }
        std::vector<uint32_t> matched_words;
        if (key_encodes_scc) {
          const ConSanTargetProfile *matched_target = consan_target_profile(arch);
          if (matched_target == nullptr || !consan_detail::append_restore_moi_scc_from_route_key(
                                               matched_words,
                                               consan_detail::MoiEncodedSccRestoreRequest{
                                                   .encoded_sgpr = key_sgpr,
                                                   .normalize_encoded_sgpr = true,
                                               },
                                               *matched_target)) {
            return false;
          }
        }
        const uint64_t matched_offset =
            dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
        if (!append_moi_scc_preserving_indirect_jump(matched_words, matched_offset, destination,
                                                     jump_pc_sgpr, saved_scc_sgpr,
                                                     /*capture_scc=*/false, arch))
          return false;
        const auto skip =
            instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
        if (!skip)
          return false;
        dispatcher_words.push_back(*skip);
        dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
        return true;
      };
      if (!append_dispatch_target(/*caller_return=*/island_offset, host_body_offset)) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier relay could not encode its host dispatch");
        return;
      }
      for (const DenseBarrierBody &body : bodies) {
        const ConSanBarrierSite &site = planned_candidates[body.planned_index].candidate.site;
        if (!append_dispatch_target(site.text_offset + sizeof(uint32_t), body.offset)) {
          result.errors.emplace_back(
              "ConSan MOI dense barrier relay could not encode a barrier dispatch");
          return;
        }
      }
      dispatcher_words.push_back(build_s_endpgm(arch));
      if (dispatcher_words.size() * sizeof(uint32_t) > dispatcher_reservation) {
        result.errors.emplace_back(
            "ConSan MOI dense barrier dispatcher exceeds its reserved relay space");
        return;
      }
      std::memcpy(new_text.data() + dispatcher_offset, dispatcher_words.data(),
                  dispatcher_words.size() * sizeof(uint32_t));

      ConSanPatchInfo host_info;
      host_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      host_info.anchor_offset = host_offset;
      host_info.original_size = static_cast<uint32_t>(host_bytes);
      host_info.trampoline_offset = host_body_offset;
      host_info.trampoline_size = static_cast<uint32_t>(host_words.size() * sizeof(uint32_t));
      host_info.owner_descriptor_file_offsets = group_owner;
      patches.push_back(std::move(host_info));
      ConSanPatchInfo dispatcher_info;
      dispatcher_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
      dispatcher_info.anchor_offset =
          planned_candidates[indices.front()].candidate.site.text_offset;
      dispatcher_info.trampoline_offset = dispatcher_offset;
      dispatcher_info.trampoline_size =
          static_cast<uint32_t>(dispatcher_words.size() * sizeof(uint32_t));
      dispatcher_info.owner_descriptor_file_offsets = group_owner;
      patches.push_back(std::move(dispatcher_info));

      for (const DenseBarrierBody &body : bodies) {
        const PlannedInlineEpochBarrier &planned = planned_candidates[body.planned_index];
        const ConSanBarrierSite &site = planned.candidate.site;
        const auto call_delta = compute_sopp_branch_simm16(site.text_offset, island_offset);
        const auto call = call_delta ? build_direct_call(*call_delta) : std::nullopt;
        if (!call) {
          result.errors.emplace_back(
              "ConSan MOI dense barrier relay could not encode a barrier call");
          return;
        }
        std::memcpy(new_text.data() + site.text_offset, &*call, sizeof(uint32_t));
        ConSanPatchInfo info;
        info.kind = barrier_patch_kind;
        info.anchor_offset = site.text_offset;
        info.trampoline_offset = body.offset;
        info.original_size = site.size;
        info.trampoline_size = static_cast<uint32_t>(body.words.size() * sizeof(uint32_t));
        uint32_t original_barrier_word = 0u;
        if (emits_barrier_record) {
          const uint64_t site_offset =
              planned_candidates[body.planned_index].candidate.site.text_offset;
          if (site_offset <= old_text.size() - sizeof(uint32_t))
            std::memcpy(&original_barrier_word, old_text.data() + site_offset,
                        sizeof(original_barrier_word));
        }
        note_relocated_barrier_instruction(info, body.words, original_barrier_word, body.offset,
                                           body.relocated_guest_instruction_offset,
                                           emits_barrier_record);
        static_cast<ConSanPatchAbiEffects &>(info) = planned.patch_abi;
        if (!append_moi_sync_lowering_commit(result.observation_plan(), result.errors,
                                             planned.candidate, info, "barrier epoch patch",
                                             lowering_commits))
          return;
        patches.push_back(std::move(info));
      }
    }
  }

  if (patches.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow barrier epoch patch found no patchable barriers");
    return;
  }

  std::unordered_set<uint64_t> emitted_owner_descriptors;
  bool has_unscoped_emitted_patch = false;
  for (const ConSanPatchLoweringProduct &patch : patches) {
    has_unscoped_emitted_patch |= patch.owner_descriptor_file_offsets.empty();
    emitted_owner_descriptors.insert(patch.owner_descriptor_file_offsets.begin(),
                                     patch.owner_descriptor_file_offsets.end());
  }
  merge_maximum_descriptor_extents(visible_evidence_vgpr_requirements,
                                   persistent_epoch_vgpr_requirements);
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
  if (!apply_moi_descriptor_requirements(
          patcher, code_object, result.program_inventory, visible_evidence_vgpr_requirements,
          visible_evidence_sgpr_requirements, visible_evidence_private_requirements, nullptr,
          nullptr, arch, "ConSan MOI inline-shadow barrier epoch patch", result.errors)) {
    return;
  }
  (void)publish_moi_sync_patch(result, patcher, new_text, options.patched_image_growth_limit,
                               "MOI inline-shadow barrier epoch patch", "barrier epoch patch",
                               std::move(lowering_commits), std::move(patches));
}

void try_apply_inline_shadow_barrier_patch(std::span<const uint8_t> bytes,
                                           const ConSanOptions &options,
                                           const ConSanMoiOperatingPoint &operating_point,
                                           rj_code_arch_t arch,
                                           MoiResourcePlanningState &resource_state,
                                           const MoiObjectModeSemantics &semantics,
                                           ConSanTransformArtifacts &result) {
  if (!options.moi_track_barriers)
    return;
  ConSanMoiOperatingPoint barrier_point = operating_point;
  if (barrier_point.has_inline_moi_scalar_spill())
    barrier_point.moi_exec_save_sgpr.reset();
  try_apply_shared_barrier_patch(bytes, options, barrier_point, arch, resource_state, std::nullopt,
                                 semantics, result);
}
