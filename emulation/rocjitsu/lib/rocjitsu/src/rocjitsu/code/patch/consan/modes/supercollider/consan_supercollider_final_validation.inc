void validate_supercollider_final_semantics(const FinalValidationEnvironment &environment,
                                            const ConSanFinalValidationInput &result,
                                            const ConSanPristineValidationInventory *pristine,
                                            std::vector<std::string> &errors) {
  const size_t perturbation_patch_count = static_cast<size_t>(std::ranges::count(
      result.patches, ConSanPatchKind::TrampolineScPerturbation, &ConSanPatchInfo::kind));
  const bool composite = has_consan_patch_phase(result, ConSanPatchPhase::Mutation);
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineScPerturbation &&
        (patch.perturbation_edge || !patch.perturbation_sequence_identity.empty() ||
         !patch.perturbation_source_sequence_identity.empty() ||
         patch.perturbation_composite_atomic_overlap ||
         patch.perturbation_composite_removed_boundary)) {
      errors.emplace_back(
          "ConSan SC perturbation proof found semantic metadata on another patch kind");
    }
  }
  if (result.mutation.perturbation.applied != perturbation_patch_count) {
    errors.emplace_back("ConSan SC perturbation proof found inconsistent applied patch count");
  }
  if (perturbation_patch_count == 0u)
    return;
  if (!environment.original_text.available || !environment.replacement_text.available) {
    errors.emplace_back(
        "ConSan SC perturbation proof requires one original and replacement text section");
    return;
  }

  if (pristine == nullptr || !pristine->analysis_succeeded) {
    errors.emplace_back(
        "ConSan SC perturbation proof could not rederive pristine synchronization semantics");
    return;
  }

  const std::span<const uint8_t> original_text = environment.original_text.bytes;
  const std::span<const uint8_t> replacement_text = environment.replacement_text.bytes;
  const rj_code_arch_t arch = environment.arch();
  const uint32_t nop = build_s_nop(0, arch);
  std::unordered_set<std::string> proven_edges;
  const SynchronizationInventoryView pristine_events = pristine->program_inventory.sync();

  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineScPerturbation)
      continue;
    if (patch.phase != ConSanPatchPhase::Instrumentation) {
      errors.emplace_back("ConSan SC perturbation proof found a non-instrumentation patch phase");
      continue;
    }
    if (!patch.perturbation_edge || patch.perturbation_sequence_identity.empty()) {
      errors.emplace_back("ConSan SC perturbation proof found incomplete semantic identity");
      continue;
    }
    if (composite && patch.perturbation_source_sequence_identity.empty()) {
      errors.emplace_back("ConSan composite perturbation proof found no pristine sequence");
      continue;
    }
    const std::string &proof_sequence_identity = composite
                                                     ? patch.perturbation_source_sequence_identity
                                                     : patch.perturbation_sequence_identity;
    const std::string proven_edge =
        proof_sequence_identity +
        (*patch.perturbation_edge == ConSanPerturbationEdge::Release ? "|release" : "|acquire");
    if (!proven_edges.insert(proven_edge).second) {
      errors.emplace_back("ConSan SC perturbation proof found a duplicate sequence edge");
      continue;
    }
    if (patch.original_size != sizeof(uint32_t) && patch.original_size != 2u * sizeof(uint32_t) &&
        patch.original_size != 3u * sizeof(uint32_t)) {
      errors.emplace_back("ConSan SC perturbation proof found an unsupported boundary size");
      continue;
    }
    const uint32_t expected_trampoline_size = patch.original_size + 2u * sizeof(uint32_t);
    if (patch.trampoline_size != expected_trampoline_size) {
      errors.emplace_back("ConSan SC perturbation proof found a non-exact trampoline size");
      continue;
    }
    if (patch.anchor_offset > replacement_text.size() ||
        patch.original_size > replacement_text.size() - patch.anchor_offset ||
        patch.trampoline_offset > replacement_text.size() ||
        patch.trampoline_size > replacement_text.size() - patch.trampoline_offset) {
      errors.emplace_back("ConSan SC perturbation proof found an out-of-range patch");
      continue;
    }

    const ProgramInventory &semantic_inventory = pristine->program_inventory;
    const std::span<const ConSanPerturbationCandidate> semantic_candidates =
        pristine->perturbation_candidates;
    const auto candidate_matches = [&](const ConSanPerturbationCandidate &candidate) {
      const ConSanSyncSequence *sequence = pristine_events.find_sequence(candidate.sequence);
      return candidate.eligible && sequence != nullptr &&
             sequence->identity == proof_sequence_identity &&
             candidate.edge == *patch.perturbation_edge;
    };
    const size_t candidate_count =
        static_cast<size_t>(std::ranges::count_if(semantic_candidates, candidate_matches));
    const auto candidate = std::ranges::find_if(semantic_candidates, candidate_matches);
    if (candidate_count != 1u || candidate == semantic_candidates.end()) {
      errors.emplace_back(
          "ConSan SC perturbation proof could not match one pristine admitted sequence edge");
      continue;
    }
    const ConSanSyncSequence *sequence = pristine_events.find_sequence(candidate->sequence);
    if (sequence == nullptr) {
      errors.emplace_back("ConSan SC perturbation proof lost its sequence arena edge");
      continue;
    }
    const ConSanProgramContainer *sequence_container = pristine_events.container(*sequence);
    if (sequence_container == nullptr) {
      errors.emplace_back("ConSan SC perturbation proof lost its sequence container");
      continue;
    }
    if (sequence_container->is_kernel()) {
      const ConSanProgramContainer *owner =
          semantic_inventory.find_kernel_by_name(sequence_container->name);
      if (owner == nullptr || patch.owner_descriptor_file_offsets.size() != 1u ||
          patch.owner_descriptor_file_offsets.front() != owner->descriptor_file_offset) {
        errors.emplace_back("ConSan SC perturbation proof did not retain its exact kernel owner");
        continue;
      }
    } else if (!patch.owner_descriptor_file_offsets.empty()) {
      errors.emplace_back(
          "ConSan SC perturbation proof attached a kernel owner to a local function");
      continue;
    }
    if (!sequence->basic_block_index ||
        !consan_sync_confidence_meets(sequence->confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        perturbation_rejection_reason(pristine_events, *sequence, candidate->kind,
                                      *patch.perturbation_edge) !=
            ConSanPerturbationRejectionReason::None) {
      errors.emplace_back("ConSan SC perturbation proof could not rederive sequence role, "
                          "confidence, container, and block");
      continue;
    }
    const size_t outer_member_index = *patch.perturbation_edge == ConSanPerturbationEdge::Release
                                          ? 0u
                                          : sequence->member_event_ids.size() - 1u;
    const ConSanSyncEvent *anchor_event =
        pristine_events.find_event(sequence->member_event_ids[outer_member_index]);
    const ConSanProgramSite *anchor_source =
        anchor_event == nullptr
            ? nullptr
            : pristine->program_inventory.program_site(anchor_event->source_site);
    if (candidate->anchor_event != sequence->member_event_ids[outer_member_index] ||
        anchor_event == nullptr || anchor_source == nullptr ||
        anchor_source->container != sequence_container->id ||
        anchor_event->text_offset() > original_text.size() ||
        anchor_source->size() > original_text.size() - anchor_event->text_offset() ||
        patch.original_size != anchor_source->size()) {
      errors.emplace_back(
          "ConSan SC perturbation proof did not target the rederived outer sequence member");
      continue;
    }
    const auto forward = compute_sopp_branch_simm16(patch.anchor_offset, patch.trampoline_offset);
    uint32_t anchor_word = 0;
    std::memcpy(&anchor_word, replacement_text.data() + patch.anchor_offset, sizeof(anchor_word));
    if (!forward || anchor_word != build_s_branch(*forward, arch))
      errors.emplace_back("ConSan SC perturbation proof found an invalid anchor branch");
    for (uint32_t offset = sizeof(uint32_t); offset < patch.original_size;
         offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, replacement_text.data() + patch.anchor_offset + offset, sizeof(word));
      if (word != nop) {
        errors.emplace_back("ConSan SC perturbation proof found a non-NOP anchor tail");
        break;
      }
    }

    const uint32_t original_body_offset =
        *patch.perturbation_edge == ConSanPerturbationEdge::Release ? sizeof(uint32_t) : 0u;
    const uint8_t *body = replacement_text.data() + patch.trampoline_offset + original_body_offset;
    if (patch.perturbation_composite_atomic_overlap && composite) {
      const auto mutation = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &item) {
        return item.phase == ConSanPatchPhase::Mutation &&
               item.anchor_offset == patch.anchor_offset &&
               item.original_size == patch.original_size &&
               (item.kind == ConSanPatchKind::InlineAtomicAddressRewrite ||
                item.kind == ConSanPatchKind::InlineAtomicOrderRewrite ||
                item.kind == ConSanPatchKind::InlineAtomicScopeRewrite);
      });
      if (mutation == result.patches.end() ||
          patch.perturbation_composite_removed_boundary !=
              (mutation->kind == ConSanPatchKind::InlineAtomicOrderRewrite)) {
        errors.emplace_back(
            "ConSan SC perturbation proof could not match its exact overlapping mutation");
      } else if (patch.perturbation_composite_removed_boundary) {
        bool all_nops = true;
        for (uint32_t offset = 0; offset < patch.original_size; offset += sizeof(uint32_t)) {
          uint32_t word = 0;
          std::memcpy(&word, body + offset, sizeof(word));
          all_nops &= word == nop;
        }
        if (!all_nops)
          errors.emplace_back("ConSan SC perturbation proof resurrected a removed cache operation");
      }
    } else if (std::memcmp(original_text.data() + anchor_event->text_offset(), body,
                           patch.original_size) != 0) {
      errors.emplace_back(
          "ConSan SC perturbation proof did not preserve the exact original boundary bytes");
    }
    const uint32_t sleep_body_offset =
        *patch.perturbation_edge == ConSanPerturbationEdge::Release ? 0u : patch.original_size;
    uint32_t sleep_word = 0;
    std::memcpy(&sleep_word, replacement_text.data() + patch.trampoline_offset + sleep_body_offset,
                sizeof(sleep_word));
    bool bounded_sleep = false;
    for (uint16_t immediate = 1; immediate <= 15; ++immediate)
      bounded_sleep |= sleep_word == build_s_sleep(immediate, arch);
    if (!bounded_sleep) {
      errors.emplace_back(
          "ConSan SC perturbation proof found no single bounded sleep on the declared side");
    }

    const uint64_t return_offset =
        patch.trampoline_offset + patch.trampoline_size - sizeof(uint32_t);
    const auto ret =
        compute_sopp_branch_simm16(return_offset, patch.anchor_offset + patch.original_size);
    uint32_t return_word = 0;
    std::memcpy(&return_word, replacement_text.data() + return_offset, sizeof(return_word));
    if (!ret || return_word != build_s_branch(*ret, arch)) {
      errors.emplace_back("ConSan SC perturbation proof found an invalid exact boundary return");
    }
  }
  return;
}
