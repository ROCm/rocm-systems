struct SampledBarrierWindow {
  uint32_t first_slot = 0;
  uint32_t bank_count = 1;

  auto operator<=>(const SampledBarrierWindow &) const = default;
};

struct SampledBarrierCandidate {
  const ConSanSyncSequence *sequence = nullptr;
  MoiBarrierEvidenceSitePlan barrier;
  const ConSanBarrierSite *site = nullptr;
  uint64_t owner_descriptor_file_offset = 0;
  std::vector<SampledBarrierWindow> windows;
  bool uses_cluster_workgroup_id = false;
};

/// Resolve one exact, immutable emission product at the Sampled sync planning
/// boundary. Direct and dense barriers and every atomic route consume only
/// this product; broad request, resource, and operating-point state cannot be
/// rediscovered while sizing or encoding a cave.
[[nodiscard]] std::optional<MoiSampledSyncEmissionPlan> plan_sampled_sync_emission(
    const ConSanRequest &request, const BoundRuntimeResources &bound_resources,
    const ConSanMoiOperatingPoint &base_point, const ConSanMoiOperatingPoint &accepted_point,
    std::span<const uint64_t> owner_descriptors, const ConSanMoiPrivateStateLayout *private_layout,
    uint16_t scratch_vgpr, uint16_t scratch_owner_vgpr, uint16_t scratch_epoch_vgpr,
    bool transient_assignment_already_resolved, std::string_view diagnostic_subject,
    std::vector<std::string> &errors) {
  if (!bound_resources.moi_report_buffer_address) {
    errors.emplace_back("ConSan MOI " + std::string(diagnostic_subject) +
                        " lacks a bound report buffer");
    return std::nullopt;
  }
  ConSanMoiOperatingPoint point = base_point;
  if (!transient_assignment_already_resolved &&
      !apply_moi_transient_sgpr_assignment(request, point, accepted_point, owner_descriptors)) {
    errors.emplace_back("ConSan MOI " + std::string(diagnostic_subject) +
                        " lost its transient scalar assignment");
    return std::nullopt;
  }
  ConSanMoiOwnerEpochVgprSources owner_epoch_vgprs;
  if (private_layout || point.moi_persistent_sgprs.complete()) {
    owner_epoch_vgprs = {
        .owner = scratch_owner_vgpr,
        .epoch = scratch_epoch_vgpr,
    };
  } else if (!apply_moi_persistent_vgpr_assignment(point, accepted_point, owner_descriptors)) {
    errors.emplace_back("ConSan MOI " + std::string(diagnostic_subject) +
                        " lost its persistent VGPR assignment");
    return std::nullopt;
  } else {
    owner_epoch_vgprs = moi_owner_epoch_vgpr_sources(point.moi_owner_epoch_vgprs);
  }
  const auto workgroup_sources = moi_exact_entry_workgroup_sources(
      point, private_layout ? &private_layout->exact_workgroup_offsets : nullptr);
  if (!workgroup_sources)
    return std::nullopt;
  return MoiSampledSyncEmissionPlan{
      .report_buffer_address = *bound_resources.moi_report_buffer_address,
      .report_generation = bound_resources.moi_report_generation,
      .exec_save_sgpr = point.moi_exec_save_sgpr,
      .automatic_private_epoch = point.automatic_moi_private_epoch,
      .workitem_owner = request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
      .persistent_sgprs = point.moi_persistent_sgprs,
      .scalar_abi = plan_moi_scalar_abi(request.moi_engine, moi_scalar_preservation_state(point)),
      .dispatch_id = consan_moi_detail::moi_bound_dispatch_id_sources({point, bound_resources}),
      .workgroup_sources = *workgroup_sources,
      .owner_epoch_vgprs = owner_epoch_vgprs,
      .scratch_vgpr = scratch_vgpr,
  };
}
using SampledCfgDistance = MoiCfgDistance;

[[nodiscard]] std::optional<SampledCfgDistance>
sampled_cfg_forward_distance(const MoiResourcePlanningState &state, uint64_t descriptor_offset,
                             uint64_t from_text_offset, uint64_t to_text_offset) {
  const auto index = make_moi_cfg_forward_distance_index(state, descriptor_offset, to_text_offset);
  if (!index)
    return std::nullopt;
  return moi_cfg_forward_distance(*index, from_text_offset);
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_sampled_barrier_sync_cave_words(
    std::span<const uint8_t> bytes, const SampledBarrierCandidate &candidate,
    const MoiSampledSyncEmissionPlan &plan, const VgprSpillSequence *spill,
    const SgprSpillSequence *scalar_spill, rj_code_arch_t arch,
    const ConSanMoiReportBufferLayout &layout, const ConSanMoiPrivateStateLayout *private_layout,
    std::optional<uint16_t> selectable_vgpr_bank_mode, std::vector<std::string> &errors,
    uint32_t *guest_instruction_offset = nullptr) {
  const bool scalar_persistent = plan.persistent_sgprs.complete();
  if (!plan.owner_epoch_vgprs.epoch || !plan.exec_save_sgpr || !plan.owner_epoch_vgprs.owner ||
      !plan.workitem_owner || candidate.owner_descriptor_file_offset == 0u ||
      static_cast<uint32_t>(plan.scratch_vgpr) + (scalar_persistent ? 9u : 7u) > kMaxVgprs ||
      candidate.windows.empty() || candidate.sequence == nullptr || candidate.site == nullptr ||
      std::ranges::any_of(candidate.windows, [&](const SampledBarrierWindow &window) {
        return window.first_slot >= layout.sampled_sync_metadata_capacity ||
               window.bank_count == 0 ||
               window.bank_count > layout.sampled_sync_metadata_capacity - window.first_slot;
      }))
    return std::nullopt;
  if (candidate.site->file_offset > bytes.size() ||
      candidate.site->size > bytes.size() - candidate.site->file_offset) {
    errors.emplace_back("ConSan MOI sampled barrier sequence exceeds ELF bytes");
    return std::nullopt;
  }
  const auto encoded = encode_consan_moi_sampled_sync_metadata({
      .kind = ConSanMoiSampledSyncKind::Barrier,
      .role = ConSanMoiSampledSyncRole::AcquireRelease,
      .scope = candidate.sequence->barrier_scope == ConSanBarrierSite::Scope::Cluster
                   ? ConSanMoiSampledSyncScope::Cluster
                   : ConSanMoiSampledSyncScope::Workgroup,
      .outcome = ConSanMoiSampledSyncOutcome::NotApplicable,
      .epoch_before = 0,
      .epoch_after = 1,
  });
  if (encoded.classification != ConSanMoiSampledSyncClassification::Valid)
    return std::nullopt;

  const uint16_t base = plan.scratch_vgpr;
  const uint16_t value = static_cast<uint16_t>(base + 2u);
  const uint16_t expected = static_cast<uint16_t>(base + 3u);
  const uint16_t next_epoch = static_cast<uint16_t>(base + 4u);
  const uint16_t lane_rank = static_cast<uint16_t>(base + 5u);
  const uint16_t bank_vgpr = static_cast<uint16_t>(base + 6u);
  const uint16_t temporary_exec = *plan.exec_save_sgpr;
  const uint16_t original_exec = static_cast<uint16_t>(*plan.exec_save_sgpr + 6u);
  const uint64_t report_base = plan.report_buffer_address;
  const uint64_t first_window_address = report_base + layout.sampled_causal_windows_offset;
  const uint64_t first_metadata_address = report_base + layout.sampled_sync_metadata_offset;

  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  MoiEmissionRequirement require_emission(sequence, errors);
  ConSanMoiRecordEmitter record(words, base, value, arch);
  const auto existing_label = sequence.make_label();
  const auto collision_label = sequence.make_label();
  const auto restore_label = sequence.make_label();
  const auto fast_return_label = sequence.make_label();
  const auto matched_window_label = sequence.make_label();
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return std::nullopt;
  const bool control_selectable_vgpr_bank = target->has_selectable_vgpr_bank;
  const bool restore_guest_vgpr_bank =
      control_selectable_vgpr_bank && selectable_vgpr_bank_mode.value_or(0u) != 0u;
  // A patched s_call_i64 can be immediately followed by the guest's next
  // s_set_vgpr_msb.  That program-state update may take effect as control
  // transfers, even when the replaced barrier itself ran in mode zero.  Make
  // every CDNA5 cave establish its low-register invariant explicitly before
  // touching scratch VGPRs.
  if (control_selectable_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(selectable_vgpr_bank_mode.value_or(0u)), 0u, arch));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->save_words.begin(), scalar_spill->save_words.end());
  if (restore_guest_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*selectable_vgpr_bank_mode), arch));
  if (guest_instruction_offset)
    *guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  for (uint64_t offset = 0; offset < candidate.site->size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site->file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (restore_guest_vgpr_bank)
    words.push_back(*instrumentation::build_s_set_vgpr_msb_transition(
        static_cast<uint8_t>(*selectable_vgpr_bank_mode), 0u, arch));
  // Runtime sampling selects LDS addresses, not workgroups. Every workgroup
  // which reaches the guest barrier must therefore inspect its banked causal
  // window and advance its epoch. A separate workgroup-residue gate can skip a
  // workgroup which touched a selected address, leaving its recorded accesses
  // in one epoch with no synchronization metadata.
  if (private_layout &&
      !append_sampled_private_owner_epoch_load(words, bytes, candidate.owner_descriptor_file_offset,
                                               plan.automatic_private_epoch, plan.owner_epoch_vgprs,
                                               *private_layout, arch, errors))
    return std::nullopt;
  if (scalar_persistent) {
    words.push_back(
        build_v_mov_b32_e32(*plan.owner_epoch_vgprs.owner, *plan.persistent_sgprs.owner(), arch));
    words.push_back(
        build_v_mov_b32_e32(*plan.owner_epoch_vgprs.epoch, *plan.persistent_sgprs.epoch(), arch));
  }
  require_emission(append_save_moi_special_state(words, plan.scalar_abi.special_state, arch),
                   "ConSan MOI sampled barrier failed at special-state save");
  require_emission(
      sequence.emit(instrumentation::build_s_mov_b64(original_exec, kAmdGpuExecLo, arch)),
      "ConSan MOI sampled barrier failed at EXEC snapshot");
  for (const SampledBarrierWindow &window : candidate.windows) {
    const auto next_window_label = sequence.make_label();
    require_emission(
        sequence.emit(instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch)),
        "ConSan MOI sampled barrier failed at window-scan EXEC restoration");
    require_emission(append_sampled_window_bank_index(
                         words, plan.dispatch_id, plan.workgroup_sources, window.bank_count,
                         bank_vgpr, value, *plan.owner_epoch_vgprs.owner, arch),
                     "ConSan MOI sampled barrier failed at causal-window bank selection");
    if (window.first_slot != 0u) {
      // VALUE is dead after bank selection and is distinct from BANK. Use it
      // as CDNA4's explicit literal temporary while keeping the result in BANK.
      require_emission(sequence.emit(instrumentation::build_v_add_u32_literal(
                           bank_vgpr, value, window.first_slot, bank_vgpr, arch)),
                       "ConSan MOI sampled barrier failed at absolute slot materialization");
    }
    require_emission(consan_detail::append_moi_indexed_address(
                         words,
                         {.table_address = first_window_address,
                          .stride_bytes = sizeof(ConSanMoiSampledCausalWindow),
                          .address_vgpr = base,
                          .index_vgpr = bank_vgpr},
                         *target),
                     "ConSan MOI sampled barrier failed at causal-window addressing");

    require_emission(consan_moi_impl::append_sampled_causal_window_validation(
                         words, sequence,
                         {.generation = plan.report_generation,
                          .dispatch_id = plan.dispatch_id,
                          .workgroup_sources = plan.workgroup_sources,
                          .address_vgpr = base,
                          .value_vgpr = value,
                          .expected_vgpr = expected,
                          .epoch_vgpr = *plan.owner_epoch_vgprs.epoch,
                          .first_entry_vgpr = bank_vgpr,
                          .temporary_exec_sgpr = temporary_exec,
                          .mismatch_label = next_window_label,
                          .arch = arch}),
                     "ConSan MOI sampled barrier failed at causal-window validation");
    sequence.branch(matched_window_label, InstructionSequence::BranchKind::Unconditional)
        .bind_label(next_window_label);
  }
  sequence.branch(restore_label, InstructionSequence::BranchKind::Unconditional)
      .bind_label(matched_window_label);

  const auto narrow_current = [&]() {
    sequence.append(instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch))
        .branch(restore_label, InstructionSequence::BranchKind::SccZero);
  };
  // A workgroup barrier resumes every wave. Elect the first logical owner
  // before the wave-local lane election below so exactly one wave can publish
  // the workgroup-wide transition. Without this narrowing, every wave races
  // the same metadata slot and the losing waves look like unsupported sync.
  require_emission(sequence.emit(instrumentation::build_v_cmp_eq_u32_vcc(
                       scalar_positive_inline_u32(0), *plan.owner_epoch_vgprs.owner, arch)),
                   "ConSan MOI sampled barrier failed at owner election");
  const bool began_owner_narrowing = static_cast<bool>(sequence);
  narrow_current();
  if (began_owner_narrowing && !sequence)
    errors.emplace_back("ConSan MOI sampled barrier failed at owner narrowing");
  const auto max_epoch = instrumentation::build_v_mov_b32_literal(
      expected, consan_moi_sampled_watchpoint::max_epoch, arch);
  const auto epoch_has_successor = instrumentation::build_v_cmp_ne_u32_vcc(
      vector_source_vgpr(expected), *plan.owner_epoch_vgprs.epoch, arch);
  require_emission(sequence.emit_all(max_epoch, epoch_has_successor),
                   "ConSan MOI sampled barrier failed at epoch successor check");
  const bool began_epoch_narrowing = static_cast<bool>(sequence);
  narrow_current();
  if (began_epoch_narrowing && !sequence)
    errors.emplace_back("ConSan MOI sampled barrier failed at epoch narrowing");

  require_emission(
      sequence.emit(instrumentation::build_s_mov_b64(temporary_exec, kAmdGpuExecLo, arch)) &&
          append_select_first_lane_in_exec_mask(words, lane_rank, temporary_exec, temporary_exec,
                                                arch),
      "ConSan MOI sampled barrier failed at lane election");

  require_emission(
      consan_detail::append_moi_indexed_address(
          words,
          {.table_address =
               first_metadata_address + offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
           .stride_bytes = sizeof(ConSanMoiSampledSyncMetadataPacked),
           .address_vgpr = base,
           .index_vgpr = bank_vgpr},
          *target) &&
          sequence.emit_all(instrumentation::build_v_mov_b32_literal(
                                value, kConSanMoiSampledSyncPublishingDescriptor, arch),
                            instrumentation::build_v_mov_b32_literal(expected, 0u, arch),
                            instrumentation::build_flat_atomic_cmpswap_b32(
                                base, value, value, true, kAmdGpuScopeDevice, arch)),
      "ConSan MOI sampled barrier failed at metadata claim setup");
  sequence.require(append_moi_global_atomic_wait(words, arch));
  require_emission(
      sequence.emit_all(
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch),
          instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch)),
      "ConSan MOI sampled barrier failed at metadata claim result");
  const bool began_existing_branch = static_cast<bool>(sequence);
  sequence.branch(existing_label, InstructionSequence::BranchKind::SccZero);
  if (began_existing_branch && !sequence)
    errors.emplace_back("ConSan MOI sampled barrier failed at existing metadata branch");

  const auto increment = instrumentation::build_v_add_u32(next_epoch, scalar_positive_inline_u32(1),
                                                          *plan.owner_epoch_vgprs.epoch, arch);
  require_emission(sequence.emit(increment),
                   "ConSan MOI sampled barrier failed at next-epoch materialization");
  require_emission(consan_detail::append_moi_indexed_address(
                       words,
                       {.table_address = first_metadata_address,
                        .stride_bytes = sizeof(ConSanMoiSampledSyncMetadataPacked),
                        .address_vgpr = base,
                        .index_vgpr = bank_vgpr},
                       *target),
                   "ConSan MOI sampled barrier failed at metadata addressing");
  require_emission(
      record.store_literal(0u, 0u) && record.store_literal(4u, 0u) &&
          record.store_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 0u) &&
          record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                            *plan.owner_epoch_vgprs.epoch) &&
          record.store_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after), next_epoch),
      "ConSan MOI sampled barrier failed at metadata body stores");
  require_emission(
      sequence.emit_all(
          instrumentation::build_s_wait_global_store0(arch),
          instrumentation::build_v_mov_b32_literal(value, encoded.packed.descriptor, arch),
          instrumentation::build_v_mov_b32_literal(
              expected, kConSanMoiSampledSyncPublishingDescriptor, arch)),
      "ConSan MOI sampled barrier failed at metadata commit setup");
  require_emission(consan_detail::append_moi_indexed_address(
                       words,
                       {.table_address = first_metadata_address +
                                         offsetof(ConSanMoiSampledSyncMetadataPacked, descriptor),
                        .stride_bytes = sizeof(ConSanMoiSampledSyncMetadataPacked),
                        .address_vgpr = base,
                        .index_vgpr = bank_vgpr},
                       *target),
                   "ConSan MOI sampled barrier failed at metadata commit addressing");
  require_emission(sequence.emit(instrumentation::build_flat_atomic_cmpswap_b32(
                       base, value, value, true, kAmdGpuScopeDevice, arch)),
                   "ConSan MOI sampled barrier failed at metadata commit setup");
  sequence.require(append_moi_global_atomic_wait(words, arch));
  require_emission(
      sequence.emit_all(
          instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch),
          instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch)),
      "ConSan MOI sampled barrier failed at metadata commit result");
  sequence.branch(collision_label, InstructionSequence::BranchKind::SccZero);
  require_emission(append_atomic_fetch_add_one_u32(
                       words,
                       report_base + offsetof(ConSanMoiReportHeader, sampled_sync_metadata_count),
                       value, base, arch),
                   "ConSan MOI sampled barrier failed at success accounting");
  sequence.branch(restore_label, InstructionSequence::BranchKind::Unconditional);

  // The host model admits an exact pre-existing metadata record as an
  // idempotent publication. Shared barrier helpers can expose several
  // equivalent physical sequences for one logical transition, so the device
  // path must implement the same contract instead of treating every failed
  // empty-slot claim as lost coverage. A different or partially published
  // record remains a collision.
  const bool began_existing_bind = static_cast<bool>(sequence);
  sequence.bind_label(existing_label);
  if (began_existing_bind && !sequence)
    errors.emplace_back("ConSan MOI sampled barrier failed to bind existing metadata");
  require_emission(
      sequence.emit_all(
          instrumentation::build_s_mov_b64(kAmdGpuExecLo, temporary_exec, arch),
          instrumentation::build_v_mov_b32_literal(expected, encoded.packed.descriptor, arch),
          instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch)),
      "ConSan MOI sampled barrier failed at existing metadata descriptor");
  sequence.branch(collision_label, InstructionSequence::BranchKind::VccZero).append(increment);
  require_emission(consan_detail::append_moi_indexed_address(
                       words,
                       {.table_address = first_metadata_address,
                        .stride_bytes = sizeof(ConSanMoiSampledSyncMetadataPacked),
                        .address_vgpr = base,
                        .index_vgpr = bank_vgpr},
                       *target),
                   "ConSan MOI sampled barrier failed at existing metadata addressing");
  const auto require_existing_vgpr = [&](uint32_t offset, uint16_t expected_vgpr) {
    sequence.require(record.load(offset, value))
        .append(
            instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr), value, arch))
        .branch(collision_label, InstructionSequence::BranchKind::VccZero);
  };
  const auto require_existing_literal = [&](uint32_t offset, uint32_t literal) {
    sequence.append(instrumentation::build_v_mov_b32_literal(expected, literal, arch));
    require_existing_vgpr(offset, expected);
  };
  const bool began_existing_comparison = static_cast<bool>(sequence);
  require_existing_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, address), 0u);
  require_existing_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, address) + 4u, 0u);
  require_existing_literal(offsetof(ConSanMoiSampledSyncMetadataPacked, byte_count), 0u);
  require_existing_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_before),
                        *plan.owner_epoch_vgprs.epoch);
  require_existing_vgpr(offsetof(ConSanMoiSampledSyncMetadataPacked, epoch_after), next_epoch);
  sequence.branch(restore_label, InstructionSequence::BranchKind::Unconditional);
  if (began_existing_comparison && !sequence)
    errors.emplace_back("ConSan MOI sampled barrier failed at existing metadata comparison");

  sequence.bind_label(collision_label);
  require_emission(
      sequence.emit_all(instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch),
                        instrumentation::build_s_mov_b64(temporary_exec, original_exec, arch)) &&
          append_select_first_lane_in_exec_mask(words, lane_rank, temporary_exec, temporary_exec,
                                                arch),
      "ConSan MOI sampled barrier failed at collision lane election");
  require_emission(
      append_atomic_fetch_add_one_u32(
          words, report_base + offsetof(ConSanMoiReportHeader, sampled_unsupported_sync_count),
          value, base, arch),
      "ConSan MOI sampled barrier failed at collision accounting");

  sequence.bind_label(restore_label);
  require_emission(
      sequence.emit_all(
          instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch), max_epoch,
          epoch_has_successor,
          instrumentation::build_s_and_saveexec_b64(temporary_exec, kAmdGpuVccLo, arch),
          instrumentation::build_v_add_u32(*plan.owner_epoch_vgprs.epoch,
                                           scalar_positive_inline_u32(1),
                                           *plan.owner_epoch_vgprs.epoch, arch),
          instrumentation::build_s_mov_b64(kAmdGpuExecLo, original_exec, arch)),
      "ConSan MOI sampled barrier failed at epoch-advance narrowing");
  if (scalar_persistent) {
    require_emission(
        sequence.emit_all(instrumentation::build_v_readfirstlane_b32(
                              *plan.persistent_sgprs.epoch(), *plan.owner_epoch_vgprs.epoch, arch),
                          instrumentation::build_salu_dependency_delay(arch)),
        "ConSan MOI sampled barrier failed at scalar epoch persistence");
  }
  if (private_layout) {
    require_emission(
        sequence.emit_all(instrumentation::build_private_store_b32(
                              *plan.owner_epoch_vgprs.epoch, private_layout->epoch_offset, arch),
                          instrumentation::build_s_wait_private_store0(arch)),
        "ConSan MOI sampled barrier failed at private epoch store");
  }
  require_emission(append_restore_moi_special_state(words, plan.scalar_abi.special_state, arch),
                   "ConSan MOI sampled barrier failed at special-state restore");
  sequence.bind_label(fast_return_label);
  if (scalar_spill)
    words.insert(words.end(), scalar_spill->restore_words.begin(),
                 scalar_spill->restore_words.end());
  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (restore_guest_vgpr_bank)
    sequence.append(instrumentation::build_s_set_vgpr_msb_transition(
        0u, static_cast<uint8_t>(*selectable_vgpr_bank_mode), arch));
  if (!sequence.finish(arch)) {
    errors.emplace_back("ConSan MOI sampled barrier local branch is out of range");
    return std::nullopt;
  }
  return words;
}

void try_apply_sampled_barrier_sync_patch(
    std::span<const uint8_t> bytes, const ConSanOptions &options,
    const ConSanMoiOperatingPoint &operating_point, rj_code_arch_t arch,
    MoiResourcePlanningState &resource_state, std::span<const ConSanMoiCandidate> admitted,
    const MoiObjectModeSemantics &mode_semantics, ConSanTransformArtifacts &result) {
  if (!options.moi_track_barriers)
    return;
  const ConSanTargetProfile *target_profile = consan_target_profile(arch);
  if (!options.moi_report_buffer_address || target_profile == nullptr) {
    result.warnings.emplace_back(
        "ConSan MOI sampled barrier sync requires a supported report-buffer target");
    return;
  }
  const ConSanMoiReportBufferLayout &layout = mode_semantics.report_layout;
  const ConSanRuntimeStaticMapping runtime_mapping =
      sampled_runtime_mapping_including_staged(result);
  const ConSanRuntimeStaticMapping::Sampled *sampled_accesses = runtime_mapping.sampled();
  std::vector<const ConSanSampledStaticAccessMapping *> selected_windows;
  selected_windows.reserve(sampled_accesses == nullptr ? 0u : sampled_accesses->size());
  std::unordered_set<uint64_t> admitted_offsets;
  admitted_offsets.reserve(admitted.size());
  for (const ConSanMoiCandidate &candidate : admitted) {
    if (candidate.kind == ConSanLdsAccessKind::Read || candidate.kind == ConSanLdsAccessKind::Write)
      admitted_offsets.insert(candidate.anchor());
  }
  bool has_multi_range_window = false;
  if (sampled_accesses != nullptr) {
    for (const ConSanSampledStaticAccessMapping &mapping : *sampled_accesses) {
      if (!admitted_offsets.contains(mapping.access.original_site.original_text_offset) ||
          mapping.range_count == 0u || mapping.bank_count == 0u)
        continue;
      selected_windows.push_back(&mapping);
      has_multi_range_window |= mapping.range_count > 1u;
    }
  }
  if (selected_windows.empty() || layout.sampled_sync_metadata_capacity == 0) {
    result.warnings.emplace_back(
        "ConSan MOI sampled barrier sync found no selected Ready causal-window capacity");
    return;
  }
  if (has_multi_range_window) {
    result.warnings.emplace_back(
        "ConSan MOI sampled barrier sync currently attaches only the first logical range of a "
        "multi-range access");
  }
  if (!moi_resource_is_valid(resource_state)) {
    result.warnings.emplace_back("ConSan MOI sampled barrier sync lacks a complete kernel CFG");
    return;
  }
  std::unordered_map<uint64_t, std::vector<const ConSanSampledStaticAccessMapping *>>
      windows_by_descriptor;
  for (const ConSanSampledStaticAccessMapping *window : selected_windows) {
    for (ConSanProgramContainerId owner : window->access.execution_owner_kernel_ids) {
      const ConSanProgramContainer *kernel = result.program_inventory.container(owner);
      if (kernel != nullptr && kernel->is_kernel())
        windows_by_descriptor[kernel->descriptor_file_offset].push_back(window);
    }
  }

  const std::vector<MoiBarrierEvidenceSitePlan> barrier_plans =
      build_moi_barrier_evidence_site_plans(result.program_inventory, result.observation_plan(),
                                            ConSanProbeIntentKind::SampledBarrierEpoch,
                                            result.errors);
  if (!result.errors.empty())
    return;
  std::vector<SampledBarrierCandidate> candidates;
  for (const MoiBarrierEvidenceSitePlan &barrier_plan : barrier_plans) {
    const ConSanSyncSequence *sequence_ptr =
        result.program_inventory.sync().find_sequence(barrier_plan.sequence);
    const std::vector<ConSanExecutionOwner> sequence_owners =
        sequence_ptr != nullptr ? result.program_inventory.sync().execution_owners(*sequence_ptr)
                                : std::vector<ConSanExecutionOwner>{};
    if (sequence_ptr == nullptr ||
        !consan_moi_sampled_qualifies_barrier_sequence(*sequence_ptr, !sequence_owners.empty())) {
      result.errors.emplace_back("ConSan MOI sampled barrier evidence lost its qualified sequence");
      return;
    }
    const ConSanSyncSequence &sequence = *sequence_ptr;
    const std::string_view sequence_container_name =
        result.program_inventory.sync().container_name(sequence);
    std::vector<SampledBarrierWindow> windows;
    bool every_owner_has_window = true;
    std::string owner_lowering_failure;
    bool uses_cluster_workgroup_id = false;
    bool first_owner = true;
    std::optional<ConSanMoiWorkgroupSources> common_workgroup_sources;
    std::optional<uint16_t> common_owner_shift;
    for (const ConSanExecutionOwner &owner : sequence_owners) {
      const ConSanProgramContainer *owner_kernel = result.program_inventory.kernel(owner);
      if (owner_kernel == nullptr ||
          !moi_resource_has_owner_context(resource_state, owner_kernel->descriptor_file_offset)) {
        owner_lowering_failure = "missing kernel resource context";
        break;
      }
      const uint64_t owner_descriptor_file_offset = owner_kernel->descriptor_file_offset;
      if (!first_owner && uses_cluster_workgroup_id != owner_kernel->uses_cluster_workgroup_id) {
        owner_lowering_failure = "incompatible cluster workgroup ABI";
        break;
      }
      uses_cluster_workgroup_id = owner_kernel->uses_cluster_workgroup_id;
      first_owner = false;
      const auto workgroup_sources = moi_descriptor_workgroup_sources(
          bytes, owner_descriptor_file_offset, arch, result.errors, uses_cluster_workgroup_id);
      ConSanMoiOperatingPoint owner_point = operating_point;
      const std::array<uint64_t, 1> owner_descriptor = {owner_descriptor_file_offset};
      if (!apply_moi_persistent_vgpr_assignment(owner_point, result.moi_operating_point,
                                                owner_descriptor)) {
        owner_lowering_failure = "missing owner-local persistent assignment";
        break;
      }
      const auto owner_shift =
          owner_point.automatic_moi_private_epoch
              ? moi_descriptor_owner_shift(bytes, owner_descriptor_file_offset, arch, result.errors)
              : std::optional<uint16_t>(0u);
      if (!workgroup_sources || !owner_shift ||
          (common_workgroup_sources && *common_workgroup_sources != *workgroup_sources) ||
          (common_owner_shift && *common_owner_shift != *owner_shift)) {
        owner_lowering_failure = "incompatible descriptor ABI inputs";
        break;
      }
      common_workgroup_sources = *workgroup_sources;
      common_owner_shift = *owner_shift;
      const auto distance_index = make_moi_cfg_forward_distance_index(
          resource_state, owner_descriptor_file_offset, sequence.begin_text_offset);
      if (!distance_index) {
        owner_lowering_failure = "missing owner CFG distance proof";
        break;
      }
      const auto owner_windows = windows_by_descriptor.find(owner_descriptor_file_offset);
      if (owner_windows == windows_by_descriptor.end()) {
        every_owner_has_window = false;
        break;
      }
      std::optional<SampledBarrierWindow> selected_window;
      std::optional<SampledCfgDistance> closest;
      for (const ConSanSampledStaticAccessMapping *window : owner_windows->second) {
        const auto distance = moi_cfg_forward_distance(
            *distance_index, window->access.original_site.original_text_offset);
        if (!distance || (closest && *closest <= *distance))
          continue;
        closest = *distance;
        selected_window = SampledBarrierWindow{window->first_slot, window->bank_count};
      }
      if (!selected_window) {
        every_owner_has_window = false;
        break;
      }
      windows.push_back(*selected_window);
    }
    std::ranges::sort(windows);
    windows.erase(std::ranges::unique(windows).begin(), windows.end());
    if (!owner_lowering_failure.empty()) {
      // The sequence is semantically applicable and must remain in the strict
      // coverage denominator. Leaving its Supported disposition unpatched
      // records a durable placement/lowering failure instead of hiding a
      // descriptor or CFG incompatibility as a missing causal window.
      result.warnings.emplace_back("ConSan MOI sampled barrier sync cannot lower every owner in " +
                                   std::string(sequence_container_name) + ": " +
                                   owner_lowering_failure);
      continue;
    }
    if (!every_owner_has_window || windows.empty()) {
      result.warnings.emplace_back(
          "ConSan MOI sampled barrier sync has no preceding selected causal window for every "
          "owner in " +
          std::string(sequence_container_name));
      continue;
    }
    if (sequence_owners.empty()) {
      result.errors.emplace_back("ConSan MOI sampled barrier evidence has no execution owner");
      return;
    }
    const uint64_t owner_descriptor =
        result.program_inventory.kernel(sequence_owners.front())->descriptor_file_offset;
    const ConSanBarrierSite *barrier_site =
        result.program_inventory.program_site<ConSanBarrierSite>(barrier_plan.source_site);
    if (barrier_site == nullptr) {
      result.errors.emplace_back("ConSan MOI sampled barrier lost its decoded source");
      return;
    }
    candidates.push_back({&sequence, barrier_plan, barrier_site, owner_descriptor,
                          std::move(windows), uses_cluster_workgroup_id});
  }
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled barrier sync found no qualified complete barrier sequence");
    return;
  }
  std::ranges::sort(candidates, {}, [](const SampledBarrierCandidate &candidate) {
    return candidate.site->text_offset;
  });
  if (candidates.size() > options.max_patches)
    candidates.resize(options.max_patches);

  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI sampled barrier sync found no .text section");
    return;
  }

  struct Planned : MoiPlannedProbeResources {
    SampledBarrierCandidate candidate;
    ConSanMoiOperatingPoint point;
  };
  std::vector<Planned> selected;
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  std::vector<std::pair<uint64_t, uint64_t>> patch_claims;
  patch_claims.reserve(result.patches.size());
  for (const ConSanPatchLoweringProduct &patch : result.patches) {
    const uint64_t size = std::max<uint32_t>(patch.original_size, 1u);
    patch_claims.emplace_back(patch.anchor_offset, patch.anchor_offset + size);
  }
  std::ranges::sort(patch_claims);
  size_t merged_patch_claim_count = 0u;
  for (const auto &claim : patch_claims) {
    if (merged_patch_claim_count != 0u &&
        patch_claims[merged_patch_claim_count - 1u].second >= claim.first) {
      patch_claims[merged_patch_claim_count - 1u].second =
          std::max(patch_claims[merged_patch_claim_count - 1u].second, claim.second);
    } else {
      patch_claims[merged_patch_claim_count++] = claim;
    }
  }
  patch_claims.resize(merged_patch_claim_count);
  const auto overlaps_patch = [&](uint64_t begin, uint64_t end) {
    const auto after = std::ranges::lower_bound(patch_claims, end, {},
                                                [](const auto &range) { return range.first; });
    return after != patch_claims.begin() && std::prev(after)->second > begin;
  };
  std::unordered_map<uint64_t, size_t> barrier_plan_by_offset;
  barrier_plan_by_offset.reserve(result.resource_plans.size());
  for (size_t index = 0; index < result.resource_plans.size(); ++index) {
    const ConSanCandidateResourcePlan &plan = result.resource_plans[index];
    if (plan.site_kind == ConSanResourceSiteKind::Barrier)
      barrier_plan_by_offset.try_emplace(plan.text_offset, index);
  }
  for (size_t index = 0; index < candidates.size(); ++index) {
    SampledBarrierCandidate candidate = candidates[index];
    const ConSanBarrierSite &site = *candidate.site;
    if (site.size == 0 || site.size % sizeof(uint32_t) != 0 || site.text_offset > old_text.size() ||
        site.size > old_text.size() - site.text_offset)
      continue;
    if (overlaps_patch(site.text_offset, site.text_offset + site.size))
      continue;
    const auto existing = barrier_plan_by_offset.find(site.text_offset);
    ConSanCandidateResourcePlan plan;
    size_t resource_plan_index = 0u;
    if (existing != barrier_plan_by_offset.end()) {
      resource_plan_index = existing->second;
      plan = result.resource_plans[resource_plan_index];
    } else {
      plan = plan_moi_resource_site(resource_state, options, options, operating_point,
                                    ConSanResourceSiteKind::Barrier, index, site.text_offset,
                                    candidate.barrier.source_site,
                                    operating_point.automatic_moi_private_epoch ? 9u : 7u);
      plan.intent_ids = {candidate.barrier.evidence_intent};
      result.resource_plans.push_back(plan);
      resource_plan_index = result.resource_plans.size() - 1u;
      barrier_plan_by_offset.emplace(site.text_offset, resource_plan_index);
    }
    ConSanMoiOperatingPoint candidate_point = operating_point;
    if (!apply_moi_transient_sgpr_assignment(options, candidate_point, result.moi_operating_point,
                                             plan.owner_descriptor_file_offsets) ||
        !apply_moi_persistent_vgpr_assignment(candidate_point, result.moi_operating_point,
                                              plan.owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI sampled barrier lost its owner-local persistent or transient assignment");
      continue;
    }
    const uint16_t scratch_count = candidate_point.automatic_moi_private_epoch ||
                                           candidate_point.moi_persistent_sgprs.complete()
                                       ? 9u
                                       : 7u;
    if (plan.scratch_vgpr_count != scratch_count) {
      plan = plan_moi_resource_site(resource_state, options, options, candidate_point,
                                    ConSanResourceSiteKind::Barrier, index, site.text_offset,
                                    candidate.barrier.source_site, scratch_count);
      plan.intent_ids = {candidate.barrier.evidence_intent};
      result.resource_plans[resource_plan_index] = plan;
    }
    const auto resources =
        resolve_moi_scratch_plan(plan, candidate_point, result.moi_operating_point, scratch_count);
    if (!resources)
      continue;
    std::optional<ConSanMoiPrivateStateLayout> private_layout;
    if (candidate_point.automatic_moi_private_epoch) {
      private_layout = build_moi_private_state_layout(
          result.program_inventory, *resources, arch, result.warnings,
          {.owner = options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
           .exact_workgroup = true});
      if (!private_layout)
        continue;
    }
    auto probe = plan_moi_probe_resources(
        result.program_inventory, *resources, options, options, candidate_point, mode_semantics,
        spill_managers, arch, std::move(private_layout),
        candidate_point.has_compact_moi_scalar_spill(), result.warnings);
    if (!probe)
      continue;
    note_moi_probe_private_requirements(private_requirements, *probe);
    note_descriptor_requirements(descriptor_requirements, probe->resources);
    selected.push_back({std::move(*probe), std::move(candidate), std::move(candidate_point)});
  }
  if (selected.empty()) {
    result.warnings.emplace_back("ConSan MOI sampled barrier sync found no patchable sequence");
    return;
  }
  MoiDescriptorSgprRequirements scalar_requirements;
  std::vector<ConSanTextFragment> staged_fragments;
  staged_fragments.reserve(selected.size());
  for (const Planned &planned : selected) {
    note_moi_sgpr_requirements(scalar_requirements, planned.resources, options, options,
                               planned.point, mode_semantics, arch);
    const ConSanBarrierSite &site = *planned.candidate.site;
    const ConSanProgramContainer *container = resolve_moi_evidence_container(
        result.program_inventory, planned.candidate.barrier.source_site);
    if (container == nullptr) {
      result.errors.emplace_back("ConSan MOI sampled barrier lost its source container");
      return;
    }
    const auto emission_plan = plan_sampled_sync_emission(
        options, options, planned.point, result.moi_operating_point,
        planned.resources.owner_descriptor_file_offsets,
        planned.private_layout ? &*planned.private_layout : nullptr, planned.resources.base,
        static_cast<uint16_t>(planned.resources.base + 7u),
        static_cast<uint16_t>(planned.resources.base + 8u),
        /*transient_assignment_already_resolved=*/true, "sampled barrier sync", result.errors);
    if (!emission_plan)
      return;
    const std::optional<uint16_t> selectable_vgpr_bank_mode =
        consan_selectable_vgpr_bank_mode_at(arch, active_bytes, container->text_file_offset,
                                            container->entry_text_offset, site.file_offset);
    uint32_t guest_instruction_offset = 0u;
    auto words = build_sampled_barrier_sync_cave_words(
        bytes, planned.candidate, *emission_plan, planned.spill ? &*planned.spill : nullptr,
        planned.scalar_spill ? &*planned.scalar_spill : nullptr, arch, layout,
        planned.private_layout ? &*planned.private_layout : nullptr, selectable_vgpr_bank_mode,
        result.errors, &guest_instruction_offset);
    if (!words)
      return;
    ConSanPatchInfo patch;
    patch.kind = ConSanPatchKind::TrampolineMoiSampledSyncMetadata;
    patch.anchor_offset = site.text_offset;
    patch.original_size = site.size;
    note_moi_probe_patch_info(patch, planned);
    auto fragment = make_consan_around_text_fragment(
        std::move(*words), guest_instruction_offset, site.size,
        planned.candidate.barrier.intent_ids(), {}, std::move(patch), result.errors,
        "MOI sampled barrier sync");
    if (!fragment)
      return;
    staged_fragments.push_back(std::move(*fragment));
  }
  const MoiDescriptorLdsRequirements no_lds;
  (void)stage_consan_text_rewrites(
      code_object, arch,
      {descriptor_requirements, scalar_requirements, private_requirements, no_lds},
      moi_descriptor_mutation_policy(nullptr, arch), "MOI sampled barrier sync",
      std::move(staged_fragments), result);
  return;
}

void try_apply_sampled_atomic_sync_patch(std::span<const uint8_t> bytes,
                                         const ConSanOptions &options,
                                         const ConSanMoiOperatingPoint &operating_point,
                                         rj_code_arch_t arch,
                                         MoiResourcePlanningState &resource_state,
                                         const MoiObjectModeSemantics &mode_semantics,
                                         ConSanTransformArtifacts &result) {
  if (!options.moi_track_atomics)
    return;
  const ConSanTargetProfile *target_profile = consan_target_profile(arch);
  if (!options.moi_report_buffer_address || target_profile == nullptr) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync requires a supported report-buffer target");
    return;
  }
  const ConSanMoiReportBufferLayout &layout = mode_semantics.report_layout;
  ConSanRuntimeStaticMapping runtime_mapping = sampled_runtime_mapping_including_staged(result);
  ConSanRuntimeStaticMapping::Sampled *sampled_accesses = runtime_mapping.sampled();
  const uint32_t selected_window_count =
      sampled_accesses == nullptr ? 0u : static_cast<uint32_t>(sampled_accesses->size());
  if (selected_window_count == 0 || options.max_patches == 0 ||
      layout.sampled_sync_metadata_capacity == 0) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync found no selected Ready causal-window capacity");
    return;
  }

  struct Candidate {
    MoiAtomicEvidenceSitePlan atomic;
    ConSanAtomicSite site;
    consan_detail::SampledAtomicSemantics semantics;
    uint64_t ordered_sequence_end_text_offset = 0;
    uint32_t selected_slot = 0;
    uint32_t bank_count = 1;
    bool deferred_acquire = false;
    std::optional<uint32_t> release_selected_slot;
  };
  std::vector<Candidate> candidates;
  const SynchronizationInventoryView graph = result.program_inventory.sync();
  for (const ConSanAtomicSiteDecision &decision : result.observation_plan().atomic_site_decisions) {
    if (decision.reason != ConSanAtomicPolicyReason::CompareExchangeOutcomeUnavailable)
      continue;
    for (const std::string &container_name :
         result.program_inventory.source_container_names(decision.semantic_site.physical)) {
      result.warnings.emplace_back("ConSan MOI sampled atomic sync skipped " + container_name +
                                   ": compare-exchange-dynamic-outcome-unavailable");
    }
  }
  std::vector<MoiAtomicEvidenceSitePlan> admitted = build_moi_atomic_evidence_site_plans(
      result.program_inventory, result.observation_plan(),
      ConSanProbeIntentKind::SampledAtomicOrdering, result.errors);
  if (!result.errors.empty())
    return;
  for (MoiAtomicEvidenceSitePlan &atomic : admitted) {
    const std::optional<ConSanAtomicSite> site = materialize_moi_communication_site(
        result.program_inventory, atomic.source_site, atomic.sequence);
    if (!site) {
      result.errors.emplace_back("ConSan MOI sampled atomic sync lost its decoded source site");
      return;
    }
    if (atomic.is_rmw) {
      const SampledAtomicSemanticsResult semantics =
          sampled_atomic_semantics_for_plan(graph, atomic, *site);
      if (!semantics.semantics) {
        result.errors.emplace_back(
            "ConSan MOI admitted sampled atomic evidence is invalid: " +
            std::string(sampled_atomic_semantics_reason_name(semantics.reason)));
        return;
      }
      const uint64_t sequence_end = atomic.ordered_sequence_end_text_offset;
      candidates.push_back({std::move(atomic), *site, *semantics.semantics, sequence_end, 0, 1,
                            false, std::nullopt});
      continue;
    }
    const auto role = sampled_atomic_role(atomic.event_kind, /*is_rmw=*/false);
    const auto scope = site->scope ? consan_moi_sampled_sync_scope(*site->scope) : std::nullopt;
    if (!role || !scope || site->width_bits == 0u || site->width_bits % 8u != 0u) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync skipped an unqualified ordinary atomic sequence in " +
          moi_evidence_container_name(result.program_inventory, atomic.source_site));
      continue;
    }
    consan_detail::SampledAtomicSemantics semantics{
        .role = *role,
        .scope = *scope,
        .outcome = ConSanMoiSampledSyncOutcome::NotApplicable,
        .byte_count = site->width_bits / 8u,
        .descriptor = 0u,
        .cas_failure_descriptor = std::nullopt,
    };
    const auto encoded = encode_consan_moi_sampled_sync_metadata({
        .address = 1u,
        .byte_count = semantics.byte_count,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = semantics.role,
        .scope = semantics.scope,
        .outcome = semantics.outcome,
    });
    if (encoded.classification != ConSanMoiSampledSyncClassification::Valid) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync rejected ordinary atomic metadata in " +
          moi_evidence_container_name(result.program_inventory, atomic.source_site));
      continue;
    }
    semantics.descriptor = encoded.packed.descriptor;
    const uint64_t sequence_end = atomic.ordered_sequence_end_text_offset;
    candidates.push_back(
        {std::move(atomic), *site, semantics, sequence_end, 0, 1, false, std::nullopt});
  }
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync found no qualified atomic sequence");
    return;
  }

  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan MOI sampled atomic sync found no .text section");
    return;
  }

  // Direct sampled access instrumentation may already have grown .text and
  // recorded the then-active descriptor offsets on prologue patches. A second
  // growth would otherwise leave those intermediate offsets orphaned. Keep
  // owner evidence canonical in the original inventory coordinate system.
  std::unordered_map<uint64_t, uint64_t> active_to_canonical_owner;
  for (const AmdGpuKernelInfo &active_kernel : code_object.kernels()) {
    const ConSanProgramContainer *canonical =
        result.program_inventory.find_kernel_by_name(active_kernel.name);
    if (canonical != nullptr)
      active_to_canonical_owner.emplace(active_kernel.descriptor_file_offset,
                                        canonical->descriptor_file_offset);
  }
  const auto canonicalize_owners = [&](std::vector<uint64_t> &owners) {
    for (uint64_t &owner : owners) {
      const auto canonical = active_to_canonical_owner.find(owner);
      if (canonical != active_to_canonical_owner.end())
        owner = canonical->second;
    }
  };
  for (ConSanCandidateResourcePlan &plan : result.resource_plans)
    canonicalize_owners(plan.owner_descriptor_file_offsets);
  for (ConSanPatchLoweringProduct &patch : result.patches)
    canonicalize_owners(patch.owner_descriptor_file_offsets);

  if (!moi_resource_is_valid(resource_state)) {
    result.warnings.emplace_back("ConSan MOI sampled atomic sync lacks a complete kernel CFG");
    return;
  }

  struct Window {
    uint32_t slot = 0;
    uint32_t bank_count = 1;
    const ConSanSampledStaticAccessMapping *mapping = nullptr;
    ConSanLdsAccessKind access_kind = ConSanLdsAccessKind::Other;
    uint64_t owner_descriptor_file_offset = 0;
  };
  std::vector<Window> windows;
  windows.reserve(selected_window_count);
  bool has_multi_range_window = false;
  if (sampled_accesses != nullptr) {
    for (const ConSanSampledStaticAccessMapping &mapping : *sampled_accesses) {
      if (mapping.access_kind != ConSanLdsAccessKind::Read &&
          mapping.access_kind != ConSanLdsAccessKind::Write &&
          mapping.access_kind != ConSanLdsAccessKind::Atomic)
        continue;
      has_multi_range_window |= mapping.range_count > 1u;
      for (const ConSanProgramContainer &kernel : result.program_inventory.kernels()) {
        if (!moi_resource_has_owner_context(resource_state, kernel.descriptor_file_offset))
          continue;
        const auto &owners = mapping.access.execution_owner_kernel_ids;
        const bool owns_mapping =
            owners.empty()
                ? kernel.has_text_range &&
                      mapping.access.original_site.original_text_offset >=
                          kernel.entry_text_offset &&
                      mapping.access.original_site.original_text_offset - kernel.entry_text_offset <
                          kernel.code_size
                : std::ranges::find(owners, kernel.id) != owners.end();
        if (!owns_mapping)
          continue;
        // One physical access in a shared helper is one sampled slot, but it has
        // an independent causal view in every kernel that can execute it. Those
        // owners are mutually exclusive dispatches, so each may safely attach
        // one synchronization edge to the same metadata slot.
        windows.push_back({mapping.first_slot, mapping.bank_count, &mapping, mapping.access_kind,
                           kernel.descriptor_file_offset});
      }
    }
  }
  if (has_multi_range_window) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync currently attaches only the first logical range of a "
        "multi-range access");
  }

  struct Association {
    enum class Side : uint8_t { Release, Acquire };

    size_t candidate_index = 0;
    size_t window_index = 0;
    SampledCfgDistance distance;
    Side side = Side::Release;
  };
  std::vector<Association> proposed;
  const auto owner_executes_candidate = [&](uint64_t owner_descriptor_file_offset,
                                            const Candidate &candidate) {
    const ConSanProgramContainer *owner =
        result.program_inventory.find_kernel_by_descriptor(owner_descriptor_file_offset);
    const ConSanProgramSite *site =
        result.program_inventory.program_site(candidate.atomic.source_site);
    return owner != nullptr && site != nullptr &&
           std::ranges::find(site->execution_owners, owner->id, &ConSanExecutionOwner::kernel) !=
               site->execution_owners.end();
  };
  for (size_t window_index = 0; window_index < windows.size(); ++window_index) {
    const Window &window = windows[window_index];
    std::optional<Association> best;
    bool ambiguous = false;
    for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
      const Candidate &candidate = candidates[candidate_index];
      if (!owner_executes_candidate(window.owner_descriptor_file_offset, candidate))
        continue;
      const uint32_t semantic_role = static_cast<uint32_t>(candidate.semantics.role);
      const bool release_side =
          (window.access_kind == ConSanLdsAccessKind::Write ||
           window.access_kind == ConSanLdsAccessKind::Atomic) &&
          (semantic_role & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Release)) != 0u;
      const bool acquire_side =
          (window.access_kind == ConSanLdsAccessKind::Read ||
           window.access_kind == ConSanLdsAccessKind::Atomic) &&
          (semantic_role & static_cast<uint32_t>(ConSanMoiSampledSyncRole::Acquire)) != 0u;
      std::optional<SampledCfgDistance> distance;
      Association::Side side = Association::Side::Release;
      if (release_side) {
        distance =
            sampled_cfg_forward_distance(resource_state, window.owner_descriptor_file_offset,
                                         window.mapping->access.original_site.original_text_offset,
                                         candidate.site.text_offset);
      } else if (acquire_side) {
        side = Association::Side::Acquire;
        distance = sampled_cfg_forward_distance(
            resource_state, window.owner_descriptor_file_offset, candidate.site.text_offset,
            window.mapping->access.original_site.original_text_offset);
      }
      if (!distance)
        continue;
      Association current{candidate_index, window_index, *distance, side};
      if (!best || current.distance < best->distance) {
        best = current;
        ambiguous = false;
      } else if (current.distance == best->distance) {
        ambiguous = true;
      }
    }
    if (best && !ambiguous)
      proposed.push_back(*best);
  }

  // Generic access and atomic helpers can both have recovered kernel owners
  // without an interprocedural CFG edge connecting their reusable bodies.
  // The emitted probe still validates dispatch, workgroup, owner, epoch,
  // Ready-window identity, and watchpoint state before it publishes. Attach
  // otherwise-unmatched atomics to the closest role-compatible window owned
  // by the same kernel; concurrent/static aliases fail through the existing
  // transactional collision path rather than overwriting metadata.
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const Candidate &candidate = candidates[candidate_index];
    for (const Association::Side missing_side :
         {Association::Side::Release, Association::Side::Acquire}) {
      if (std::ranges::any_of(proposed, [&](const Association &association) {
            return association.candidate_index == candidate_index &&
                   association.side == missing_side;
          }))
        continue;
      const uint32_t semantic_role = static_cast<uint32_t>(candidate.semantics.role);
      const bool role_has_side =
          (semantic_role & static_cast<uint32_t>(missing_side == Association::Side::Release
                                                     ? ConSanMoiSampledSyncRole::Release
                                                     : ConSanMoiSampledSyncRole::Acquire)) != 0u;
      if (!role_has_side)
        continue;
      std::optional<Association> best;
      for (size_t window_index = 0; window_index < windows.size(); ++window_index) {
        const Window &window = windows[window_index];
        if (!owner_executes_candidate(window.owner_descriptor_file_offset, candidate))
          continue;
        const bool window_has_side = missing_side == Association::Side::Release
                                         ? window.access_kind == ConSanLdsAccessKind::Write ||
                                               window.access_kind == ConSanLdsAccessKind::Atomic
                                         : window.access_kind == ConSanLdsAccessKind::Read ||
                                               window.access_kind == ConSanLdsAccessKind::Atomic;
        if (!window_has_side)
          continue;
        const uint64_t text_distance =
            window.mapping->access.original_site.original_text_offset <
                    candidate.site.text_offset
                ? candidate.site.text_offset -
                      window.mapping->access.original_site.original_text_offset
                : window.mapping->access.original_site.original_text_offset -
                      candidate.site.text_offset;
        Association current{candidate_index,
                            window_index,
                            {std::numeric_limits<uint32_t>::max(), text_distance},
                            missing_side};
        if (!best || current.distance < best->distance)
          best = current;
      }
      if (best)
        proposed.push_back(*best);
    }
  }

  // One sampled window has one metadata slot. Multiple synchronization sites
  // must not race to claim that slot: the earliest dynamic publisher would
  // otherwise win even when a later, CFG-nearer acquire is the edge that
  // actually orders the payload. Prefer the strongest static association and
  // reject equal evidence rather than making execution order part of the
  // sanitizer's answer. Window indices include their recovered kernel-owner
  // context, so mutually exclusive shared-helper owners remain independent.
  std::vector<bool> rejected(proposed.size(), false);
  for (size_t i = 0; i < proposed.size(); ++i) {
    for (size_t j = i + 1u; j < proposed.size(); ++j) {
      if (proposed[i].window_index != proposed[j].window_index)
        continue;
      if (proposed[i].distance < proposed[j].distance)
        rejected[j] = true;
      else if (proposed[j].distance < proposed[i].distance)
        rejected[i] = true;
      else
        rejected[i] = rejected[j] = true;
    }
  }

  // Each directional half may order only one selected access window. An
  // acquire-release RMW may legitimately attach its release half to a
  // preceding write and its acquire half to a later read.
  for (size_t i = 0; i < proposed.size(); ++i) {
    for (size_t j = i + 1u; j < proposed.size(); ++j) {
      if (rejected[i] || rejected[j])
        continue;
      if (proposed[i].candidate_index != proposed[j].candidate_index ||
          proposed[i].side != proposed[j].side)
        continue;
      const Window &lhs = windows[proposed[i].window_index];
      const Window &rhs = windows[proposed[j].window_index];
      if (lhs.mapping == rhs.mapping && lhs.slot == rhs.slot) {
        // The same physical access and atomic in a shared helper appear once
        // per owning kernel CFG. They need one physical patch, not competing
        // duplicate attachments.
        rejected[j] = true;
        continue;
      }
      if (proposed[i].distance < proposed[j].distance)
        rejected[j] = true;
      else if (proposed[j].distance < proposed[i].distance)
        rejected[i] = true;
      else
        rejected[i] = rejected[j] = true;
    }
  }
  std::vector<Candidate> associated;
  // One physical atomic patch can carry both directional associations.
  const size_t static_patch_budget = std::min<size_t>(options.max_patches, candidates.size());
  associated.reserve(static_patch_budget);
  for (size_t candidate_index = 0;
       candidate_index < candidates.size() && associated.size() < static_patch_budget;
       ++candidate_index) {
    std::optional<size_t> release_association;
    std::optional<size_t> acquire_association;
    for (size_t i = 0; i < proposed.size(); ++i) {
      if (rejected[i] || proposed[i].candidate_index != candidate_index)
        continue;
      if (proposed[i].side == Association::Side::Release)
        release_association = i;
      else
        acquire_association = i;
    }
    const std::optional<size_t> primary =
        acquire_association ? acquire_association : release_association;
    if (!primary)
      continue;
    Candidate candidate = candidates[candidate_index];
    const Window &primary_window = windows[proposed[*primary].window_index];
    candidate.selected_slot = primary_window.slot;
    candidate.bank_count = primary_window.bank_count;
    candidate.deferred_acquire = acquire_association.has_value();
    if (candidate.deferred_acquire && release_association) {
      const Window &release_window = windows[proposed[*release_association].window_index];
      if (release_window.bank_count == candidate.bank_count)
        candidate.release_selected_slot = release_window.slot;
      else
        result.warnings.emplace_back(
            "ConSan MOI sampled acquire-release edge has incompatible access-window banks");
    }
    associated.push_back(std::move(candidate));
  }
  std::ranges::sort(associated, {}, &Candidate::selected_slot);
  candidates = std::move(associated);
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI sampled atomic sync found no unambiguous owner/CFG-associated edge");
    return;
  }

  struct Planned : MoiPlannedProbeResources {
    Candidate candidate;
    uint64_t owner_descriptor_file_offset = 0;
    ConSanMoiAtomicAddressPlan address_plan;
    uint64_t physical_anchor_offset = 0;
    uint32_t relocated_guest_size = 0;
    std::vector<uint32_t> trailing_guest_words;
    ConSanMoiOperatingPoint point;
    bool composes_access_fragment = false;
  };
  std::vector<Planned> selected;
  MoiSpillManagers spill_managers;
  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  const size_t remaining = static_patch_budget;
  const uint16_t scratch_count = sampled_atomic_scratch_count();
  for (size_t index = 0; index < candidates.size() && selected.size() < remaining; ++index) {
    Candidate candidate = candidates[index];
    const ConSanAtomicSite &site = candidate.site;
    const auto composed_access_fragment =
        std::ranges::find_if(result.staged_text_fragments, [&](const ConSanTextFragment &fragment) {
          return (fragment.patch.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
                  fragment.patch.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore) &&
                 fragment.patch.anchor_offset == site.text_offset &&
                 fragment.patch.original_size == site.size;
        });
    const uint64_t physical_anchor_offset = site.text_offset;
    uint32_t relocated_guest_size = site.size;
    std::vector<uint32_t> trailing_guest_words;
    if (candidate.deferred_acquire &&
        composed_access_fragment == result.staged_text_fragments.end() &&
        candidate.ordered_sequence_end_text_offset > site.text_offset + site.size) {
      const uint64_t sequence_end = candidate.ordered_sequence_end_text_offset;
      const uint64_t sequence_size = sequence_end - site.text_offset;
      if (!moi_resource_offsets_share_block(resource_state, site.text_offset,
                                            sequence_end - sizeof(uint32_t)) ||
          sequence_size > std::numeric_limits<uint32_t>::max() ||
          sequence_size % sizeof(uint32_t) != 0u || sequence_end > old_text.size()) {
        result.warnings.emplace_back(
            "ConSan MOI sampled atomic sync could not relocate its bounded acquire suffix");
        continue;
      }
      relocated_guest_size = static_cast<uint32_t>(sequence_size);
      trailing_guest_words.resize((relocated_guest_size - site.size) / sizeof(uint32_t));
      std::memcpy(trailing_guest_words.data(), old_text.data() + site.text_offset + site.size,
                  trailing_guest_words.size() * sizeof(uint32_t));
    }
    if (physical_anchor_offset > old_text.size() ||
        relocated_guest_size > old_text.size() - physical_anchor_offset)
      continue;
    const bool overlaps_committed = std::any_of(
        result.patches.begin(), result.patches.end(), [&](const ConSanPatchLoweringProduct &patch) {
          const uint64_t patch_end =
              patch.anchor_offset + std::max<uint32_t>(patch.original_size, 1u);
          return site.text_offset < patch_end &&
                 patch.anchor_offset < site.text_offset + relocated_guest_size;
        });
    const bool overlaps_staged =
        std::ranges::any_of(result.staged_text_fragments, [&](const ConSanTextFragment &fragment) {
          if (composed_access_fragment != result.staged_text_fragments.end() &&
              &fragment == &*composed_access_fragment)
            return false;
          const uint64_t fragment_end =
              fragment.patch.anchor_offset + std::max<uint32_t>(fragment.patch.original_size, 1u);
          return site.text_offset < fragment_end &&
                 fragment.patch.anchor_offset < site.text_offset + relocated_guest_size;
        });
    if (overlaps_committed || overlaps_staged)
      continue;
    const ConSanCandidateResourcePlan *existing = resource_plan_for_site(
        result.resource_plans, ConSanResourceSiteKind::Atomic, site.text_offset);
    ConSanCandidateResourcePlan plan =
        existing ? *existing
                 : plan_moi_resource_site(resource_state, options, options, operating_point,
                                          ConSanResourceSiteKind::Atomic, index, site.text_offset,
                                          candidate.atomic.source_site, scratch_count, nullptr,
                                          &candidate.atomic.lowering_form);
    plan.intent_ids = {candidate.atomic.address_capture_intent, candidate.atomic.evidence_intent};
    if (!existing)
      result.resource_plans.push_back(plan);
    const auto resources =
        resolve_moi_scratch_plan(plan, operating_point, result.moi_operating_point, scratch_count);
    if (!resources) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync has no " + std::to_string(scratch_count) +
          "-VGPR resource plan in " +
          moi_evidence_container_name(result.program_inventory, candidate.atomic.source_site));
      continue;
    }
    ConSanMoiOperatingPoint candidate_point = operating_point;
    if (!apply_moi_transient_sgpr_assignment(options, candidate_point, result.moi_operating_point,
                                             resources->owner_descriptor_file_offsets)) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync lost its owner-local transient scalar assignment");
      continue;
    }
    const bool allow_post_guest_spill_operand_overlap =
        resources->source == ConSanRegisterAllocationSource::SpillRequired &&
        sampled_atomic_guest_preserves_address(candidate.atomic.lowering_form);
    const ConSanMoiAtomicAddressPlan address_plan = plan_consan_moi_atomic_address(
        candidate.atomic.lowering_form, resources->base, resources->count, resources->source,
        allow_post_guest_spill_operand_overlap);
    if (!address_plan.supported()) {
      result.warnings.emplace_back(
          "ConSan MOI sampled atomic sync rejected exact address plan: " +
          std::string(consan_moi_atomic_address_support_name(address_plan.support)));
      continue;
    }
    const std::optional<uint64_t> owner_descriptor =
        common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
    if (!owner_descriptor)
      continue;
    std::optional<ConSanMoiPrivateStateLayout> private_layout;
    if (operating_point.automatic_moi_private_epoch) {
      private_layout = build_moi_private_state_layout(
          result.program_inventory, *resources, arch, result.warnings,
          {.owner = options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
           .exact_workgroup = true});
      if (!private_layout)
        continue;
    }
    auto probe = plan_moi_probe_resources(
        result.program_inventory, *resources, options, options, candidate_point, mode_semantics,
        spill_managers, arch, std::move(private_layout),
        candidate_point.has_compact_moi_scalar_spill(), result.warnings);
    if (!probe)
      continue;
    note_moi_probe_private_requirements(private_requirements, *probe);
    note_descriptor_requirements(descriptor_requirements, probe->resources);
    selected.push_back({std::move(*probe), std::move(candidate), *owner_descriptor, address_plan,
                        physical_anchor_offset, relocated_guest_size,
                        std::move(trailing_guest_words), std::move(candidate_point),
                        composed_access_fragment != result.staged_text_fragments.end()});
  }
  if (selected.empty()) {
    result.warnings.emplace_back("ConSan MOI sampled atomic sync found no patchable sequence");
    return;
  }
  MoiDescriptorSgprRequirements scalar_requirements;
  std::vector<ConSanTextFragment> staged_fragments;
  std::vector<ConSanPatchInfo> scalar_clause_patches;
  staged_fragments.reserve(selected.size() * 2u);
  for (const Planned &planned : selected) {
    if (!append_atomic_scalar_clause_patch(
            old_text, planned.candidate.site,
            planned.candidate.atomic.scalar_clause_text_offset, arch, scalar_clause_patches,
            result.errors))
      return;
    note_moi_sgpr_requirements(scalar_requirements, planned.resources, options, options,
                               planned.point, mode_semantics, arch);
    const auto emission_plan = plan_sampled_sync_emission(
        options, options, planned.point, result.moi_operating_point,
        planned.resources.owner_descriptor_file_offsets,
        planned.private_layout ? &*planned.private_layout : nullptr, planned.resources.base,
        static_cast<uint16_t>(planned.resources.base + SampledAtomicScratchLayout::kOwner),
        static_cast<uint16_t>(planned.resources.base + SampledAtomicScratchLayout::kEpoch),
        /*transient_assignment_already_resolved=*/true, "sampled atomic sync", result.errors);
    if (!emission_plan)
      return;
    uint32_t guest_instruction_offset = 0u;
    uint32_t emitted_guest_size = 0u;
    auto words =
        planned.candidate.deferred_acquire
            ? build_sampled_pending_acquire_cave_words(
                  bytes, planned.candidate.atomic, planned.candidate.site,
                  planned.owner_descriptor_file_offset,
                  planned.address_plan, *emission_plan, planned.spill ? &*planned.spill : nullptr,
                  planned.scalar_spill ? &*planned.scalar_spill : nullptr,
                  planned.private_layout ? &*planned.private_layout : nullptr, arch,
                  planned.candidate.selected_slot, planned.candidate.bank_count,
                  planned.candidate.release_selected_slot, layout, result.errors,
                  &guest_instruction_offset, planned.trailing_guest_words, &emitted_guest_size)
            : build_sampled_atomic_sync_cave_words(
                  bytes, planned.candidate.atomic, planned.candidate.site,
                  planned.owner_descriptor_file_offset,
                  planned.address_plan, *emission_plan, planned.spill ? &*planned.spill : nullptr,
                  planned.scalar_spill ? &*planned.scalar_spill : nullptr,
                  planned.private_layout ? &*planned.private_layout : nullptr, arch,
                  planned.candidate.selected_slot, planned.candidate.bank_count, layout,
                  result.errors, &guest_instruction_offset, planned.trailing_guest_words,
                  &emitted_guest_size);
    if (!words)
      return;
    ConSanPatchInfo patch;
    patch.kind = ConSanPatchKind::TrampolineMoiSampledSyncMetadata;
    patch.anchor_offset = planned.candidate.site.text_offset;
    patch.original_size = planned.relocated_guest_size;
    note_moi_probe_patch_info(patch, planned);
    auto fragment = make_consan_around_text_fragment(
        std::move(*words), guest_instruction_offset, planned.relocated_guest_size,
        planned.candidate.atomic.intent_ids(), {}, std::move(patch), result.errors,
        "MOI sampled atomic sync", emitted_guest_size);
    if (!fragment)
      return;
    if (planned.composes_access_fragment)
      fragment->replacement_words.clear();
    staged_fragments.push_back(std::move(*fragment));
  }
  for (ConSanPatchInfo &patch : scalar_clause_patches)
    staged_fragments.push_back(
        ConSanTextFragment::replacement({build_s_nop(0u, arch)}, std::move(patch)));
  const MoiDescriptorLdsRequirements no_lds;
  (void)stage_consan_text_rewrites(
      code_object, arch,
      {descriptor_requirements, scalar_requirements, private_requirements, no_lds},
      moi_descriptor_mutation_policy(nullptr, arch), "MOI sampled atomic sync",
      std::move(staged_fragments), result);
  return;
}
