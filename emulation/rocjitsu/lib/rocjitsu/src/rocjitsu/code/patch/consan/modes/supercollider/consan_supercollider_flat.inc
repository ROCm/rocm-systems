[[nodiscard]] uint16_t flat_check_trap_comparison_scratch_vgprs(const ConSanProgramSite &access) {
  if (!access.lowering.form)
    return 0u;
  const ConSanAccessLoweringForm &form = *access.lowering.form;
  return form.register_value_placement == ConSanAccessRegisterValuePlacement::High16 ||
                 (form.access_kind == ConSanLdsAccessKind::Write && form.element_width_bits == 8u)
             ? 1u
             : 0u;
}

void try_apply_flat_check_trap_patch(const AmdGpuCodeObject &code_object,
                                     const ConSanTargetProfile &target,
                                     const ConSanOptions &options,
                                     ConSanTransformArtifacts &result) {
  const rj_code_arch_t arch = target.arch;

  const std::unordered_set<uint64_t> admitted_file_offsets = sc_admitted_access_file_offsets(
      result.program_inventory, result.coverage_ledger, ConSanAccessOrigin::Flat);
  std::map<uint64_t, const ConSanProgramSite *> admitted_sites;
  for (const ConSanProgramSite &access : result.program_inventory.access_sites()) {
    if (access.origin != ConSanAccessOrigin::Flat ||
        !admitted_file_offsets.contains(access.decoded_file_offset()))
      continue;
    admitted_sites.try_emplace(access.decoded_file_offset(), &access);
  }
  if (admitted_sites.empty()) {
    result.warnings.emplace_back(
        "ConSan flat check/trap proof found no supported likely group-FLAT load/store site");
    return;
  }

  const auto delay_words =
      consan_sc_build_delay_words(arch, options, result.errors, "ConSan flat check/trap proof");
  if (!delay_words)
    return;

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  ScOwnerAnalysis owner_analysis(code_object, result.program_inventory, arch,
                                 /*restrict_to_preflight_candidates=*/false);

  struct FlatCheckTrapCandidate {
    const ConSanProgramSite *site = nullptr;
    uint16_t scratch_vgpr = 0;
    std::vector<uint64_t> owner_descriptor_file_offsets;
    uint16_t vcc_save_sgpr = 0;
    std::optional<ConSanSuperColliderScalarVccSpill> scalar_vcc_spill;
    std::optional<VgprSpillSequence> vgpr_spill;
    std::optional<DynamicStackBorrowedSgprSpillSequence> dynamic_stack_spill;
  };

  std::vector<FlatCheckTrapCandidate> candidates;
  size_t supported_candidate_count = 0;
  size_t scratchable_candidate_count = 0;
  size_t spill_backed_candidate_count = 0;
  size_t mixed_stack_owner_spill_rejection_count = 0;
  size_t dynamic_stack_spill_encoding_failure_count = 0;
  size_t private_spill_encoding_failure_count = 0;
  auto visit_site = [&](const ConSanProgramSite &site, const Instruction *site_inst,
                        std::optional<uint16_t> min_auto_scratch_vgpr,
                        std::optional<uint16_t> max_auto_scratch_vgpr,
                        std::optional<uint16_t> min_fresh_sgpr,
                        std::optional<uint16_t> allocated_sgpr_count,
                        uint32_t original_private_segment_size, bool any_dynamic_stack,
                        bool every_dynamic_stack,
                        std::span<const LivenessAnalysis *const> owner_liveness,
                        std::vector<uint64_t> owner_descriptor_file_offsets) {
    ++supported_candidate_count;
    auto required_vgprs = access_dword_count(site);
    if (!required_vgprs)
      return;
    const uint32_t comparison_scratch_vgprs = flat_check_trap_comparison_scratch_vgprs(site);
    const uint32_t probe_scratch_vgprs = *required_vgprs + comparison_scratch_vgprs;
    const uint32_t report_alignment_padding =
        options.report_buffer_address && target.requires_even_vgpr_tuples ? 1u : 0u;
    const uint32_t instrumentation_scratch_vgprs =
        probe_scratch_vgprs + report_alignment_padding + report_action_scratch_vgprs(options);
    const bool uses_dynamic_stack = any_dynamic_stack && every_dynamic_stack;
    const bool preserve_dynamic_stack_abi = any_dynamic_stack;
    auto vcc_save = choose_sc_dead_vcc_save_sgpr(
        site_inst, owner_liveness, allocated_sgpr_count.value_or(0u), min_fresh_sgpr,
        REGISTER_SET_ALLOCATABLE_SGPRS, 2u, false, options, preserve_dynamic_stack_abi);
    bool spill_vcc_save_pair_to_vgprs = false;
    if (!vcc_save && target.scalar_placement_model == ConSanScalarPlacementModel::SpillBacked &&
        allocated_sgpr_count) {
      vcc_save = choose_sc_borrowed_scalar_range(*allocated_sgpr_count, 2u, options,
                                                 preserve_dynamic_stack_abi);
      spill_vcc_save_pair_to_vgprs = vcc_save.has_value();
    }
    if (!vcc_save) {
      (void)publish_sc_access_rejection(
          result.coverage_ledger, result.errors, site.physical_id,
          ConSanLoweringOutcomeKind::ResourceRejected,
          "SuperCollider FLAT lowerer found no safe VCC-save scalar pair");
      return;
    }
    uint32_t scratch_vgprs =
        instrumentation_scratch_vgprs + (spill_vcc_save_pair_to_vgprs ? 1u : 0u);
    auto scratch = choose_sc_access_scratch_vgpr(
        site, options, site_inst, owner_liveness,
        std::max(access_scratch_search_start(site), min_auto_scratch_vgpr.value_or(0u)),
        max_auto_scratch_vgpr, scratch_vgprs, site.lowering.form->data_register_alignment,
        min_auto_scratch_vgpr.value_or(0u));
    std::optional<VgprSpillSequence> vgpr_spill;
    std::optional<DynamicStackBorrowedSgprSpillSequence> dynamic_stack_spill;
    if (!scratch && !options.scratch_vgpr && max_auto_scratch_vgpr) {
      if (any_dynamic_stack && !every_dynamic_stack) {
        ++mixed_stack_owner_spill_rejection_count;
      } else {
        // Once the VGPR window itself is live, preserve it before using it as
        // instrumentation scratch. A borrowed scalar pair additionally needs
        // one active-lane reservoir per scalar. Dynamic-stack bootstrapping
        // always preserves the selected pair because that pair temporarily
        // carries both frame state and VCC, even when it was dead at the site.
        // An EXECZ guard leaves empty waves untouched.
        const uint16_t scalar_reservoir_count =
            uses_dynamic_stack ? DynamicStackBorrowedSgprSpillSequence::kScalarReservoirCount
                               : static_cast<uint16_t>(spill_vcc_save_pair_to_vgprs ? 2u : 0u);
        scratch_vgprs = instrumentation_scratch_vgprs + scalar_reservoir_count;
        scratch = choose_spill_scratch_vgpr(site, *max_auto_scratch_vgpr, scratch_vgprs);
        if (scratch && uses_dynamic_stack) {
          dynamic_stack_spill = build_dynamic_stack_borrowed_sgpr_spill_sequence(
              *scratch, scratch_vgprs, *vcc_save,
              static_cast<uint16_t>(*scratch + instrumentation_scratch_vgprs),
              original_private_segment_size, arch);
        } else if (scratch) {
          const auto private_limit = consan_address_free_private_limit(arch);
          if (private_limit) {
            SpillManager manager(original_private_segment_size, *private_limit);
            vgpr_spill = build_vgpr_spill_sequence(manager, *scratch, scratch_vgprs, arch);
          }
        }
        if (vgpr_spill || dynamic_stack_spill) {
          if (dynamic_stack_spill)
            spill_vcc_save_pair_to_vgprs = true;
          ++spill_backed_candidate_count;
        } else {
          scratch.reset();
          if (uses_dynamic_stack)
            ++dynamic_stack_spill_encoding_failure_count;
          else
            ++private_spill_encoding_failure_count;
        }
      }
    }
    if (!scratch) {
      (void)publish_sc_access_rejection(
          result.coverage_ledger, result.errors, site.physical_id,
          ConSanLoweringOutcomeKind::ResourceRejected,
          "SuperCollider FLAT lowerer found no encodable scratch or spill plan");
      return;
    }
    const std::optional<ConSanSuperColliderScalarVccSpill> scalar_vcc_spill =
        spill_vcc_save_pair_to_vgprs
            ? std::optional<ConSanSuperColliderScalarVccSpill>{ConSanSuperColliderScalarVccSpill{
                  .vcc_save_sgpr = *vcc_save,
                  .reservoir_vgpr = static_cast<uint16_t>(*scratch + instrumentation_scratch_vgprs),
                  .reservoir = dynamic_stack_spill
                                   ? ConSanSuperColliderScalarVccReservoir::DynamicStackBootstrap
                               : vgpr_spill ? ConSanSuperColliderScalarVccReservoir::PrivateSpill
                                            : ConSanSuperColliderScalarVccReservoir::PackedLanes,
              }}
            : std::nullopt;
    ++scratchable_candidate_count;
    candidates.push_back({.site = &site,
                          .scratch_vgpr = *scratch,
                          .owner_descriptor_file_offsets = std::move(owner_descriptor_file_offsets),
                          .vcc_save_sgpr = *vcc_save,
                          .scalar_vcc_spill = scalar_vcc_spill,
                          .vgpr_spill = vgpr_spill,
                          .dynamic_stack_spill = dynamic_stack_spill});
  };

  for (const auto &[file_offset, access] : admitted_sites) {
    (void)file_offset;
    const ConSanProgramSite &site = *access;
    const ConSanProgramContainer *site_container =
        result.program_inventory.container(site.container);
    if (site_container == nullptr)
      continue;
    std::vector<ConSanProgramContainerId> owners =
        result.program_inventory.execution_owner_kernels(site);
    for (const ConSanFaultSite &fault : result.fault_sites) {
      const ConSanProgramSite *fault_source = result.program_inventory.program_site(fault);
      if (fault_source == nullptr || fault_source->container != site.container ||
          site_container->is_kernel())
        continue;
      const std::vector<ConSanProgramContainerId> fault_owners =
          result.program_inventory.execution_owner_kernels(fault_source->execution_owners);
      owners.insert(owners.end(), fault_owners.begin(), fault_owners.end());
    }
    auto facts = owner_analysis.site_facts(site, std::move(owners));
    if (!facts || !facts->common_unified_vgpr_count || !facts->common_sgpr_count) {
      (void)publish_sc_access_rejection(
          result.coverage_ledger, result.errors, site.physical_id,
          ConSanLoweringOutcomeKind::ResourceRejected,
          "SuperCollider FLAT lowerer found no complete planning context");
      continue;
    }

    const bool selected_by_filter =
        options.test_kernel_name_filter.empty() ||
        site_container->name.find(options.test_kernel_name_filter) != std::string::npos ||
        std::ranges::any_of(facts->contexts, [&](const ScOwnerContext *owner) {
          return owner != nullptr &&
                 owner->kernel->name.find(options.test_kernel_name_filter) != std::string::npos;
        });
    if (!selected_by_filter)
      continue;

    std::optional<uint16_t> min_auto_scratch_vgpr;
    if (facts->max_vgpr_ref && *facts->max_vgpr_ref < 255u)
      min_auto_scratch_vgpr = static_cast<uint16_t>(*facts->max_vgpr_ref + 1u);
    std::optional<uint16_t> min_fresh_sgpr;
    if (facts->max_sgpr_ref && *facts->max_sgpr_ref + 1u < REGISTER_SET_ALLOCATABLE_SGPRS)
      min_fresh_sgpr = static_cast<uint16_t>(*facts->max_sgpr_ref + 1u);
    visit_site(site, facts->instruction, min_auto_scratch_vgpr, facts->common_unified_vgpr_count,
               min_fresh_sgpr, facts->common_sgpr_count, facts->max_private_segment_size,
               facts->any_dynamic_stack, facts->every_dynamic_stack, facts->liveness,
               std::move(facts->descriptor_file_offsets));
  }
  if (!result.errors.empty())
    return;
  if (options.max_patches_is_expert_limit && options.max_patches == 0u) {
    result.warnings.emplace_back(
        "ConSan flat check/trap proof skipped: shared patch budget consumed");
    return;
  }

  const size_t max_patches =
      options.max_patches_is_expert_limit ? options.max_patches : supported_candidate_count;
  std::ranges::sort(candidates, {}, [](const FlatCheckTrapCandidate &candidate) {
    return candidate.site->decoded_file_offset();
  });
  if (candidates.size() > max_patches)
    candidates.resize(max_patches);
  std::vector<FlatCheckTrapCandidate> selected_candidates = std::move(candidates);

  const size_t selection_target = std::min(max_patches, supported_candidate_count);
  if (selected_candidates.size() < selection_target) {
    result.warnings.emplace_back(
        "ConSan flat check/trap proof selected only " + std::to_string(selected_candidates.size()) +
        "/" + std::to_string(selection_target) + " supported sites; " +
        "mixed_stack_owner_spill_rejections=" +
        std::to_string(mixed_stack_owner_spill_rejection_count) +
        " dynamic_stack_spill_encoding_failures=" +
        std::to_string(dynamic_stack_spill_encoding_failure_count) +
        " private_spill_encoding_failures=" + std::to_string(private_spill_encoding_failure_count));
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan flat check/trap proof found no selectable likely group-FLAT load/store site; "
        "supported_candidates=" +
        std::to_string(supported_candidate_count) +
        " scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " spill_backed_candidates=" + std::to_string(spill_backed_candidate_count) +
        " mixed_stack_owner_spill_rejections=" +
        std::to_string(mixed_stack_owner_spill_rejection_count) +
        " dynamic_stack_spill_encoding_failures=" +
        std::to_string(dynamic_stack_spill_encoding_failure_count) +
        " private_spill_encoding_failures=" + std::to_string(private_spill_encoding_failure_count));
    return;
  }

  struct PlannedFlatCheckTrapPatch {
    const ConSanProgramSite *site = nullptr;
    std::vector<ConSanProbeIntentId> intent_ids;
    std::vector<uint32_t> words;
    ConSanPatchInfo info;
  };

  const auto wait_flat = consan_sc_build_guest_flat_completion_wait(arch);
  if (!wait_flat) {
    result.errors.emplace_back("ConSan flat check/trap proof could not encode sequence");
    return;
  }

  std::vector<PlannedFlatCheckTrapPatch> planned_patches;
  planned_patches.reserve(selected_candidates.size());
  for (const FlatCheckTrapCandidate &candidate : selected_candidates) {
    if (candidate.site == nullptr) {
      result.errors.emplace_back("ConSan flat check/trap proof selected missing site");
      return;
    }
    const ConSanProgramSite &site = *candidate.site;
    const auto required_vgprs = access_dword_count(site);
    const auto compare_vgpr = flat_check_trap_compare_vgpr(site);
    if (!required_vgprs || !compare_vgpr) {
      result.errors.emplace_back("ConSan flat check/trap proof selected unsupported site");
      return;
    }

    const size_t access_words = site.size() / sizeof(uint32_t);
    if (access_words < 2u || access_words > 3u) {
      result.errors.emplace_back("ConSan flat check/trap proof selected invalid access size");
      return;
    }
    std::array<uint32_t, 3> original_access{};
    if (!read_words_at(original_bytes, site.decoded_file_offset(),
                       std::span<uint32_t>(original_access).first(access_words))) {
      result.errors.emplace_back("ConSan flat check/trap proof could not read access");
      return;
    }

    std::optional<std::array<uint32_t, 3>> duplicate_load;
    ConSanPatchKind patch_kind = ConSanPatchKind::FlatLoadCheckTrap;
    if (!access_scratch_tuple_base_is_valid(site, candidate.scratch_vgpr)) {
      result.errors.emplace_back(
          "ConSan flat check/trap proof selected a misaligned tuple scratch base");
      return;
    }
    if (site.kind == ConSanLdsAccessKind::Read) {
      duplicate_load = retarget_flat_load_vdst(original_access, candidate.scratch_vgpr, arch);
    } else if (site.kind == ConSanLdsAccessKind::Write) {
      duplicate_load = build_flat_load_from_flat_store(
          original_access, site.lowering.form->element_width_bits, candidate.scratch_vgpr, arch);
      patch_kind = ConSanPatchKind::FlatStoreCheckTrap;
    }
    if (!duplicate_load) {
      result.errors.emplace_back("ConSan flat check/trap proof could not encode readback");
      return;
    }

    PlannedFlatCheckTrapPatch planned;
    planned.site = &site;
    planned.intent_ids = result.coverage_ledger.pending_intent_ids(
        ConSanProbeIntentKind::RedundantAccessObservation, site.physical_id);
    if (planned.intent_ids.empty()) {
      result.errors.emplace_back(
          "ConSan flat check/trap proof selected a site without a pending semantic intent");
      return;
    }
    planned.info.kind = patch_kind;
    planned.info.anchor_offset = site.physical_id.original_text_offset;
    planned.info.scratch_vgpr = candidate.scratch_vgpr;
    planned.info.sc_scalar_vcc_spill = candidate.scalar_vcc_spill;
    planned.info.owner_descriptor_file_offsets = candidate.owner_descriptor_file_offsets;
    planned.info.required_sgpr_count = static_cast<uint16_t>(candidate.vcc_save_sgpr + 2u);
    if (candidate.vgpr_spill) {
      planned.info.spilled_vgpr_count = candidate.vgpr_spill->vgpr_count;
      planned.info.required_private_segment_size = candidate.vgpr_spill->total_private_bytes;
    } else if (candidate.dynamic_stack_spill) {
      planned.info.spilled_vgpr_count = candidate.dynamic_stack_spill->vgpr_count;
      planned.info.required_private_segment_size =
          candidate.dynamic_stack_spill->total_private_bytes;
      planned.info.dynamic_private_segment_addend =
          candidate.dynamic_stack_spill->dynamic_frame_bytes;
    }
    InstructionSequence sequence(planned.words);
    const auto restore_vcc =
        instrumentation::build_s_mov_b64(kAmdGpuVccLo, candidate.vcc_save_sgpr, arch);
    if (!sequence.emit(
            instrumentation::build_s_mov_b64(candidate.vcc_save_sgpr, kAmdGpuVccLo, arch)) ||
        !restore_vcc) {
      result.errors.emplace_back("ConSan flat check/trap proof could not preserve VCC");
      return;
    }
    (void)sequence.emit(std::span(original_access).first(access_words));
    planned.words.insert(planned.words.end(), delay_words->begin(), delay_words->end());
    const bool needs_runtime_group_gate =
        target.requires_sc_runtime_flat_group_gate &&
        site.flat_address_space_hint == ConSanFlatAddressSpaceHint::MaybeGroup;
    std::optional<InstructionSequence::Label> runtime_group_gate_done;
    if (needs_runtime_group_gate) {
      constexpr uint16_t kScalarInlineMinusOne = 0xC1;
      runtime_group_gate_done = sequence.make_label();
      if (!sequence.emit(instrumentation::build_v_cmp_eq_u32_vcc(
              kScalarInlineMinusOne, static_cast<uint16_t>(*site.lowering.form->address_vgpr + 1u),
              arch)) ||
          !sequence.emit_branch(*runtime_group_gate_done,
                                InstructionSequence::BranchKind::VccZero)) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof could not encode runtime group gate");
        return;
      }
    }
    (void)sequence.emit(std::span(*duplicate_load).first(access_words));
    (void)sequence.emit(*wait_flat);
    const ConSanAccessLoweringForm &form = *site.lowering.form;
    const bool subword = form.element_width_bits < 32u;
    const bool compare_high_load =
        form.access_kind == ConSanLdsAccessKind::Read &&
        form.register_value_placement == ConSanAccessRegisterValuePlacement::High16;
    const bool compare_subword_store = form.access_kind == ConSanLdsAccessKind::Write && subword;
    const uint16_t comparison_scratch_vgprs = flat_check_trap_comparison_scratch_vgprs(site);
    uint16_t report_scratch_vgpr =
        static_cast<uint16_t>(candidate.scratch_vgpr + *required_vgprs + comparison_scratch_vgprs);
    if (options.report_buffer_address && target.requires_even_vgpr_tuples &&
        report_scratch_vgpr % 2u != 0)
      ++report_scratch_vgpr;
    auto mismatch_action = build_mismatch_action_words(
        options, target, report_scratch_vgpr, result.errors, "ConSan flat check/trap proof");
    if (!mismatch_action)
      return;
    if (*required_vgprs > 1u) {
      const InstructionSequence::Label mismatch_action_label = sequence.make_label();
      const InstructionSequence::Label mismatch_done = sequence.make_label();
      for (uint16_t i = 0; i < *required_vgprs; ++i) {
        if (!sequence.emit(consan_sc_build_v_cmp_ne_u32(
                static_cast<uint16_t>(*compare_vgpr + i),
                static_cast<uint16_t>(candidate.scratch_vgpr + i), arch)) ||
            !sequence.emit_branch(mismatch_action_label,
                                  InstructionSequence::BranchKind::VccNonzero)) {
          result.errors.emplace_back("ConSan flat check/trap proof could not encode chunk compare");
          return;
        }
      }
      if (!sequence.emit_branch(mismatch_done, InstructionSequence::BranchKind::Unconditional) ||
          !sequence.bind(mismatch_action_label) || !sequence.emit(*mismatch_action) ||
          !sequence.bind(mismatch_done)) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof could not encode shared mismatch action");
        return;
      }
    } else {
      uint16_t compare_lhs_vgpr = *compare_vgpr;
      uint16_t compare_rhs_vgpr = candidate.scratch_vgpr;
      const uint16_t comparison_scratch_vgpr =
          static_cast<uint16_t>(candidate.scratch_vgpr + *required_vgprs);
      if (compare_high_load) {
        const uint16_t original_high_vgpr = comparison_scratch_vgpr;
        const auto select_original_high = instrumentation::build_v_lshrrev_b32(
            original_high_vgpr, scalar_positive_inline_u32(16u), *compare_vgpr, arch);
        const auto select_duplicate_high = instrumentation::build_v_lshrrev_b32(
            candidate.scratch_vgpr, scalar_positive_inline_u32(16u), candidate.scratch_vgpr, arch);
        if (!select_original_high || !select_duplicate_high) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not select high-half load values");
          return;
        }
        planned.words.push_back(*select_original_high);
        planned.words.push_back(*select_duplicate_high);
        compare_lhs_vgpr = original_high_vgpr;
      } else if (compare_subword_store) {
        if (form.register_value_placement == ConSanAccessRegisterValuePlacement::High16) {
          const auto select_store_high = instrumentation::build_v_lshrrev_b32(
              comparison_scratch_vgpr, scalar_positive_inline_u32(16u), *compare_vgpr, arch);
          if (!select_store_high) {
            result.errors.emplace_back(
                "ConSan flat check/trap proof could not select high-half store value");
            return;
          }
          planned.words.push_back(*select_store_high);
          compare_lhs_vgpr = comparison_scratch_vgpr;
        }
        if (form.element_width_bits == 8u) {
          const auto mask_store_byte = instrumentation::build_v_and_b32_literal(
              comparison_scratch_vgpr, 0xffu, compare_lhs_vgpr, arch);
          if (!mask_store_byte) {
            result.errors.emplace_back(
                "ConSan flat check/trap proof could not mask byte store value");
            return;
          }
          planned.words.insert(planned.words.end(), mask_store_byte->begin(),
                               mask_store_byte->end());
          compare_lhs_vgpr = comparison_scratch_vgpr;
        }
      }
      const auto chunk_cmp_ne =
          subword ? consan_sc_build_v_cmp_ne_u16(compare_lhs_vgpr, compare_rhs_vgpr, arch)
                  : consan_sc_build_v_cmp_ne_u32(compare_lhs_vgpr, compare_rhs_vgpr, arch);
      const InstructionSequence::Label mismatch_done = sequence.make_label();
      if (!sequence.emit(chunk_cmp_ne) ||
          !sequence.emit_branch(mismatch_done, InstructionSequence::BranchKind::VccZero) ||
          !sequence.emit(*mismatch_action) || !sequence.bind(mismatch_done)) {
        result.errors.emplace_back("ConSan flat check/trap proof could not encode chunk compare");
        return;
      }
    }
    if ((runtime_group_gate_done && !sequence.bind(*runtime_group_gate_done)) ||
        !sequence.emit(restore_vcc) || !sequence.resolve_branches(arch)) {
      result.errors.emplace_back("ConSan flat check/trap proof could not resolve local branches");
      return;
    }
    if (candidate.vgpr_spill || candidate.dynamic_stack_spill) {
      if (candidate.dynamic_stack_spill) {
        if (!candidate.scalar_vcc_spill ||
            candidate.scalar_vcc_spill->reservoir !=
                ConSanSuperColliderScalarVccReservoir::DynamicStackBootstrap) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof lost its dynamic scalar spill window");
          return;
        }
        std::vector<uint32_t> wrapped;
        InstructionSequence wrapped_sequence(wrapped);
        const InstructionSequence::Label spill_done = wrapped_sequence.make_label();
        if (!wrapped_sequence.emit_branch(spill_done, InstructionSequence::BranchKind::ExecZero) ||
            !wrapped_sequence.emit(candidate.dynamic_stack_spill->save_words) ||
            !wrapped_sequence.emit(planned.words) ||
            !wrapped_sequence.emit(candidate.dynamic_stack_spill->restore_words) ||
            !wrapped_sequence.bind(spill_done) || !wrapped_sequence.resolve_branches(arch)) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not encode its dynamic empty-wave spill guard");
          return;
        }
        planned.words = std::move(wrapped);
      } else {
        const bool spills_scalar_pair = candidate.scalar_vcc_spill.has_value();
        if (spills_scalar_pair && candidate.scalar_vcc_spill->reservoir !=
                                      ConSanSuperColliderScalarVccReservoir::PrivateSpill) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof has an invalid scalar spill window");
          return;
        }
        std::array<uint32_t, 2> save_scalar_pair{};
        std::array<uint32_t, 2> restore_scalar_pair{};
        if (spills_scalar_pair) {
          for (uint16_t i = 0; i < 2u; ++i) {
            save_scalar_pair[i] = build_v_mov_b32_e32(
                static_cast<uint16_t>(candidate.scalar_vcc_spill->reservoir_vgpr + i),
                static_cast<uint16_t>(candidate.scalar_vcc_spill->vcc_save_sgpr + i), arch);
            const auto restore = instrumentation::build_v_readfirstlane_b32(
                static_cast<uint16_t>(candidate.scalar_vcc_spill->vcc_save_sgpr + i),
                static_cast<uint16_t>(candidate.scalar_vcc_spill->reservoir_vgpr + i), arch);
            if (!restore) {
              result.errors.emplace_back(
                  "ConSan flat check/trap proof could not encode its active-lane scalar restore");
              return;
            }
            restore_scalar_pair[i] = *restore;
          }
        }
        const size_t guarded_words = candidate.vgpr_spill->save_words.size() +
                                     (spills_scalar_pair ? save_scalar_pair.size() : 0u) +
                                     planned.words.size() +
                                     (spills_scalar_pair ? restore_scalar_pair.size() : 0u) +
                                     candidate.vgpr_spill->restore_words.size();
        if (guarded_words > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof spill guard exceeds s_cbranch_execz range");
          return;
        }
        const auto skip_empty_wave =
            instrumentation::build_s_cbranch_execz(static_cast<int16_t>(guarded_words), arch);
        if (!skip_empty_wave) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not encode its empty-wave spill guard");
          return;
        }
        std::vector<uint32_t> wrapped;
        wrapped.reserve(1u + guarded_words);
        wrapped.push_back(*skip_empty_wave);
        wrapped.insert(wrapped.end(), candidate.vgpr_spill->save_words.begin(),
                       candidate.vgpr_spill->save_words.end());
        if (spills_scalar_pair)
          wrapped.insert(wrapped.end(), save_scalar_pair.begin(), save_scalar_pair.end());
        wrapped.insert(wrapped.end(), planned.words.begin(), planned.words.end());
        if (spills_scalar_pair)
          wrapped.insert(wrapped.end(), restore_scalar_pair.begin(), restore_scalar_pair.end());
        wrapped.insert(wrapped.end(), candidate.vgpr_spill->restore_words.begin(),
                       candidate.vgpr_spill->restore_words.end());
        planned.words = std::move(wrapped);
      }
    } else if (candidate.scalar_vcc_spill) {
      if (candidate.scalar_vcc_spill->reservoir !=
          ConSanSuperColliderScalarVccReservoir::PackedLanes) {
        result.errors.emplace_back(
            "ConSan flat check/trap proof has an invalid fixed-lane scalar spill window");
        return;
      }
      std::array<uint32_t, 4> save_scalar_pair{};
      std::array<uint32_t, 4> restore_scalar_pair{};
      for (uint16_t i = 0; i < 2u; ++i) {
        const auto save = instrumentation::build_v_writelane_b32(
            candidate.scalar_vcc_spill->reservoir_vgpr,
            static_cast<uint16_t>(candidate.scalar_vcc_spill->vcc_save_sgpr + i), i, arch);
        const auto restore = instrumentation::build_v_readlane_b32(
            static_cast<uint16_t>(candidate.scalar_vcc_spill->vcc_save_sgpr + i),
            candidate.scalar_vcc_spill->reservoir_vgpr, i, arch);
        if (!save || !restore) {
          result.errors.emplace_back(
              "ConSan flat check/trap proof could not encode its scalar VCC-save spill");
          return;
        }
        std::copy(save->begin(), save->end(), save_scalar_pair.begin() + 2u * i);
        std::copy(restore->begin(), restore->end(), restore_scalar_pair.begin() + 2u * i);
      }
      std::vector<uint32_t> wrapped;
      wrapped.reserve(save_scalar_pair.size() + planned.words.size() + restore_scalar_pair.size());
      wrapped.insert(wrapped.end(), save_scalar_pair.begin(), save_scalar_pair.end());
      wrapped.insert(wrapped.end(), planned.words.begin(), planned.words.end());
      wrapped.insert(wrapped.end(), restore_scalar_pair.begin(), restore_scalar_pair.end());
      planned.words = std::move(wrapped);
    }

    planned.info.original_size = site.size();
    planned.info.trampoline_size = static_cast<uint32_t>(planned.words.size() * sizeof(uint32_t));

    planned_patches.push_back(std::move(planned));
  }
  ConSanDescriptorRegisterRequirements descriptor_sgpr_growths;
  const ConSanDescriptorRegisterRequirements no_vgpr_requirements;
  const ConSanDescriptorMemoryRequirements no_group_segment_requirements;
  std::unordered_map<uint64_t, uint32_t> descriptor_private_requirements;
  for (const PlannedFlatCheckTrapPatch &planned : planned_patches) {
    for (uint64_t owner : planned.info.owner_descriptor_file_offsets) {
      note_maximum_descriptor_extent(descriptor_sgpr_growths, owner,
                                     planned.info.required_sgpr_count);
      if (planned.info.required_private_segment_size != 0) {
        uint32_t &required = descriptor_private_requirements[owner];
        required = std::max(required, planned.info.required_private_segment_size);
      }
    }
  }
  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(planned_patches.size());
  for (PlannedFlatCheckTrapPatch &planned : planned_patches)
    fragments.push_back(ConSanTextFragment::replacement(
        std::move(planned.words), std::move(planned.info), std::move(planned.intent_ids)));
  (void)stage_consan_text_rewrites(code_object, arch,
                                   {no_vgpr_requirements, descriptor_sgpr_growths,
                                    descriptor_private_requirements, no_group_segment_requirements},
                                   {.maximum_ordinary_vgpr_count = REGISTER_SET_MAX_VGPRS,
                                    .inventory_proves_empty_accumulator_bank = false,
                                    .maximum_group_segment_bytes = std::nullopt},
                                   "ConSan flat check/trap proof", std::move(fragments), result);
}
