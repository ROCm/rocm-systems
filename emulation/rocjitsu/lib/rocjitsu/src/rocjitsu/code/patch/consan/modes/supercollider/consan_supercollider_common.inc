[[nodiscard]] std::unordered_set<uint64_t>
sc_admitted_access_file_offsets(const ProgramInventory &inventory,
                                const ConSanCoverageLedger &coverage, ConSanAccessOrigin origin) {
  std::unordered_set<uint64_t> offsets;
  for (const ConSanProgramSite &access : inventory.access_sites())
    if (access.origin == origin &&
        !coverage
             .pending_intent_ids(ConSanProbeIntentKind::RedundantAccessObservation,
                                 access.physical_id)
             .empty())
      offsets.insert(access.decoded_file_offset());
  return offsets;
}

[[nodiscard]] bool publish_sc_access_rejection(ConSanCoverageLedger &coverage,
                                               std::vector<std::string> &errors,
                                               const PhysicalSiteId &site,
                                               ConSanLoweringOutcomeKind outcome,
                                               std::string detail) {
  std::vector<ConSanProbeIntentId> intent_ids =
      coverage.pending_intent_ids(ConSanProbeIntentKind::RedundantAccessObservation, site);
  if (intent_ids.empty())
    return true;
  if (!coverage.publish_lowering_rejection(intent_ids, outcome, std::move(detail))) {
    errors.emplace_back("ConSan SuperCollider could not publish an intent-bound access rejection");
    return false;
  }
  return true;
}

/// Publish an explicit placement rejection for every SuperCollider access
/// intent that no site-local lowering transaction consumed. This fallback
/// consults only immutable intent identity and the ledger's pending state; it
/// never reconstructs success or failure from patch telemetry.
void publish_unplaced_sc_access_rejections(ConSanCoverageLedger &coverage,
                                           std::vector<std::string> &errors) {
  for (const ConSanProbeIntent &intent : coverage.observation_plan().probe_intents) {
    if (intent.kind != ConSanProbeIntentKind::RedundantAccessObservation)
      continue;
    const ConSanIntentCoverageEntry *entry = coverage.intent_entry(intent.id);
    if (entry == nullptr || entry->lowering != ConSanLoweringOutcomeKind::Pending)
      continue;
    (void)publish_sc_access_rejection(coverage, errors, intent.physical_site,
                                      ConSanLoweringOutcomeKind::PlacementRejected,
                                      "SuperCollider lowerer selected no access placement");
  }
}

struct ScOwnerContext {
  const ConSanProgramContainer *kernel = nullptr;
  KernelCfgScope scope;
  std::unique_ptr<LivenessAnalysis> liveness;
};

struct ScSiteOwnerFacts {
  const Instruction *instruction = nullptr;
  std::vector<uint64_t> descriptor_file_offsets;
  std::vector<const ScOwnerContext *> contexts;
  std::vector<const LivenessAnalysis *> liveness;
  std::optional<uint16_t> common_ordinary_vgpr_count;
  std::optional<uint16_t> common_unified_vgpr_count;
  std::optional<uint16_t> common_sgpr_count;
  std::optional<uint16_t> max_vgpr_ref;
  std::optional<uint16_t> max_sgpr_ref;
  uint32_t max_private_segment_size = 0;
  bool any_dynamic_stack = false;
  bool every_dynamic_stack = true;
  bool every_owner_can_grow_ordinary_vgprs = true;
};

class ScOwnerAnalysis {
public:
  ScOwnerAnalysis(const AmdGpuCodeObject &code_object, const ProgramInventory &inventory,
                  rj_code_arch_t arch, bool restrict_to_preflight_candidates)
      : arch_(arch), original_bytes_(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                                     code_object.image_size()),
        decoder_(Decoder::create(arch)) {
    for (const ConSanProgramContainer &kernel : inventory.kernels())
      kernels_.emplace(kernel.descriptor_file_offset, &kernel);
    if (!decoder_)
      return;

    consan_detail::ConSanCfgBuildInputs cfg =
        consan_detail::build_consan_cfg_inputs(code_object, inventory.containers());
    if (restrict_to_preflight_candidates) {
      cfg.leaders.clear();
      cfg.kernel_entries.clear();
      cfg.code_ranges.clear();
      for (const ConSanProgramContainer &kernel : inventory.kernels()) {
        if (!kernel.has_text_range || kernel.preflight_action != ConSanPreflightAction::Candidate)
          continue;
        cfg.leaders.push_back(kernel.entry_text_offset);
        cfg.kernel_entries.push_back(kernel.entry_text_offset);
        if (kernel.code_size != 0u)
          cfg.code_ranges.push_back(
              {.start_offset = kernel.entry_text_offset, .size = kernel.code_size});
      }
      std::ranges::sort(cfg.leaders);
      cfg.leaders.erase(std::ranges::unique(cfg.leaders).begin(), cfg.leaders.end());
      std::ranges::sort(cfg.kernel_entries);
      cfg.kernel_entries.erase(std::ranges::unique(cfg.kernel_entries).begin(),
                               cfg.kernel_entries.end());
    }
    blocks_ = BasicBlock::build(code_object, *decoder_, arch, cfg.leaders, cfg.code_ranges);
    block_ptrs_ = block_ptrs_for(blocks_);
    block_index_ = build_block_offset_index(blocks_);
    kernel_entries_ = std::move(cfg.kernel_entries);
    if (code_object.text_sections().size() == 1u) {
      const Section *text = code_object.text_sections().front();
      pristine_text_ =
          std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
    }
    contexts_.reserve(inventory.kernels().size());
    failed_contexts_.reserve(inventory.kernels().size());
    max_refs_.reserve(inventory.kernels().size());
  }

  [[nodiscard]] const ConSanProgramContainer *kernel(uint64_t descriptor_file_offset) const {
    const auto found = kernels_.find(descriptor_file_offset);
    return found == kernels_.end() ? nullptr : found->second;
  }

  [[nodiscard]] const ScOwnerContext *context(uint64_t descriptor_file_offset) {
    if (const auto found = contexts_.find(descriptor_file_offset); found != contexts_.end())
      return &found->second;
    const ConSanProgramContainer *owner = kernel(descriptor_file_offset);
    if (!decoder_ || owner == nullptr || !owner->has_text_range ||
        failed_contexts_.contains(descriptor_file_offset))
      return nullptr;
    auto scope = build_kernel_cfg_scope(blocks_, block_index_,
                                        KernelScopeRequest{.entry_offset = owner->entry_text_offset,
                                                           .additional_entry_offsets = {}},
                                        kernel_entries_, pristine_text_);
    if (!scope) {
      failed_contexts_.insert(descriptor_file_offset);
      return nullptr;
    }
    ScOwnerContext built{.kernel = owner, .scope = std::move(*scope), .liveness = nullptr};
    built.liveness =
        std::make_unique<LivenessAnalysis>(KernelBlockScope(built.scope.blocks),
                                           LivenessAnalysisOptions{}, built.scope.liveness_edges);
    return &contexts_.emplace(descriptor_file_offset, std::move(built)).first->second;
  }

  [[nodiscard]] std::optional<ScSiteOwnerFacts> site_facts(const ConSanProgramSite &site,
                                                           std::vector<uint64_t> owners) {
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
    ScSiteOwnerFacts facts;
    facts.descriptor_file_offsets = std::move(owners);
    facts.instruction = find_instruction_at_text_offset(KernelBlockScope(block_ptrs_),
                                                        site.physical_id.original_text_offset);
    if (!facts.instruction || facts.descriptor_file_offsets.empty())
      return std::nullopt;

    for (uint64_t owner_id : facts.descriptor_file_offsets) {
      const ScOwnerContext *owner = context(owner_id);
      const auto descriptor = read_kernel_descriptor(original_bytes_, owner_id);
      const Instruction *owner_instruction =
          owner == nullptr ? nullptr
                           : find_instruction_at_text_offset(KernelBlockScope(owner->scope.blocks),
                                                             site.physical_id.original_text_offset);
      if (owner == nullptr || owner->kernel->preflight_action == ConSanPreflightAction::Blocked ||
          owner->liveness == nullptr || owner_instruction != facts.instruction || !descriptor)
        return std::nullopt;

      auto refs = max_refs_.find(owner_id);
      if (refs == max_refs_.end())
        refs = max_refs_
                   .emplace(owner_id, max_register_refs_in_kernel(
                                          *owner->kernel, KernelBlockScope(owner->scope.blocks)))
                   .first;
      facts.contexts.push_back(owner);
      facts.liveness.push_back(owner->liveness.get());
      const uint16_t ordinary = descriptor_ordinary_vgpr_allocation_count(*descriptor, arch_);
      const uint16_t unified = descriptor_vgpr_allocation_count(*descriptor, arch_);
      const uint16_t sgprs = descriptor_sgpr_allocation_count(*descriptor, arch_);
      const auto minimum = [](std::optional<uint16_t> current, uint16_t value) {
        return current ? std::min(*current, value) : value;
      };
      facts.common_ordinary_vgpr_count = minimum(facts.common_ordinary_vgpr_count, ordinary);
      facts.common_unified_vgpr_count = minimum(facts.common_unified_vgpr_count, unified);
      facts.common_sgpr_count = minimum(facts.common_sgpr_count, sgprs);
      facts.max_private_segment_size =
          std::max(facts.max_private_segment_size, descriptor->private_segment_fixed_size);
      if (refs->second.vgpr)
        facts.max_vgpr_ref = facts.max_vgpr_ref ? std::max(*facts.max_vgpr_ref, *refs->second.vgpr)
                                                : refs->second.vgpr;
      if (refs->second.sgpr)
        facts.max_sgpr_ref = facts.max_sgpr_ref ? std::max(*facts.max_sgpr_ref, *refs->second.sgpr)
                                                : refs->second.sgpr;
      const bool dynamic = owner->kernel->uses_dynamic_stack.value_or(false);
      facts.any_dynamic_stack |= dynamic;
      facts.every_dynamic_stack &= dynamic;
      facts.every_owner_can_grow_ordinary_vgprs &= ordinary == unified;
    }
    return facts;
  }

private:
  rj_code_arch_t arch_;
  std::span<const uint8_t> original_bytes_;
  std::unique_ptr<Decoder> decoder_;
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  std::vector<BasicBlock *> block_ptrs_;
  BlockOffsetIndex block_index_;
  std::vector<uint64_t> kernel_entries_;
  std::span<const uint8_t> pristine_text_;
  std::unordered_map<uint64_t, const ConSanProgramContainer *> kernels_;
  std::unordered_map<uint64_t, ScOwnerContext> contexts_;
  std::unordered_set<uint64_t> failed_contexts_;
  std::unordered_map<uint64_t, KernelMaxRegisterRefs> max_refs_;
};

[[nodiscard]] std::optional<uint16_t> choose_sc_access_scratch_vgpr(
    const ConSanProgramSite &access, const ConSanOptions &options, const Instruction *inst,
    std::span<const LivenessAnalysis *const> owner_liveness, uint16_t search_start,
    std::optional<uint16_t> allocation_limit, uint16_t required_vgprs, uint16_t alignment,
    uint16_t max_referenced_count = 0u) {
  if (options.scratch_vgpr)
    return choose_scratch_vgpr(access, options.scratch_vgpr, inst, nullptr, search_start,
                               allocation_limit, required_vgprs);
  if (!inst || owner_liveness.empty() || required_vgprs > UINT8_MAX ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return !liveness; }) ||
      !access.lowering.form)
    return std::nullopt;

  RegisterSet live;
  for (const LivenessAnalysis *liveness : owner_liveness)
    live |= liveness->live_before(*inst);
  ConSanRegisterRequest request{
      .reg_class = RegClass::VGPR,
      .count = required_vgprs,
      .alignment = alignment,
      .search_start = search_start,
      .current_allocation_count =
          std::min(allocation_limit.value_or(kConSanOrdinaryVgprLimit), kConSanOrdinaryVgprLimit),
      .max_referenced_count = max_referenced_count,
      .architecture_limit =
          std::min(allocation_limit.value_or(kConSanOrdinaryVgprLimit), kConSanOrdinaryVgprLimit),
      .explicit_base = std::nullopt,
      .forbidden = {},
      .allow_spill = false,
      .force_spill = false,
      .allow_spill_descriptor_growth = false,
  };
  const ConSanAccessLoweringForm &form = *access.lowering.form;
  const auto forbid = [&](std::optional<uint16_t> base, uint16_t count) {
    if (base)
      request.forbidden.expand({RegClass::VGPR, *base, static_cast<uint8_t>(count)});
  };
  forbid(form.address_vgpr, form.address_vgpr_count);
  forbid(form.destination_vgpr, form.destination_register_count);
  const uint16_t data_count =
      form.second_data_vgpr ? form.element_register_count : form.data_register_count;
  forbid(form.data_vgpr, data_count);
  forbid(form.second_data_vgpr, data_count);
  const ConSanRegisterPlan plan = plan_consan_registers(request, live);
  return plan.source == ConSanRegisterAllocationSource::LivenessDead ? plan.base : std::nullopt;
}

[[nodiscard]] bool sc_sgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_width, uint16_t rhs_base,
                                          uint16_t rhs_width) {
  return static_cast<uint32_t>(lhs_base) < static_cast<uint32_t>(rhs_base) + rhs_width &&
         static_cast<uint32_t>(rhs_base) < static_cast<uint32_t>(lhs_base) + lhs_width;
}

[[nodiscard]] std::optional<uint16_t> sc_delay_source_sgpr(const ConSanOptions &options) {
  if (options.delay_nops == 0 || options.delay_mode != ConSanDelayMode::SleepVar ||
      options.delay_var_ssrc >= REGISTER_SET_ALLOCATABLE_SGPRS)
    return std::nullopt;
  return options.delay_var_ssrc;
}

[[nodiscard]] bool sc_sgpr_range_overlaps_delay_source(uint16_t base, uint16_t width,
                                                       const ConSanOptions &options) {
  const auto delay_source = sc_delay_source_sgpr(options);
  return delay_source && sc_sgpr_ranges_overlap(base, width, *delay_source, 1u);
}

[[nodiscard]] uint16_t sc_vcc_save_sgpr_width(const ConSanTargetProfile &target) {
  return consan_arch_is_cdna3_or_cdna4(target.arch) ? 2u : 1u;
}

[[nodiscard]] std::optional<uint16_t>
choose_sc_borrowed_scalar_range(uint16_t allocated_sgpr_count, uint16_t width,
                                const ConSanOptions &options,
                                bool preserve_dynamic_stack_abi = false) {
  const uint16_t limit = std::min<uint16_t>(allocated_sgpr_count, REGISTER_SET_ALLOCATABLE_SGPRS);
  for (uint16_t base = 0; static_cast<uint32_t>(base) + width <= limit; base += width)
    if (!sc_sgpr_range_overlaps_delay_source(base, width, options) &&
        !(preserve_dynamic_stack_abi &&
          sc_sgpr_ranges_overlap(base, width, kDynamicStackTopSgpr, 2u)))
      return base;
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> choose_sc_dead_vcc_save_sgpr(
    const Instruction *inst, std::span<const LivenessAnalysis *const> owner_liveness,
    uint16_t allocated_sgpr_count, std::optional<uint16_t> min_preferred_sgpr,
    uint16_t selection_limit, uint16_t width, bool prefer_allocated_hole,
    const ConSanOptions &options, bool preserve_dynamic_stack_abi = false) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;
  const uint16_t limit = std::min<uint16_t>(selection_limit, REGISTER_SET_ALLOCATABLE_SGPRS);
  const auto find_free = [&](uint16_t start, uint16_t end) -> std::optional<uint16_t> {
    if (width == 2u && (start & 1u) != 0u)
      ++start;
    for (uint16_t base = start; static_cast<uint32_t>(base) + width <= end;
         base = static_cast<uint16_t>(base + width)) {
      if (sc_sgpr_range_overlaps_delay_source(base, width, options) ||
          (preserve_dynamic_stack_abi &&
           sc_sgpr_ranges_overlap(base, width, kDynamicStackTopSgpr, 2u)))
        continue;
      const RegisterRef ref{RegClass::SGPR, base, static_cast<uint8_t>(width)};
      if (std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
            return !liveness->any_live_before(*inst, ref);
          }))
        return base;
    }
    return std::nullopt;
  };
  const uint16_t allocated_limit = std::min(allocated_sgpr_count, limit);
  if (prefer_allocated_hole) {
    // CDNA SGPR allocation participates in occupancy. Reuse an all-owner dead
    // hole inside the current allocation before growing every descriptor.
    if (auto existing = find_free(0, allocated_limit))
      return existing;
    const uint16_t fresh_start =
        std::max(allocated_limit, min_preferred_sgpr.value_or(allocated_limit));
    return find_free(fresh_start, limit);
  }

  // RDNA allocation is effectively fixed, so retain the high-first policy.
  // Besides avoiding churn in established physical workloads, this keeps
  // instrumentation temporaries away from the guest's referenced scalar
  // range whenever a fresh register is available.
  if (min_preferred_sgpr) {
    if (auto preferred = find_free(*min_preferred_sgpr, limit))
      return preferred;
  }
  return find_free(0, limit);
}

[[nodiscard]] bool is_d16_lds_load(const ConSanProgramSite &access) {
  return access.lowering.form && access.lowering.form->destination_preserves_unwritten_bits;
}

[[nodiscard]] bool is_subdword_lds_store(const ConSanProgramSite &access) {
  return access.lowering.form && access.lowering.form->access_kind == ConSanLdsAccessKind::Write &&
         access.lowering.form->element_width_bits < 32u;
}

[[nodiscard]] bool is_high_d16_lds_store(const ConSanProgramSite &access) {
  return access.lowering.form && access.lowering.form->register_value_placement ==
                                     ConSanAccessRegisterValuePlacement::High16;
}

struct SplitTwoAddressLdsExpansionPlan {
  ConSanScTwoAddressLdsByteOffsets offsets;
  uint16_t element_dwords = 0;

  [[nodiscard]] uint16_t large_offset_count() const {
    return static_cast<uint16_t>(offsets.first > UINT16_MAX) +
           static_cast<uint16_t>(offsets.second > UINT16_MAX);
  }
};

[[nodiscard]] std::optional<SplitTwoAddressLdsExpansionPlan>
split_two_address_lds_expansion_plan(std::span<const uint8_t> original_bytes,
                                     const ConSanProgramSite &access,
                                     const ConSanTargetProfile &target) {
  if (!target.requires_split_two_address_lds_relocation || !access.lowering.form ||
      access.lowering.form->kind != ConSanAccessLoweringFormKind::NativeTwoRange)
    return std::nullopt;
  std::array<uint32_t, 1> words{};
  if (!read_words_at(original_bytes, access.decoded_file_offset(), std::span<uint32_t>(words)))
    return std::nullopt;
  const auto offsets = consan_sc_two_address_lds_byte_offsets(*access.lowering.form, words.front());
  if (!offsets)
    return std::nullopt;
  return SplitTwoAddressLdsExpansionPlan{
      .offsets = *offsets,
      .element_dwords = static_cast<uint16_t>(access.lowering.form->element_width_bits / 32u),
  };
}

[[nodiscard]] uint16_t lds_check_trap_extra_scratch_vgprs(
    const ConSanProgramSite &site,
    const std::optional<SplitTwoAddressLdsExpansionPlan> &expansion) {
  return static_cast<uint16_t>(
      (is_subdword_lds_store(site) ||
               (site.lowering.form && site.lowering.form->destination_accvgpr)
           ? 1u
           : 0u) +
      (expansion && expansion->large_offset_count() != 0u ? 1u : 0u));
}

[[nodiscard]] std::optional<uint32_t>
build_v_mov_b32_e32_vgpr_word(uint16_t vdst, uint16_t src_vgpr, rj_code_arch_t arch) {
  if (vdst > 255 || src_vgpr > 255)
    return std::nullopt;
  return build_v_mov_b32_e32(vdst, vector_source_vgpr(src_vgpr), arch);
}

[[nodiscard]] uint32_t report_action_scratch_vgprs(const ConSanOptions &options) {
  return options.report_buffer_address ? 3u : 0u;
}

[[nodiscard]] bool lds_report_action_reuses_readback(const ConSanOptions &options,
                                                     uint16_t readback_vgprs) {
  // The automatic SuperCollider report is an idempotent one-word marker. Once
  // a comparison has found a mismatch, its LDS readback values need not
  // survive the marker store: clobbering them can only cause another marker
  // store in a later chunk. Reuse a wide readback tuple instead of requiring
  // three additional ordinary VGPRs at an AccVGPR boundary.
  return options.report_buffer_address.has_value() && readback_vgprs >= 3u;
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_mismatch_action_words(const ConSanOptions &options, const ConSanTargetProfile &target,
                            uint16_t report_scratch_vgpr, std::vector<std::string> &errors,
                            std::string_view context) {
  const rj_code_arch_t arch = target.arch;
  if (!options.report_buffer_address) {
    auto trap = instrumentation::build_s_trap(0, arch);
    if (!trap) {
      errors.emplace_back(std::string(context) + " could not encode s_trap");
      return std::nullopt;
    }
    return std::vector<uint32_t>{*trap};
  }

  if (static_cast<uint32_t>(report_scratch_vgpr) + report_action_scratch_vgprs(options) > 256u) {
    errors.emplace_back(std::string(context) +
                        " report-buffer action has insufficient scratch VGPR headroom");
    return std::nullopt;
  }
  if (target.requires_even_vgpr_tuples && report_scratch_vgpr % 2u != 0u) {
    errors.emplace_back(std::string(context) +
                        " report-buffer action selected a misaligned address tuple");
    return std::nullopt;
  }

  const uint64_t address = *options.report_buffer_address;
  const uint16_t address_lo_vgpr = report_scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(report_scratch_vgpr + 1u);
  const uint16_t marker_vgpr = static_cast<uint16_t>(report_scratch_vgpr + 2u);

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto mov_marker =
      instrumentation::build_v_mov_b32_literal(marker_vgpr, options.report_marker, arch);
  const auto store = instrumentation::build_flat_store_b32(address_lo_vgpr, marker_vgpr, arch);
  if (!mov_address_lo || !mov_address_hi || !mov_marker || !store) {
    errors.emplace_back(std::string(context) + " could not encode report-buffer action");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  words.reserve(12);
  words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
  words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
  words.insert(words.end(), mov_marker->begin(), mov_marker->end());
  words.insert(words.end(), store->begin(), store->end());
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_lds_check_trap_words(
    std::span<const uint8_t> original_bytes, const ConSanProgramSite &site,
    const ConSanTargetProfile &target, const ConSanOptions &options, uint16_t scratch_vgpr,
    uint16_t vcc_save_sgpr, uint16_t selectable_vgpr_bank_mode,
    std::optional<uint32_t> &guest_instruction_word_offset, std::vector<std::string> &errors) {
  const rj_code_arch_t arch = target.arch;
  guest_instruction_word_offset.reset();
  auto required_vgprs = access_dword_count(site);
  if (!required_vgprs) {
    errors.emplace_back("ConSan LDS check/trap proof selected unsupported width");
    return std::nullopt;
  }
  const ConSanAccessLoweringForm &form = *site.lowering.form;

  std::array<uint32_t, 2> original_access{};
  if (!read_words_at(original_bytes, site.decoded_file_offset(),
                     std::span<uint32_t>(original_access))) {
    errors.emplace_back("ConSan LDS check/trap proof could not read LDS access");
    return std::nullopt;
  }

  std::array<uint32_t, 2> duplicate_load = original_access;
  const bool preserve_address = lds_load_clobbers_address(site);
  const auto split_two_address = split_two_address_lds_expansion_plan(original_bytes, site, target);
  const uint16_t compare_scratch_vgpr =
      static_cast<uint16_t>(scratch_vgpr + static_cast<uint16_t>(*required_vgprs));
  const uint16_t core_scratch_end = static_cast<uint16_t>(
      compare_scratch_vgpr + lds_check_trap_extra_scratch_vgprs(site, split_two_address));
  const bool reuse_readback_for_report =
      lds_report_action_reuses_readback(options, static_cast<uint16_t>(*required_vgprs));
  uint16_t report_scratch_vgpr = reuse_readback_for_report ? scratch_vgpr : core_scratch_end;
  if (!reuse_readback_for_report && options.report_buffer_address &&
      target.requires_even_vgpr_tuples && report_scratch_vgpr % 2u != 0)
    ++report_scratch_vgpr;
  const uint16_t report_scratch_end = static_cast<uint16_t>(
      report_scratch_vgpr + static_cast<uint16_t>(report_action_scratch_vgprs(options)));
  const uint16_t saved_address_vgpr = std::max(core_scratch_end, report_scratch_end);
  if (site.kind == ConSanLdsAccessKind::Read) {
    if (form.destination_accvgpr &&
        target.accumulator_model == ConSanAccumulatorModel::DescriptorPartitioned)
      duplicate_load[0] &= ~(uint32_t{1} << 25u);
    duplicate_load[1] =
        (duplicate_load[1] & 0x00FFFFFFu) | (static_cast<uint32_t>(scratch_vgpr) << 24u);
    if (preserve_address)
      duplicate_load[1] = (duplicate_load[1] & 0xFFFFFF00u) | saved_address_vgpr;
  } else if (site.kind == ConSanLdsAccessKind::Write) {
    auto word0 = consan_sc_build_ds_load_word0(form, original_access[0], arch);
    if (!word0) {
      errors.emplace_back("ConSan LDS check/trap proof could not encode store readback");
      return std::nullopt;
    }
    duplicate_load[0] = *word0;
    duplicate_load[1] = consan_sc_build_ds_load_word1(*form.address_vgpr, scratch_vgpr);
  } else {
    errors.emplace_back("ConSan LDS check/trap proof selected unsupported site");
    return std::nullopt;
  }

  if (!access_scratch_tuple_base_is_valid(site, scratch_vgpr)) {
    errors.emplace_back("ConSan LDS check/trap proof selected a misaligned tuple scratch base");
    return std::nullopt;
  }

  auto wait_dscnt = instrumentation::build_s_wait_lds0(arch);
  auto first_compare_vgpr =
      form.destination_accvgpr &&
              target.accumulator_model == ConSanAccumulatorModel::DescriptorPartitioned
          ? std::optional<uint16_t>(compare_scratch_vgpr)
          : check_trap_compare_vgpr(site, 0u, arch, selectable_vgpr_bank_mode);
  if (!first_compare_vgpr) {
    errors.emplace_back("ConSan LDS check/trap proof selected site without compare VGPR");
    return std::nullopt;
  }
  auto mismatch_action = build_mismatch_action_words(options, target, report_scratch_vgpr, errors,
                                                     "ConSan LDS check/trap proof");
  if (!mismatch_action)
    return std::nullopt;
  if (mismatch_action->size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan LDS check/trap proof mismatch action is too large");
    return std::nullopt;
  }
  auto skip_action =
      instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(mismatch_action->size()), arch);
  constexpr uint16_t kAmdGpuVccLo = 106;
  const bool legacy_cdna = consan_arch_is_cdna3_or_cdna4(target.arch);
  const std::optional<uint32_t> save_vcc =
      legacy_cdna ? instrumentation::build_s_mov_b64(vcc_save_sgpr, kAmdGpuVccLo, arch)
                  : std::optional<uint32_t>(build_s_mov_b32(vcc_save_sgpr, kAmdGpuVccLo, arch));
  const std::optional<uint32_t> restore_vcc =
      legacy_cdna ? instrumentation::build_s_mov_b64(kAmdGpuVccLo, vcc_save_sgpr, arch)
                  : std::optional<uint32_t>(build_s_mov_b32(kAmdGpuVccLo, vcc_save_sgpr, arch));
  if (!wait_dscnt || !skip_action || !save_vcc || !restore_vcc) {
    errors.emplace_back("ConSan LDS check/trap proof could not encode sequence");
    return std::nullopt;
  }

  std::optional<std::vector<uint32_t>> masked_store_value;
  if (is_subdword_lds_store(site)) {
    std::vector<uint32_t> narrowed;
    uint16_t narrow_source = *first_compare_vgpr;
    if (is_high_d16_lds_store(site)) {
      const auto shift = instrumentation::build_v_lshrrev_b32(
          compare_scratch_vgpr, scalar_positive_inline_u32(16u), *first_compare_vgpr, arch);
      if (!shift) {
        errors.emplace_back("ConSan LDS check/trap proof could not select high store subword");
        return std::nullopt;
      }
      narrowed.push_back(*shift);
      narrow_source = compare_scratch_vgpr;
    }
    const auto mask = instrumentation::build_v_and_b32_literal(
        compare_scratch_vgpr, (uint32_t{1} << form.element_width_bits) - 1u, narrow_source, arch);
    if (!mask) {
      errors.emplace_back("ConSan LDS check/trap proof could not mask sub-dword store value");
      return std::nullopt;
    }
    narrowed.insert(narrowed.end(), mask->begin(), mask->end());
    masked_store_value = std::move(narrowed);
  }

  std::optional<uint32_t> d16_seed_scratch;
  if (is_d16_lds_load(site)) {
    d16_seed_scratch = build_v_mov_b32_e32_vgpr_word(scratch_vgpr, *first_compare_vgpr, arch);
    if (!d16_seed_scratch) {
      errors.emplace_back("ConSan LDS check/trap proof could not encode d16 scratch "
                          "seed move");
      return std::nullopt;
    }
  }

  std::vector<uint32_t> words;
  if (preserve_address) {
    auto save_address = build_v_mov_b32_e32_vgpr_word(saved_address_vgpr, *form.address_vgpr, arch);
    if (!save_address) {
      errors.emplace_back("ConSan LDS check/trap proof could not preserve a clobbered address");
      return std::nullopt;
    }
    words.push_back(*save_address);
  }
  if (split_two_address) {
    const uint16_t replay_address_vgpr = preserve_address ? saved_address_vgpr : *form.address_vgpr;
    const std::optional<uint16_t> first_data_vgpr =
        site.kind == ConSanLdsAccessKind::Read ? form.destination_vgpr : form.data_vgpr;
    const std::optional<uint16_t> second_data_vgpr =
        site.kind == ConSanLdsAccessKind::Read && first_data_vgpr
            ? std::optional<uint16_t>(
                  static_cast<uint16_t>(*first_data_vgpr + split_two_address->element_dwords))
            : form.second_data_vgpr;
    const auto split =
        first_data_vgpr && second_data_vgpr
            ? consan_sc_build_split_single_address_lds_pair(
                  split_two_address->offsets, split_two_address->element_dwords,
                  replay_address_vgpr, *first_data_vgpr, *second_data_vgpr, compare_scratch_vgpr,
                  site.kind == ConSanLdsAccessKind::Read, target)
            : std::nullopt;
    if (!split) {
      errors.emplace_back("ConSan LDS check/trap proof could not split CDNA5 two-address access");
      return std::nullopt;
    }
    words.insert(words.end(), split->begin(), split->end());
  } else {
    guest_instruction_word_offset = static_cast<uint32_t>(words.size());
    words.push_back(original_access[0]);
    words.push_back(original_access[1]);
  }
  if (d16_seed_scratch) {
    words.push_back(*wait_dscnt);
    words.push_back(*d16_seed_scratch);
  }
  if (!consan_sc_append_delay_words(words, arch, options, errors, "ConSan LDS check/trap proof"))
    return std::nullopt;
  if (split_two_address) {
    const uint16_t replay_address_vgpr = preserve_address ? saved_address_vgpr : *form.address_vgpr;
    const auto split = consan_sc_build_split_single_address_lds_pair(
        split_two_address->offsets, split_two_address->element_dwords, replay_address_vgpr,
        scratch_vgpr, static_cast<uint16_t>(scratch_vgpr + split_two_address->element_dwords),
        compare_scratch_vgpr, true, target);
    if (!split) {
      errors.emplace_back("ConSan LDS check/trap proof could not split CDNA5 two-address readback");
      return std::nullopt;
    }
    words.insert(words.end(), split->begin(), split->end());
  } else {
    words.push_back(duplicate_load[0]);
    words.push_back(duplicate_load[1]);
  }
  words.push_back(*wait_dscnt);
  if (masked_store_value)
    words.insert(words.end(), masked_store_value->begin(), masked_store_value->end());
  words.push_back(*save_vcc);
  for (uint16_t i = 0; i < *required_vgprs; ++i) {
    auto compare_vgpr = check_trap_compare_vgpr(site, i, arch, selectable_vgpr_bank_mode);
    if (form.destination_accvgpr && legacy_cdna) {
      const auto acc_read = consan_sc_build_cdna_accvgpr_read_b32(
          compare_scratch_vgpr, static_cast<uint16_t>(*form.destination_accvgpr + i), target);
      if (!acc_read) {
        errors.emplace_back("ConSan LDS check/trap proof could not encode accumulator read");
        return std::nullopt;
      }
      words.insert(words.end(), acc_read->begin(), acc_read->end());
      compare_vgpr = compare_scratch_vgpr;
    }
    const uint16_t compare_mode = target.has_selectable_vgpr_bank && compare_vgpr
                                      ? static_cast<uint16_t>(*compare_vgpr / 256u)
                                      : 0u;
    if (compare_mode != 0u) {
      auto select_compare_bank = instrumentation::build_s_set_vgpr_msb_transition(
          0u, static_cast<uint8_t>(compare_mode), arch);
      if (!select_compare_bank) {
        errors.emplace_back("ConSan LDS check/trap proof could not select compare VGPR bank");
        return std::nullopt;
      }
      words.push_back(*select_compare_bank);
    }
    const std::optional<uint16_t> effective_compare_vgpr =
        is_subdword_lds_store(site) ? std::optional<uint16_t>(compare_scratch_vgpr) : compare_vgpr;
    auto chunk_cmp_ne =
        effective_compare_vgpr
            ? consan_sc_build_v_cmp_ne_u32(static_cast<uint16_t>(*effective_compare_vgpr & 0xFFu),
                                           static_cast<uint16_t>(scratch_vgpr + i), arch)
            : std::nullopt;
    if (!chunk_cmp_ne) {
      errors.emplace_back("ConSan LDS check/trap proof could not encode chunk compare");
      return std::nullopt;
    }
    words.push_back(*chunk_cmp_ne);
    if (compare_mode != 0u) {
      auto restore_low_bank = instrumentation::build_s_set_vgpr_msb_transition(
          static_cast<uint8_t>(compare_mode), 0u, arch);
      if (!restore_low_bank) {
        errors.emplace_back("ConSan LDS check/trap proof could not restore compare VGPR bank");
        return std::nullopt;
      }
      words.push_back(*restore_low_bank);
    }
    words.push_back(*skip_action);
    words.insert(words.end(), mismatch_action->begin(), mismatch_action->end());
  }
  words.push_back(*restore_vcc);
  return words;
}
