[[nodiscard]] std::unordered_set<uint64_t>
supercollider_admitted_access_file_offsets(const ProgramInventory &inventory,
                                           const CoverageLedger &coverage, AccessOrigin origin) {
  std::unordered_set<uint64_t> offsets;
  for (const ProgramSite &access : inventory.access_sites())
    if (access.origin == origin &&
        !coverage
             .pending_intent_ids(ProbeIntentKind::RedundantAccessObservation, access.physical_id)
             .empty())
      offsets.insert(access.decoded_file_offset());
  return offsets;
}

[[nodiscard]] bool publish_supercollider_access_rejection(CoverageLedger &coverage,
                                                          std::vector<std::string> &errors,
                                                          const PhysicalSiteId &site,
                                                          LoweringOutcomeKind outcome,
                                                          std::string detail) {
  std::vector<ProbeIntentId> intent_ids =
      coverage.pending_intent_ids(ProbeIntentKind::RedundantAccessObservation, site);
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
void publish_unplaced_supercollider_access_rejections(CoverageLedger &coverage,
                                                      std::vector<std::string> &errors) {
  for (const ProbeIntent &intent : coverage.observation_plan().probe_intents) {
    if (intent.kind != ProbeIntentKind::RedundantAccessObservation)
      continue;
    const IntentCoverageEntry *entry = coverage.intent_entry(intent.id);
    if (entry == nullptr || entry->lowering != LoweringOutcomeKind::Pending)
      continue;
    (void)publish_supercollider_access_rejection(
        coverage, errors, intent.physical_site, LoweringOutcomeKind::PlacementRejected,
        "SuperCollider lowerer selected no access placement");
  }
}

struct SuperColliderOwnerContext {
  const ProgramContainer *kernel = nullptr;
  KernelCfgScope scope;
  std::unique_ptr<LivenessAnalysis> liveness;
  KernelMaxRegisterRefs max_refs;
};

struct SuperColliderSiteOwnerFacts {
  const Instruction *instruction = nullptr;
  std::vector<uint64_t> descriptor_file_offsets;
  std::vector<const SuperColliderOwnerContext *> contexts;
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

enum class SuperColliderOwnerCfgRestriction {
  None,
  PreflightCandidates,
  AllowlistedKernels,
};

class SuperColliderOwnerAnalysis {
public:
  SuperColliderOwnerAnalysis(const AmdGpuCodeObject &code_object, const ProgramInventory &inventory,
                             rj_code_arch_t arch, SuperColliderOwnerCfgRestriction restriction,
                             std::span<const std::string> kernel_name_allowlist,
                             std::span<const uint64_t> liveness_query_offsets)
      : inventory_(inventory), arch_(arch),
        original_bytes_(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                        code_object.image_size()),
        decoder_(Decoder::create(arch)) {
    liveness_query_offsets_.insert(liveness_query_offsets.begin(), liveness_query_offsets.end());
    if (!decoder_)
      return;

    detail::CfgBuildInputs cfg = detail::build_cfg_inputs(code_object, inventory.containers());
    if (restriction != SuperColliderOwnerCfgRestriction::None) {
      cfg.leaders.clear();
      cfg.kernel_entries.clear();
      cfg.code_ranges.clear();
      for (const ProgramContainer &kernel : inventory.kernels()) {
        if (!kernel.has_text_range)
          continue;
        if (restriction == SuperColliderOwnerCfgRestriction::PreflightCandidates &&
            kernel.preflight_action != PreflightAction::Candidate)
          continue;
        if (!kernel_name_allowlist.empty() &&
            !std::ranges::any_of(kernel_name_allowlist, [&](std::string_view allowed) {
              return kernel_name_matches(allowed, kernel.name);
            }))
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
    contexts_.resize(inventory.containers().size());
    failed_contexts_.resize(inventory.containers().size());
  }

  [[nodiscard]] const ProgramContainer *kernel(ProgramContainerId id) const {
    const ProgramContainer *record = inventory_.container(id);
    return record != nullptr && record->is_kernel() ? record : nullptr;
  }

  [[nodiscard]] const SuperColliderOwnerContext *context(ProgramContainerId id) {
    if (!id.valid() || id.ordinal >= contexts_.size())
      return nullptr;
    if (contexts_[id.ordinal])
      return &*contexts_[id.ordinal];
    const ProgramContainer *owner = kernel(id);
    if (!decoder_ || owner == nullptr || !owner->has_text_range || failed_contexts_[id.ordinal])
      return nullptr;
    auto scope = build_kernel_cfg_scope(blocks_, block_index_,
                                        KernelScopeRequest{.entry_offset = owner->entry_text_offset,
                                                           .additional_entry_offsets = {}},
                                        kernel_entries_, pristine_text_);
    if (!scope) {
      failed_contexts_[id.ordinal] = true;
      return nullptr;
    }
    SuperColliderOwnerContext &built = contexts_[id.ordinal].emplace(SuperColliderOwnerContext{
        .kernel = owner, .scope = std::move(*scope), .liveness = nullptr, .max_refs = {}});
    std::vector<const Instruction *> liveness_queries;
    for (BasicBlock *block : built.scope.blocks) {
      if (block == nullptr)
        continue;
      for (const Instruction &instruction : block->instructions())
        if (liveness_query_offsets_.contains(instruction.src_loc()))
          liveness_queries.push_back(&instruction);
    }
    LivenessAnalysisOptions liveness_options;
    liveness_options.restrict_live_before_to_instructions = true;
    liveness_options.live_before_instructions = liveness_queries;
    built.liveness = std::make_unique<LivenessAnalysis>(
        KernelBlockScope(built.scope.blocks), liveness_options, built.scope.liveness_edges);
    built.max_refs = max_register_refs_in_kernel(*owner, KernelBlockScope(built.scope.blocks));
    return &built;
  }

  [[nodiscard]] std::optional<SuperColliderSiteOwnerFacts>
  site_facts(const ProgramSite &site, std::vector<ProgramContainerId> owners) {
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
    SuperColliderSiteOwnerFacts facts;
    facts.instruction = find_instruction_at_text_offset(KernelBlockScope(block_ptrs_),
                                                        site.physical_id.original_text_offset);
    if (!facts.instruction || owners.empty())
      return std::nullopt;

    for (ProgramContainerId owner_id : owners) {
      const SuperColliderOwnerContext *owner = context(owner_id);
      const Instruction *owner_instruction =
          owner == nullptr ? nullptr
                           : find_instruction_at_text_offset(KernelBlockScope(owner->scope.blocks),
                                                             site.physical_id.original_text_offset);
      if (owner == nullptr || owner->kernel->preflight_action == PreflightAction::Blocked ||
          owner->liveness == nullptr || owner_instruction != facts.instruction)
        return std::nullopt;
      const auto descriptor =
          read_kernel_descriptor(original_bytes_, owner->kernel->descriptor_file_offset);
      if (!descriptor)
        return std::nullopt;

      facts.descriptor_file_offsets.push_back(owner->kernel->descriptor_file_offset);
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
      if (owner->max_refs.vgpr)
        facts.max_vgpr_ref = facts.max_vgpr_ref
                                 ? std::max(*facts.max_vgpr_ref, *owner->max_refs.vgpr)
                                 : owner->max_refs.vgpr;
      if (owner->max_refs.sgpr)
        facts.max_sgpr_ref = facts.max_sgpr_ref
                                 ? std::max(*facts.max_sgpr_ref, *owner->max_refs.sgpr)
                                 : owner->max_refs.sgpr;
      const bool dynamic = owner->kernel->uses_dynamic_stack.value_or(false);
      facts.any_dynamic_stack |= dynamic;
      facts.every_dynamic_stack &= dynamic;
      facts.every_owner_can_grow_ordinary_vgprs &= ordinary == unified;
    }
    return facts;
  }

private:
  const ProgramInventory &inventory_;
  rj_code_arch_t arch_;
  std::span<const uint8_t> original_bytes_;
  std::unique_ptr<Decoder> decoder_;
  std::vector<std::unique_ptr<BasicBlock>> blocks_;
  std::vector<BasicBlock *> block_ptrs_;
  BlockOffsetIndex block_index_;
  std::vector<uint64_t> kernel_entries_;
  std::span<const uint8_t> pristine_text_;
  std::vector<std::optional<SuperColliderOwnerContext>> contexts_;
  std::vector<bool> failed_contexts_;
  std::unordered_set<uint64_t> liveness_query_offsets_;
};

[[nodiscard]] std::optional<uint16_t> choose_supercollider_access_scratch_vgpr(
    const ProgramSite &access, const Options &options, const Instruction *inst,
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
  RegisterRequest request{
      .reg_class = RegClass::VGPR,
      .count = required_vgprs,
      .alignment = alignment,
      .search_start = search_start,
      .current_allocation_count =
          std::min(allocation_limit.value_or(kOrdinaryVgprLimit), kOrdinaryVgprLimit),
      .max_referenced_count = max_referenced_count,
      .architecture_limit =
          std::min(allocation_limit.value_or(kOrdinaryVgprLimit), kOrdinaryVgprLimit),
      .explicit_base = std::nullopt,
      .forbidden = {},
      .allow_spill = false,
      .force_spill = false,
      .allow_spill_descriptor_growth = false,
  };
  const AccessLoweringForm &form = *access.lowering.form;
  const auto forbid = [&](std::optional<uint16_t> base, uint16_t count) {
    if (base)
      request.forbidden.expand({RegClass::VGPR, *base, static_cast<uint8_t>(count)});
  };
  forbid(form.address_vgpr, form.address_vgpr_count);
  forbid(form.direct_memory_address_vgpr, form.direct_memory_address_vgpr_count);
  forbid(form.destination_vgpr, form.destination_register_count);
  const uint16_t data_count =
      form.second_data_vgpr ? form.element_register_count : form.data_register_count;
  forbid(form.data_vgpr, data_count);
  forbid(form.second_data_vgpr, data_count);
  const RegisterPlan plan = plan_registers(request, live);
  return plan.source == RegisterAllocationSource::LivenessDead ? plan.base : std::nullopt;
}

[[nodiscard]] bool supercollider_sgpr_ranges_overlap(uint16_t lhs_base, uint16_t lhs_width,
                                                     uint16_t rhs_base, uint16_t rhs_width) {
  return static_cast<uint32_t>(lhs_base) < static_cast<uint32_t>(rhs_base) + rhs_width &&
         static_cast<uint32_t>(rhs_base) < static_cast<uint32_t>(lhs_base) + lhs_width;
}

[[nodiscard]] std::optional<uint16_t> supercollider_delay_source_sgpr(const Options &options) {
  if (options.supercollider_delay_nops == 0 ||
      options.supercollider_delay_mode != SuperColliderDelayMode::SleepVar ||
      options.supercollider_delay_var_ssrc >= REGISTER_SET_ALLOCATABLE_SGPRS)
    return std::nullopt;
  return options.supercollider_delay_var_ssrc;
}

[[nodiscard]] bool supercollider_sgpr_range_overlaps_delay_source(uint16_t base, uint16_t width,
                                                                  const Options &options) {
  const auto delay_source = supercollider_delay_source_sgpr(options);
  return delay_source && supercollider_sgpr_ranges_overlap(base, width, *delay_source, 1u);
}

[[nodiscard]] uint16_t supercollider_vcc_save_sgpr_width(const TargetProfile &target) {
  return target.scalar_placement_model == ScalarPlacementModel::DescriptorPartitioned ? 2u : 1u;
}

[[nodiscard]] std::optional<uint16_t>
choose_supercollider_borrowed_scalar_range(uint16_t allocated_sgpr_count, uint16_t width,
                                           const Options &options,
                                           bool preserve_dynamic_stack_abi = false) {
  const uint16_t limit = std::min<uint16_t>(allocated_sgpr_count, REGISTER_SET_ALLOCATABLE_SGPRS);
  for (uint16_t base = 0; static_cast<uint32_t>(base) + width <= limit; base += width)
    if (!supercollider_sgpr_range_overlaps_delay_source(base, width, options) &&
        !(preserve_dynamic_stack_abi &&
          supercollider_sgpr_ranges_overlap(base, width, kDynamicStackTopSgpr, 2u)))
      return base;
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> choose_supercollider_dead_vcc_save_sgpr(
    const Instruction *inst, std::span<const LivenessAnalysis *const> owner_liveness,
    uint16_t allocated_sgpr_count, std::optional<uint16_t> min_preferred_sgpr,
    uint16_t selection_limit, uint16_t width, bool prefer_allocated_hole, const Options &options,
    bool preserve_dynamic_stack_abi = false) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;
  const uint16_t limit = std::min<uint16_t>(selection_limit, REGISTER_SET_ALLOCATABLE_SGPRS);
  const auto find_free = [&](uint16_t start, uint16_t end) -> std::optional<uint16_t> {
    if (width == 2u && (start & 1u) != 0u)
      ++start;
    for (uint16_t base = start; static_cast<uint32_t>(base) + width <= end;
         base = static_cast<uint16_t>(base + width)) {
      if (supercollider_sgpr_range_overlaps_delay_source(base, width, options) ||
          (preserve_dynamic_stack_abi &&
           supercollider_sgpr_ranges_overlap(base, width, kDynamicStackTopSgpr, 2u)))
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

[[nodiscard]] bool is_d16_lds_load(const ProgramSite &access) {
  return access.lowering.form && access.lowering.form->destination_preserves_unwritten_bits;
}

[[nodiscard]] bool is_subdword_lds_store(const ProgramSite &access) {
  return access.lowering.form && access.lowering.form->access_kind == LdsAccessKind::Write &&
         access.lowering.form->element_width_bits < 32u;
}

[[nodiscard]] bool is_high_d16_lds_store(const ProgramSite &access) {
  return access.lowering.form &&
         access.lowering.form->register_value_placement == AccessRegisterValuePlacement::High16;
}

struct SplitTwoAddressLdsExpansionPlan {
  SuperColliderTwoAddressLdsByteOffsets offsets;
  uint16_t element_dwords = 0;

  [[nodiscard]] uint16_t large_offset_count() const {
    return static_cast<uint16_t>(offsets.first > UINT16_MAX) +
           static_cast<uint16_t>(offsets.second > UINT16_MAX);
  }
};

[[nodiscard]] std::optional<SplitTwoAddressLdsExpansionPlan>
split_two_address_lds_expansion_plan(std::span<const uint8_t> original_bytes,
                                     const ProgramSite &access, const TargetProfile &target) {
  if (!target.requires_split_two_address_lds_relocation || !access.lowering.form ||
      access.lowering.form->kind != AccessLoweringFormKind::NativeTwoRange)
    return std::nullopt;
  std::array<uint32_t, 1> words{};
  if (!read_words_at(original_bytes, access.decoded_file_offset(), std::span<uint32_t>(words)))
    return std::nullopt;
  const auto offsets =
      supercollider_two_address_lds_byte_offsets(*access.lowering.form, words.front());
  if (!offsets)
    return std::nullopt;
  return SplitTwoAddressLdsExpansionPlan{
      .offsets = *offsets,
      .element_dwords = static_cast<uint16_t>(access.lowering.form->element_width_bits / 32u),
  };
}

[[nodiscard]] uint16_t lds_check_trap_extra_scratch_vgprs(
    const ProgramSite &site, const std::optional<SplitTwoAddressLdsExpansionPlan> &expansion) {
  return static_cast<uint16_t>(
      ((site.lowering.form && site.lowering.form->destination_accvgpr) ? 1u : 0u) +
      (expansion && expansion->large_offset_count() != 0u ? 1u : 0u));
}

[[nodiscard]] std::optional<uint32_t>
build_v_mov_b32_e32_vgpr_word(uint16_t vdst, uint16_t src_vgpr, rj_code_arch_t arch) {
  if (vdst > 255 || src_vgpr > 255)
    return std::nullopt;
  return build_v_mov_b32_e32(vdst, vector_source_vgpr(src_vgpr), arch);
}

[[nodiscard]] uint32_t report_action_scratch_vgprs(const Options &options) {
  return options.supercollider_report_buffer_address ? 3u : 0u;
}

[[nodiscard]] bool
lds_report_action_reuses_readback(const Options &options, const TargetProfile &target,
                                  uint16_t readback_vgprs,
                                  std::optional<uint16_t> scratch_vgpr = std::nullopt) {
  // The automatic SuperCollider report is an idempotent one-word marker. Once
  // a comparison has found a mismatch, its LDS readback values need not
  // survive the marker store: clobbering them can only cause another marker
  // store in a later chunk. Reserve a three-register window beginning at the
  // readback rather than placing a disjoint report tuple after it. Automatic
  // placement can align that window; an explicitly odd scratch base retains
  // the disjoint fallback on targets requiring even address tuples.
  return options.supercollider_report_buffer_address.has_value() && readback_vgprs != 0u &&
         (!target.requires_even_vgpr_tuples || !scratch_vgpr || *scratch_vgpr % 2u == 0u);
}

[[nodiscard]] std::optional<std::vector<uint32_t>>
build_mismatch_action_words(const Options &options, const TargetProfile &target,
                            uint16_t report_scratch_vgpr, std::vector<std::string> &errors,
                            std::string_view context) {
  const rj_code_arch_t arch = target.arch;
  if (!options.supercollider_report_buffer_address) {
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

  const uint64_t address = *options.supercollider_report_buffer_address;
  const uint16_t address_lo_vgpr = report_scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(report_scratch_vgpr + 1u);
  const uint16_t marker_vgpr = static_cast<uint16_t>(report_scratch_vgpr + 2u);

  const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
      address_lo_vgpr, static_cast<uint32_t>(address), arch);
  const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
      address_hi_vgpr, static_cast<uint32_t>(address >> 32u), arch);
  const auto mov_marker = instrumentation::build_v_mov_b32_literal(
      marker_vgpr, options.supercollider_report_marker, arch);
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

struct SuperColliderDirectToLdsScratchLayout {
  uint16_t payload_vgpr = 0;
  uint16_t address_vgpr = 0;
  uint16_t readback_vgpr = 0;
  uint16_t report_vgpr = 0;
  uint16_t count = 0;
};

[[nodiscard]] std::optional<SuperColliderDirectToLdsScratchLayout>
direct_to_lds_scratch_layout(const ProgramSite &site, const TargetProfile &target,
                             const Options &options, uint16_t scratch_vgpr) {
  const auto dwords = access_dword_count(site);
  if (!dwords || !site.lowering.form ||
      site.lowering.form->kind != AccessLoweringFormKind::DirectToLdsLaneAddressed ||
      !access_scratch_tuple_base_is_valid(site, scratch_vgpr))
    return std::nullopt;
  const auto align_tuple = [&](uint32_t value) {
    return target.requires_even_vgpr_tuples && *dwords > 1u ? (value + 1u) & ~uint32_t{1u} : value;
  };
  const uint32_t payload = scratch_vgpr;
  const uint32_t address = payload + *dwords;
  const uint32_t readback = align_tuple(address + 1u);
  uint32_t end = readback + *dwords;
  uint32_t report = end;
  if (options.supercollider_report_buffer_address) {
    if (lds_report_action_reuses_readback(options, target, *dwords,
                                          static_cast<uint16_t>(readback))) {
      report = readback;
      end = std::max(end, report + report_action_scratch_vgprs(options));
    } else {
      report = target.requires_even_vgpr_tuples ? (end + 1u) & ~uint32_t{1u} : end;
      end = report + report_action_scratch_vgprs(options);
    }
  }
  if (end > 256u)
    return std::nullopt;
  return SuperColliderDirectToLdsScratchLayout{
      .payload_vgpr = static_cast<uint16_t>(payload),
      .address_vgpr = static_cast<uint16_t>(address),
      .readback_vgpr = static_cast<uint16_t>(readback),
      .report_vgpr = static_cast<uint16_t>(report),
      .count = static_cast<uint16_t>(end - scratch_vgpr),
  };
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_direct_to_lds_check_trap_words(
    std::span<const uint8_t> original_bytes, const ProgramSite &site, const TargetProfile &target,
    const Options &options, uint16_t scratch_vgpr, uint16_t vcc_save_sgpr,
    std::span<const uint32_t> delay_words, std::vector<std::string> &errors) {
  const rj_code_arch_t arch = target.arch;
  const auto dwords = access_dword_count(site);
  const auto layout = direct_to_lds_scratch_layout(site, target, options, scratch_vgpr);
  if (!dwords || !layout) {
    errors.emplace_back("ConSan Direct-to-LDS check/trap selected unsupported scratch layout");
    return std::nullopt;
  }
  std::array<uint32_t, 2> original{};
  if (!read_words_at(original_bytes, site.decoded_file_offset(), std::span<uint32_t>(original))) {
    errors.emplace_back("ConSan Direct-to-LDS check/trap could not read guest access");
    return std::nullopt;
  }
  const auto transfer = supercollider_build_direct_to_lds_transfer(
      original, site.lowering.form->element_width_bits, layout->address_vgpr, layout->payload_vgpr,
      layout->readback_vgpr, arch);
  const auto wait_global = instrumentation::build_s_wait_global_load0(arch);
  const auto wait_lds = instrumentation::build_s_wait_lds0(arch);
  const bool descriptor_partitioned =
      target.scalar_placement_model == ScalarPlacementModel::DescriptorPartitioned;
  const std::optional<uint32_t> save_vcc =
      descriptor_partitioned
          ? instrumentation::build_s_mov_b64(vcc_save_sgpr, kAmdGpuVccLo, arch)
          : std::optional<uint32_t>(build_s_mov_b32(vcc_save_sgpr, kAmdGpuVccLo, arch));
  const std::optional<uint32_t> restore_vcc =
      descriptor_partitioned
          ? instrumentation::build_s_mov_b64(kAmdGpuVccLo, vcc_save_sgpr, arch)
          : std::optional<uint32_t>(build_s_mov_b32(kAmdGpuVccLo, vcc_save_sgpr, arch));
  auto mismatch = build_mismatch_action_words(options, target, layout->report_vgpr, errors,
                                              "ConSan Direct-to-LDS check/trap");
  if (!transfer || !wait_global || !wait_lds || !save_vcc || !restore_vcc || !mismatch ||
      mismatch->size() > static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
    errors.emplace_back("ConSan Direct-to-LDS check/trap could not encode sequence");
    return std::nullopt;
  }
  const auto skip_action =
      instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(mismatch->size()), arch);
  if (!skip_action) {
    errors.emplace_back("ConSan Direct-to-LDS check/trap could not encode mismatch branch");
    return std::nullopt;
  }

  std::vector<uint32_t> words;
  if (!detail::append_materialize_direct_to_lds_address(
          words, site, layout->address_vgpr, layout->payload_vgpr, vcc_save_sgpr, arch)) {
    errors.emplace_back("ConSan Direct-to-LDS check/trap could not materialize LDS address");
    return std::nullopt;
  }
  words.insert(words.end(), transfer->global_load.begin(), transfer->global_load.end());
  words.push_back(*wait_global);
  words.insert(words.end(), transfer->lds_write.begin(), transfer->lds_write.end());
  words.push_back(*wait_lds);
  words.insert(words.end(), delay_words.begin(), delay_words.end());
  words.insert(words.end(), transfer->lds_readback.begin(), transfer->lds_readback.end());
  words.push_back(*wait_lds);
  words.push_back(*save_vcc);
  for (uint16_t i = 0; i < *dwords; ++i) {
    const auto compare =
        supercollider_build_v_cmp_ne_u32(static_cast<uint16_t>(layout->payload_vgpr + i),
                                         static_cast<uint16_t>(layout->readback_vgpr + i), arch);
    if (!compare) {
      errors.emplace_back("ConSan Direct-to-LDS check/trap could not encode chunk compare");
      return std::nullopt;
    }
    words.push_back(*compare);
    words.push_back(*skip_action);
    words.insert(words.end(), mismatch->begin(), mismatch->end());
  }
  words.push_back(*restore_vcc);
  return words;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_lds_check_trap_words(
    std::span<const uint8_t> original_bytes, const ProgramSite &site, const TargetProfile &target,
    const Options &options, uint16_t scratch_vgpr, uint16_t vcc_save_sgpr,
    uint16_t selectable_vgpr_bank_mode, std::span<const uint32_t> delay_words,
    std::optional<uint32_t> &guest_instruction_word_offset, std::vector<std::string> &errors) {
  const rj_code_arch_t arch = target.arch;
  guest_instruction_word_offset.reset();
  auto required_vgprs = access_dword_count(site);
  if (!required_vgprs) {
    errors.emplace_back("ConSan LDS check/trap proof selected unsupported width");
    return std::nullopt;
  }
  const AccessLoweringForm &form = *site.lowering.form;

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
  const bool reuse_readback_for_report = lds_report_action_reuses_readback(
      options, target, static_cast<uint16_t>(*required_vgprs), scratch_vgpr);
  uint16_t report_scratch_vgpr = reuse_readback_for_report ? scratch_vgpr : core_scratch_end;
  if (!reuse_readback_for_report && options.supercollider_report_buffer_address &&
      target.requires_even_vgpr_tuples && report_scratch_vgpr % 2u != 0)
    ++report_scratch_vgpr;
  const uint16_t report_scratch_end = static_cast<uint16_t>(
      report_scratch_vgpr + static_cast<uint16_t>(report_action_scratch_vgprs(options)));
  const uint16_t saved_address_vgpr = std::max(core_scratch_end, report_scratch_end);
  if (site.kind == LdsAccessKind::Read) {
    if (form.destination_accvgpr &&
        target.accumulator_model == AccumulatorModel::DescriptorPartitioned)
      duplicate_load[0] &= ~(uint32_t{1} << 25u);
    duplicate_load[1] =
        (duplicate_load[1] & 0x00FFFFFFu) | (static_cast<uint32_t>(scratch_vgpr) << 24u);
    if (preserve_address)
      duplicate_load[1] = (duplicate_load[1] & 0xFFFFFF00u) | saved_address_vgpr;
  } else if (site.kind == LdsAccessKind::Write) {
    auto word0 = supercollider_build_ds_load_word0(form, original_access[0], arch);
    if (!word0) {
      errors.emplace_back("ConSan LDS check/trap proof could not encode store readback");
      return std::nullopt;
    }
    duplicate_load[0] = *word0;
    duplicate_load[1] = supercollider_build_ds_load_word1(*form.address_vgpr, scratch_vgpr);
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
              target.accumulator_model == AccumulatorModel::DescriptorPartitioned
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
  const bool descriptor_partitioned =
      target.scalar_placement_model == ScalarPlacementModel::DescriptorPartitioned;
  const std::optional<uint32_t> save_vcc =
      descriptor_partitioned
          ? instrumentation::build_s_mov_b64(vcc_save_sgpr, kAmdGpuVccLo, arch)
          : std::optional<uint32_t>(build_s_mov_b32(vcc_save_sgpr, kAmdGpuVccLo, arch));
  const std::optional<uint32_t> restore_vcc =
      descriptor_partitioned
          ? instrumentation::build_s_mov_b64(kAmdGpuVccLo, vcc_save_sgpr, arch)
          : std::optional<uint32_t>(build_s_mov_b32(kAmdGpuVccLo, vcc_save_sgpr, arch));
  if (!wait_dscnt || !skip_action || !save_vcc || !restore_vcc) {
    errors.emplace_back("ConSan LDS check/trap proof could not encode sequence");
    return std::nullopt;
  }

  std::optional<std::vector<uint32_t>> subdword_store_difference;
  if (is_subdword_lds_store(site)) {
    // The readback is already zero-extended. Form the significant-bit
    // difference in that same scratch register instead of consuming a second
    // temporary solely to mask the guest source. This is exact for both low
    // and high D16 source placement and keeps subword stores viable at full
    // VGPR pressure.
    std::vector<uint32_t> difference;
    uint32_t significant_bit_offset = 0u;
    if (is_high_d16_lds_store(site)) {
      const auto shift = instrumentation::build_v_lshlrev_b32(
          scratch_vgpr, scalar_positive_inline_u32(16u), scratch_vgpr, arch);
      if (!shift) {
        errors.emplace_back("ConSan LDS check/trap proof could not align high store readback");
        return std::nullopt;
      }
      difference.push_back(*shift);
      significant_bit_offset = 16u;
    }
    const auto xor_difference = instrumentation::build_v_xor_b32(
        scratch_vgpr, vector_source_vgpr(*first_compare_vgpr), scratch_vgpr, arch);
    if (!xor_difference) {
      errors.emplace_back("ConSan LDS check/trap proof could not form sub-dword store difference");
      return std::nullopt;
    }
    difference.push_back(*xor_difference);
    const uint32_t significant_mask = ((uint32_t{1} << form.element_width_bits) - 1u)
                                      << significant_bit_offset;
    const auto mask = instrumentation::build_v_and_b32_literal(scratch_vgpr, significant_mask,
                                                               scratch_vgpr, arch);
    if (!mask) {
      errors.emplace_back("ConSan LDS check/trap proof could not mask sub-dword store difference");
      return std::nullopt;
    }
    difference.insert(difference.end(), mask->begin(), mask->end());
    subdword_store_difference = std::move(difference);
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
        site.kind == LdsAccessKind::Read ? form.destination_vgpr : form.data_vgpr;
    const std::optional<uint16_t> second_data_vgpr =
        site.kind == LdsAccessKind::Read && first_data_vgpr
            ? std::optional<uint16_t>(
                  static_cast<uint16_t>(*first_data_vgpr + split_two_address->element_dwords))
            : form.second_data_vgpr;
    const auto split = first_data_vgpr && second_data_vgpr
                           ? supercollider_build_split_single_address_lds_pair(
                                 split_two_address->offsets, split_two_address->element_dwords,
                                 replay_address_vgpr, *first_data_vgpr, *second_data_vgpr,
                                 compare_scratch_vgpr, site.kind == LdsAccessKind::Read, target)
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
  words.insert(words.end(), delay_words.begin(), delay_words.end());
  if (split_two_address) {
    const uint16_t replay_address_vgpr = preserve_address ? saved_address_vgpr : *form.address_vgpr;
    const auto split = supercollider_build_split_single_address_lds_pair(
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
  if (subdword_store_difference)
    words.insert(words.end(), subdword_store_difference->begin(), subdword_store_difference->end());
  words.push_back(*save_vcc);
  for (uint16_t i = 0; i < *required_vgprs; ++i) {
    auto compare_vgpr = check_trap_compare_vgpr(site, i, arch, selectable_vgpr_bank_mode);
    if (form.destination_accvgpr && descriptor_partitioned) {
      const auto acc_read = supercollider_build_cdna_accvgpr_read_b32(
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
    const auto chunk_cmp_ne =
        is_subdword_lds_store(site) ? instrumentation::build_v_cmp_ne_u32_vcc(
                                          scalar_positive_inline_u32(0u), scratch_vgpr, arch)
        : compare_vgpr
            ? supercollider_build_v_cmp_ne_u32(static_cast<uint16_t>(*compare_vgpr & 0xFFu),
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
