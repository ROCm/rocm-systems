[[nodiscard]] uint16_t
flat_check_trap_comparison_scratch_vgprs(const ConSanAccessInventorySite &access) {
  if (!access.lowering.form)
    return 0u;
  const ConSanAccessLoweringForm &form = *access.lowering.form;
  return form.register_value_placement == ConSanAccessRegisterValuePlacement::High16 ||
                 (form.access_kind == ConSanLdsAccessKind::Write && form.element_width_bits == 8u)
             ? 1u
             : 0u;
}

[[nodiscard]] std::optional<uint16_t> choose_flat_check_trap_vcc_save_sgpr(
    const Instruction *inst, std::span<const LivenessAnalysis *const> owner_liveness,
    std::optional<uint16_t> min_preferred_sgpr, const ConSanOptions &options,
    bool preserve_dynamic_stack_abi) {
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness, [](const auto *liveness) { return liveness == nullptr; }))
    return std::nullopt;
  const auto find_pair = [&](uint16_t start) -> std::optional<uint16_t> {
    while (auto pair = owner_liveness.front()->find_free_sgpr_pair(inst, start)) {
      const bool overlaps_dynamic_stack =
          preserve_dynamic_stack_abi && sc_sgpr_ranges_overlap(*pair, 2u, kDynamicStackTopSgpr, 2u);
      const RegisterRef ref{RegClass::SGPR, *pair, 2u};
      const bool dead_for_every_owner =
          std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
            return !liveness->any_live_before(*inst, ref);
          });
      if (!sc_sgpr_range_overlaps_delay_source(*pair, 2u, options) && !overlaps_dynamic_stack &&
          dead_for_every_owner)
        return pair;
      if (*pair + 2u >= REGISTER_SET_ALLOCATABLE_SGPRS)
        return std::nullopt;
      start = static_cast<uint16_t>(*pair + 2u);
    }
    return std::nullopt;
  };
  if (min_preferred_sgpr) {
    if (auto high_pair = find_pair(*min_preferred_sgpr))
      return high_pair;
  }
  return find_pair(0);
}

[[nodiscard]] std::optional<uint16_t> choose_flat_check_trap_scalar_spill_pair(
    uint16_t allocated_sgpr_count, const ConSanOptions &options, bool preserve_dynamic_stack_abi) {
  // The caller spills and restores the returned live pair, so any ordinary
  // allocated pair is legal except the active delay source. Staying within
  // the descriptor allocation (and every owner's common minimum for shared
  // code) avoids growing an owner around a site-local choice.
  const uint16_t limit = std::min<uint16_t>(allocated_sgpr_count, REGISTER_SET_ALLOCATABLE_SGPRS);
  for (uint16_t pair = 0; static_cast<uint32_t>(pair) + 2u <= limit; pair += 2u) {
    const bool overlaps_dynamic_stack =
        preserve_dynamic_stack_abi && sc_sgpr_ranges_overlap(pair, 2u, kDynamicStackTopSgpr, 2u);
    if (!sc_sgpr_range_overlaps_delay_source(pair, 2u, options) && !overlaps_dynamic_stack)
      return pair;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> choose_flat_check_trap_scratch_vgpr(
    const ConSanAccessInventorySite &access, const ConSanOptions &options, const Instruction *inst,
    std::span<const LivenessAnalysis *const> owner_liveness,
    std::optional<uint16_t> min_auto_scratch_vgpr, std::optional<uint16_t> max_auto_scratch_vgpr,
    uint16_t required_vgprs) {
  if (options.scratch_vgpr) {
    return choose_scratch_vgpr(access, options, inst, nullptr, min_auto_scratch_vgpr,
                               max_auto_scratch_vgpr, required_vgprs);
  }
  if (inst == nullptr || owner_liveness.empty() ||
      std::ranges::any_of(owner_liveness,
                          [](const auto *liveness) { return liveness == nullptr; }) ||
      required_vgprs > std::numeric_limits<uint8_t>::max())
    return std::nullopt;

  RegisterSet live_before_all_owners;
  for (const LivenessAnalysis *liveness : owner_liveness)
    live_before_all_owners |= liveness->live_before(*inst);

  ConSanRegisterRequest request;
  request.reg_class = RegClass::VGPR;
  request.count = required_vgprs;
  request.alignment = access.lowering.form ? access.lowering.form->data_register_alignment : 1u;
  request.search_start =
      std::max(access_scratch_search_start(access), min_auto_scratch_vgpr.value_or(0u));
  // RegisterSet can represent target-specific VGPR banks above 255, but the
  // SuperCollider probe ABI directly encodes only ordinary low-bank VGPRs.
  // LivenessAnalysis::find_free_run enforced this bound through its configured
  // max_free_vgpr; preserve it explicitly at the shared planner boundary.
  request.current_allocation_count =
      std::min(max_auto_scratch_vgpr.value_or(kConSanOrdinaryVgprLimit), kConSanOrdinaryVgprLimit);
  request.max_referenced_count = min_auto_scratch_vgpr.value_or(0u);
  request.architecture_limit = request.current_allocation_count;
  request.allow_spill = false;
  if (!access.lowering.form)
    return std::nullopt;
  const ConSanAccessLoweringForm &form = *access.lowering.form;
  if (form.address_vgpr)
    request.forbidden.expand(
        {RegClass::VGPR, *form.address_vgpr, static_cast<uint8_t>(form.address_vgpr_count)});
  if (form.destination_vgpr)
    request.forbidden.expand({RegClass::VGPR, *form.destination_vgpr,
                              static_cast<uint8_t>(form.destination_register_count)});
  if (form.data_vgpr)
    request.forbidden.expand(
        {RegClass::VGPR, *form.data_vgpr, static_cast<uint8_t>(form.data_register_count)});

  const ConSanRegisterPlan plan = plan_consan_registers(request, live_before_all_owners);
  return plan.source == ConSanRegisterAllocationSource::LivenessDead ? plan.base : std::nullopt;
}

void try_apply_flat_check_trap_patch(const AmdGpuCodeObject &code_object,
                                     const ConSanTargetProfile &target,
                                     const ConSanOptions &options, ConSanTransformArtifacts &result,
                                     std::span<const ByteRange>) {
  const rj_code_arch_t arch = target.arch;

  const std::unordered_set<uint64_t> admitted_file_offsets = sc_admitted_access_file_offsets(
      result.program_inventory, result.coverage_ledger, ConSanAccessOrigin::Flat);
  std::map<uint64_t, const ConSanAccessInventorySite *> admitted_sites;
  for (const ConSanAccessInventorySite &access : result.program_inventory.access_sites()) {
    if (access.origin != ConSanAccessOrigin::Flat ||
        !admitted_file_offsets.contains(access.file_offset))
      continue;
    admitted_sites.try_emplace(access.file_offset, &access);
  }
  if (admitted_sites.empty()) {
    result.warnings.emplace_back(
        "ConSan flat check/trap proof found no supported likely group-FLAT load/store site");
    return;
  }

  if (!consan_sc_delay_instruction_word_count(options, result.errors,
                                              "ConSan flat check/trap proof"))
    return;

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());

  struct FlatOwnerPlanningContext {
    uint64_t descriptor_file_offset = 0;
    KernelCfgScope scope;
    std::unique_ptr<LivenessAnalysis> liveness;
  };
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  std::vector<BasicBlock *> block_ptrs;
  BlockOffsetIndex block_index;
  std::vector<uint64_t> kernel_entries;
  std::span<const uint8_t> pristine_text;
  std::unordered_map<uint64_t, FlatOwnerPlanningContext> owner_contexts;
  std::unordered_set<uint64_t> failed_owner_contexts;
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (decoder) {
    const consan_detail::ConSanCfgBuildInputs cfg = consan_detail::build_consan_cfg_inputs(
        code_object, result.program_inventory.kernels(), result.program_inventory.functions());
    blocks = BasicBlock::build(code_object, *decoder, arch, cfg.leaders, cfg.code_ranges);
    block_ptrs = block_ptrs_for(blocks);

    block_index = build_block_offset_index(blocks);
    kernel_entries = cfg.kernel_entries;
    if (code_object.text_sections().size() == 1) {
      const Section *text = code_object.text_sections().front();
      pristine_text =
          std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
    }
  }
  owner_contexts.reserve(result.program_inventory.kernels().size());
  failed_owner_contexts.reserve(result.program_inventory.kernels().size());

  struct FlatCheckTrapCandidate {
    const ConSanAccessInventorySite *site = nullptr;
    uint16_t scratch_vgpr = 0;
    std::vector<uint64_t> owner_descriptor_file_offsets;
    uint16_t vcc_save_sgpr = 0;
    std::optional<ConSanSuperColliderScalarVccSpill> scalar_vcc_spill;
    std::optional<VgprSpillSequence> vgpr_spill;
    std::optional<DynamicStackBorrowedSgprSpillSequence> dynamic_stack_spill;
  };

  enum class FlatOwnerStackKind : uint8_t {
    Fixed,
    Dynamic,
    Mixed,
  };

  const auto make_candidate = [](const ConSanAccessInventorySite *site, uint16_t scratch_vgpr,
                                 const std::vector<uint64_t> &owner_descriptor_file_offsets) {
    FlatCheckTrapCandidate candidate;
    candidate.site = site;
    candidate.scratch_vgpr = scratch_vgpr;
    candidate.owner_descriptor_file_offsets = owner_descriptor_file_offsets;
    return candidate;
  };

  std::vector<FlatCheckTrapCandidate> candidates;
  size_t supported_candidate_count = 0;
  size_t scratchable_candidate_count = 0;
  size_t spill_backed_candidate_count = 0;
  size_t mixed_stack_owner_spill_rejection_count = 0;
  size_t dynamic_stack_spill_encoding_failure_count = 0;
  size_t private_spill_encoding_failure_count = 0;
  auto visit_site = [&](const ConSanAccessInventorySite &site,
                        std::optional<uint16_t> min_auto_scratch_vgpr,
                        std::optional<uint16_t> max_auto_scratch_vgpr,
                        std::optional<uint16_t> min_fresh_sgpr,
                        std::optional<uint16_t> allocated_sgpr_count,
                        uint32_t original_private_segment_size, FlatOwnerStackKind owner_stack_kind,
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
    const Instruction *site_inst = find_instruction_at_text_offset(
        KernelBlockScope(block_ptrs), site.physical_id.original_text_offset);
    const bool uses_dynamic_stack = owner_stack_kind == FlatOwnerStackKind::Dynamic;
    const bool preserve_dynamic_stack_abi = owner_stack_kind != FlatOwnerStackKind::Fixed;
    auto vcc_save = choose_flat_check_trap_vcc_save_sgpr(site_inst, owner_liveness, min_fresh_sgpr,
                                                         options, preserve_dynamic_stack_abi);
    bool spill_vcc_save_pair_to_vgprs = false;
    if (!vcc_save && consan_arch_is_rdna4_or_cdna5(target.arch) && allocated_sgpr_count) {
      vcc_save = choose_flat_check_trap_scalar_spill_pair(*allocated_sgpr_count, options,
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
    auto scratch = choose_flat_check_trap_scratch_vgpr(site, options, site_inst, owner_liveness,
                                                       min_auto_scratch_vgpr, max_auto_scratch_vgpr,
                                                       scratch_vgprs);
    std::optional<VgprSpillSequence> vgpr_spill;
    std::optional<DynamicStackBorrowedSgprSpillSequence> dynamic_stack_spill;
    if (!scratch && !options.scratch_vgpr && max_auto_scratch_vgpr) {
      if (owner_stack_kind == FlatOwnerStackKind::Mixed) {
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
    candidates.push_back(make_candidate(&site, *scratch, owner_descriptor_file_offsets));
    candidates.back().vcc_save_sgpr = *vcc_save;
    candidates.back().scalar_vcc_spill = scalar_vcc_spill;
    candidates.back().vgpr_spill = vgpr_spill;
    candidates.back().dynamic_stack_spill = dynamic_stack_spill;
  };

  // Shared functions can contribute hundreds of candidate sites to the same
  // owner.  Computing the owner's maximum register references for every site
  // turns otherwise modest multi-kernel objects into a quadratic scan.  The
  // answer depends only on the immutable decoded blocks and kernel range, so
  // compute it once per descriptor and reuse it for both kernel-local and
  // shared-function candidates.
  std::unordered_map<uint64_t, const ConSanKernelInfo *> kernels_by_descriptor;
  std::unordered_map<uint64_t, KernelMaxRegisterRefs> max_refs_by_descriptor;
  kernels_by_descriptor.reserve(result.program_inventory.kernels().size());
  max_refs_by_descriptor.reserve(result.program_inventory.kernels().size());
  for (const ConSanKernelInfo &kernel : result.program_inventory.kernels())
    kernels_by_descriptor.emplace(kernel.descriptor_file_offset, &kernel);
  const auto get_owner_context = [&](const ConSanKernelInfo &kernel) -> FlatOwnerPlanningContext * {
    const uint64_t descriptor_file_offset = kernel.descriptor_file_offset;
    if (const auto existing = owner_contexts.find(descriptor_file_offset);
        existing != owner_contexts.end())
      return &existing->second;
    if (decoder == nullptr || !kernel.has_text_range ||
        failed_owner_contexts.contains(descriptor_file_offset))
      return nullptr;
    auto scope = build_kernel_cfg_scope(blocks, block_index,
                                        KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                                                           .additional_entry_offsets = {}},
                                        kernel_entries, pristine_text);
    if (!scope) {
      failed_owner_contexts.insert(descriptor_file_offset);
      return nullptr;
    }
    FlatOwnerPlanningContext context{
        .descriptor_file_offset = descriptor_file_offset,
        .scope = std::move(*scope),
        .liveness = nullptr,
    };
    context.liveness =
        std::make_unique<LivenessAnalysis>(KernelBlockScope(context.scope.blocks),
                                           LivenessAnalysisOptions{}, context.scope.liveness_edges);
    return &owner_contexts.emplace(descriptor_file_offset, std::move(context)).first->second;
  };

  for (const auto &[file_offset, access] : admitted_sites) {
    (void)file_offset;
    const ConSanAccessInventorySite &site = *access;
    const Instruction *site_inst = find_instruction_at_text_offset(
        KernelBlockScope(block_ptrs), site.physical_id.original_text_offset);
    std::vector<uint64_t> owners = site.execution_and_container_owner_descriptor_file_offsets();
    for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
      if (plan.site_kind == ConSanResourceSiteKind::Access &&
          plan.text_offset == site.physical_id.original_text_offset) {
        owners.insert(owners.end(), plan.owner_descriptor_file_offsets.begin(),
                      plan.owner_descriptor_file_offsets.end());
      }
    }
    for (const ConSanFaultSite &fault : result.fault_sites) {
      if (fault.in_kernel || fault.container_name != site.container.name)
        continue;
      for (const ConSanExecutionOwner &owner : fault.execution_owners)
        owners.push_back(owner.descriptor_file_offset);
    }
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());

    bool complete_context = decoder != nullptr && site_inst != nullptr && !owners.empty();
    std::vector<const LivenessAnalysis *> owner_liveness;
    owner_liveness.reserve(owners.size());
    std::optional<uint16_t> common_allocated_vgpr_count;
    std::optional<uint16_t> common_allocated_sgpr_count;
    std::optional<uint16_t> max_owner_vgpr_ref;
    std::optional<uint16_t> max_owner_sgpr_ref;
    uint32_t max_owner_private_segment_size = 0;
    bool any_owner_uses_dynamic_stack = false;
    bool any_owner_uses_fixed_stack = false;
    for (uint64_t owner : owners) {
      const auto owner_kernel = kernels_by_descriptor.find(owner);
      FlatOwnerPlanningContext *owner_context = owner_kernel == kernels_by_descriptor.end()
                                                    ? nullptr
                                                    : get_owner_context(*owner_kernel->second);
      auto refs = max_refs_by_descriptor.find(owner);
      if (refs == max_refs_by_descriptor.end() && owner_context != nullptr) {
        refs =
            max_refs_by_descriptor
                .emplace(owner,
                         max_register_refs_in_kernel(*owner_kernel->second,
                                                     KernelBlockScope(owner_context->scope.blocks)))
                .first;
      }
      const auto descriptor = read_kernel_descriptor(original_bytes, owner);
      const Instruction *owner_site_inst =
          owner_context == nullptr
              ? nullptr
              : find_instruction_at_text_offset(KernelBlockScope(owner_context->scope.blocks),
                                                site.physical_id.original_text_offset);
      if (owner_kernel == kernels_by_descriptor.end() ||
          owner_kernel->second->preflight_action == ConSanPreflightAction::Blocked ||
          owner_context == nullptr || owner_context->liveness == nullptr ||
          owner_site_inst != site_inst || refs == max_refs_by_descriptor.end() || !descriptor) {
        complete_context = false;
        break;
      }
      owner_liveness.push_back(owner_context->liveness.get());

      const uint16_t allocated_vgprs = descriptor_vgpr_allocation_count(*descriptor, arch);
      const uint16_t allocated_sgprs = descriptor_sgpr_allocation_count(*descriptor, arch);
      common_allocated_vgpr_count = common_allocated_vgpr_count
                                        ? std::min(*common_allocated_vgpr_count, allocated_vgprs)
                                        : allocated_vgprs;
      common_allocated_sgpr_count = common_allocated_sgpr_count
                                        ? std::min(*common_allocated_sgpr_count, allocated_sgprs)
                                        : allocated_sgprs;
      max_owner_private_segment_size =
          std::max(max_owner_private_segment_size, descriptor->private_segment_fixed_size);
      if (refs->second.vgpr)
        max_owner_vgpr_ref = max_owner_vgpr_ref ? std::max(*max_owner_vgpr_ref, *refs->second.vgpr)
                                                : *refs->second.vgpr;
      if (refs->second.sgpr)
        max_owner_sgpr_ref = max_owner_sgpr_ref ? std::max(*max_owner_sgpr_ref, *refs->second.sgpr)
                                                : *refs->second.sgpr;
      if (owner_kernel->second->uses_dynamic_stack.value_or(false))
        any_owner_uses_dynamic_stack = true;
      else
        any_owner_uses_fixed_stack = true;
    }
    if (!complete_context || !common_allocated_vgpr_count || !common_allocated_sgpr_count) {
      (void)publish_sc_access_rejection(
          result.coverage_ledger, result.errors, site.physical_id,
          ConSanLoweringOutcomeKind::ResourceRejected,
          "SuperCollider FLAT lowerer found no complete planning context");
      continue;
    }

    const bool selected_by_filter =
        options.test_kernel_name_filter.empty() ||
        site.container.name.find(options.test_kernel_name_filter) != std::string::npos ||
        std::ranges::any_of(owners, [&](uint64_t owner) {
          const auto kernel = kernels_by_descriptor.find(owner);
          return kernel != kernels_by_descriptor.end() &&
                 kernel->second->name.find(options.test_kernel_name_filter) != std::string::npos;
        });
    if (!selected_by_filter)
      continue;

    std::optional<uint16_t> min_auto_scratch_vgpr;
    if (max_owner_vgpr_ref && *max_owner_vgpr_ref < 255u)
      min_auto_scratch_vgpr = static_cast<uint16_t>(*max_owner_vgpr_ref + 1u);
    std::optional<uint16_t> min_fresh_sgpr;
    if (max_owner_sgpr_ref && *max_owner_sgpr_ref + 1u < REGISTER_SET_ALLOCATABLE_SGPRS)
      min_fresh_sgpr = static_cast<uint16_t>(*max_owner_sgpr_ref + 1u);
    const FlatOwnerStackKind owner_stack_kind =
        any_owner_uses_dynamic_stack && any_owner_uses_fixed_stack ? FlatOwnerStackKind::Mixed
        : any_owner_uses_dynamic_stack                             ? FlatOwnerStackKind::Dynamic
                                                                   : FlatOwnerStackKind::Fixed;
    visit_site(site, min_auto_scratch_vgpr, common_allocated_vgpr_count, min_fresh_sgpr,
               common_allocated_sgpr_count, max_owner_private_segment_size, owner_stack_kind,
               owner_liveness, std::move(owners));
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
    return candidate.site->file_offset;
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
        " private_spill_encoding_failures=" +
        std::to_string(private_spill_encoding_failure_count));
    return;
  }

  struct PlannedFlatCheckTrapPatch {
    const ConSanAccessInventorySite *site = nullptr;
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
    const ConSanAccessInventorySite &site = *candidate.site;
    const auto required_vgprs = access_dword_count(site);
    const auto compare_vgpr = flat_check_trap_compare_vgpr(site);
    if (!required_vgprs || !compare_vgpr) {
      result.errors.emplace_back("ConSan flat check/trap proof selected unsupported site");
      return;
    }

    const size_t access_words = site.instruction_size / sizeof(uint32_t);
    if (access_words < 2u || access_words > 3u) {
      result.errors.emplace_back("ConSan flat check/trap proof selected invalid access size");
      return;
    }
    std::array<uint32_t, 3> original_access{};
    if (!read_words_at(original_bytes, site.file_offset,
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
    if (!consan_sc_append_delay_words(planned.words, arch, options, result.errors,
                                      "ConSan flat check/trap proof"))
      return;
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

    planned.info.original_size = site.instruction_size;
    planned.info.trampoline_size =
        static_cast<uint32_t>(planned.words.size() * sizeof(uint32_t));

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
  std::vector<ConSanStagedTextRewrite> rewrites;
  rewrites.reserve(planned_patches.size());
  for (PlannedFlatCheckTrapPatch &planned : planned_patches) {
    rewrites.push_back({
        .original_site = planned.site->physical_id,
        .intent_ids = std::move(planned.intent_ids),
        .words = std::move(planned.words),
        .patch = std::move(planned.info),
    });
  }
  (void)stage_consan_text_rewrites(
      code_object, arch,
      {no_vgpr_requirements, descriptor_sgpr_growths, descriptor_private_requirements,
       no_group_segment_requirements},
      {.maximum_ordinary_vgpr_count = REGISTER_SET_MAX_VGPRS,
       .inventory_proves_empty_accumulator_bank = false,
       .maximum_group_segment_bytes = std::nullopt},
      "ConSan flat check/trap proof", std::move(rewrites), result);
}

void try_apply_flat_trap_patch(const AmdGpuCodeObject &code_object,
                               const ConSanTargetProfile &target, const ConSanOptions &options,
                               ConSanTransformArtifacts &result) {
  const rj_code_arch_t arch = target.arch;
  if (!target.supports_sc_inline_flat_trap_rewrite) {
    result.warnings.emplace_back("ConSan flat trap proof currently supports only RDNA4");
    return;
  }

  auto trap = instrumentation::build_s_trap(0, arch);
  if (!trap) {
    result.errors.emplace_back("ConSan flat trap proof could not encode s_trap");
    return;
  }

  std::vector<const ConSanAccessInventorySite *> candidates;
  for (const ConSanAccessInventorySite &access : result.program_inventory.access_sites()) {
    if (access.origin != ConSanAccessOrigin::Flat ||
        (access.kind != ConSanLdsAccessKind::Read && access.kind != ConSanLdsAccessKind::Write) ||
        access.instruction_size != 3u * sizeof(uint32_t) ||
        !is_instrumentable_group_flat_hint(access.flat_address_space_hint,
                                           options.flat_provenance_mode))
      continue;
    if (access.container.kernel_descriptor_file_offset) {
      const ConSanKernelInfo *kernel = result.program_inventory.find_kernel_by_descriptor(
          *access.container.kernel_descriptor_file_offset);
      if (kernel == nullptr || kernel->preflight_action == ConSanPreflightAction::Blocked)
        continue;
    }
    candidates.push_back(&access);
  }

  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan flat trap proof found no likely group flat_load/store site");
    return;
  }

  std::ranges::sort(
      candidates, [](const ConSanAccessInventorySite *lhs, const ConSanAccessInventorySite *rhs) {
        if (lhs->file_offset != rhs->file_offset)
          return lhs->file_offset < rhs->file_offset;
        return lhs->physical_id.original_text_offset < rhs->physical_id.original_text_offset;
      });

  result.replacement.assign(reinterpret_cast<const uint8_t *>(code_object.image_data()),
                            reinterpret_cast<const uint8_t *>(code_object.image_data()) +
                                code_object.image_size());
  const std::array<uint32_t, 3> replacement = {*trap, build_s_nop(0, arch), build_s_nop(0, arch)};
  for (const ConSanAccessInventorySite *access : candidates) {
    if (access->file_offset > result.replacement.size() ||
        access->instruction_size > result.replacement.size() - access->file_offset) {
      result.errors.emplace_back("ConSan flat trap proof exceeds ELF bytes");
      result.outcome = ConSanTransformOutcome::Invalid;
      result.discard_candidate_modification();
      return;
    }
    std::memcpy(result.replacement.data() + access->file_offset, replacement.data(),
                replacement.size() * sizeof(uint32_t));

    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::InlineFlatTrapRewrite;
    info.anchor_offset = access->physical_id.original_text_offset;
    info.trampoline_offset = access->physical_id.original_text_offset + access->instruction_size;
    info.original_size = access->instruction_size;
    result.patches.push_back(info);
  }
  result.mark_modified();
}
