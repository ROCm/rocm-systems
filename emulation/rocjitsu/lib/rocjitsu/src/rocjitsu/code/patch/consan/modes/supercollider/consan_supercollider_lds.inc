void try_apply_lds_load_check_trap_patch(const AmdGpuCodeObject &code_object,
                                         const ConSanTargetProfile &target,
                                         const ConSanOptions &options,
                                         ConSanTransformArtifacts &result) {
  const rj_code_arch_t arch = target.arch;

  if (!consan_sc_delay_instruction_word_count(options, result.errors,
                                              "ConSan LDS check/trap proof"))
    return;

  const std::unordered_set<uint64_t> admitted_lds_file_offsets = sc_admitted_access_file_offsets(
      result.program_inventory, result.coverage_ledger, ConSanAccessOrigin::NativeLds);
  std::map<uint64_t, const ConSanAccessInventorySite *> admitted_sites;
  for (const ConSanAccessInventorySite &access : result.program_inventory.access_sites())
    if (access.origin == ConSanAccessOrigin::NativeLds &&
        admitted_lds_file_offsets.contains(access.file_offset))
      admitted_sites.try_emplace(access.file_offset, &access);

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  const bool has_plannable_supported_site =
      std::ranges::any_of(admitted_sites, [&](const auto &entry) {
        const std::vector<uint64_t> owners =
            entry.second->execution_and_container_owner_descriptor_file_offsets();
        return !owners.empty() && std::ranges::all_of(owners, [&](uint64_t owner) {
          const ConSanKernelInfo *kernel =
              result.program_inventory.find_kernel_by_descriptor(owner);
          return kernel != nullptr && kernel->preflight_action != ConSanPreflightAction::Blocked;
        });
      });
  if (!has_plannable_supported_site) {
    for (const auto &[file_offset, site] : admitted_sites) {
      (void)file_offset;
      (void)publish_sc_access_rejection(
          result.coverage_ledger, result.errors, site->physical_id,
          ConSanLoweringOutcomeKind::ResourceRejected,
          "SuperCollider LDS lowerer found no complete execution owner");
    }
    result.warnings.emplace_back("ConSan LDS check/trap proof found no supported "
                                 "native-LDS load or store site with complete owners");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  std::span<const uint8_t> pristine_text;
  if (code_object.text_sections().size() == 1) {
    pristine_text = std::span<const uint8_t>(
        reinterpret_cast<const uint8_t *>(code_object.text_sections().front()->data()),
        code_object.text_sections().front()->size());
  }

  std::vector<std::unique_ptr<BasicBlock>> blocks;
  std::vector<BasicBlock *> block_ptrs;
  struct ScReservoirOwnerContext {
    uint64_t descriptor_file_offset = 0;
    KernelCfgScope scope;
    std::unique_ptr<LivenessAnalysis> liveness;
  };
  std::vector<ScReservoirOwnerContext> reservoir_owner_contexts;
  if (decoder) {
    // Decoding only preflight candidates is a scalability fallback for truly
    // large generated executable text. Smaller bundled libraries still need
    // surrounding skipped kernels as potential relay donors. File size is
    // not the relevant cost signal: large metadata/rodata must not hide a
    // modest text section and create an artificial unbridgeable tail.
    constexpr uint64_t kScBoundedCfgMinimumTextBytes = 4u * 1024u * 1024u;
    uint64_t executable_text_bytes = 0u;
    for (const Section *section : code_object.text_sections())
      executable_text_bytes = saturated_add(executable_text_bytes, section->size());
    const bool restrict_to_preflight_candidates =
        result.program_inventory.functions().empty() &&
        options.flavor == ConSanFlavor::SuperCollider && !options.fault_dry_run &&
        options.sc_perturb_kind == ConSanPerturbationKind::None &&
        executable_text_bytes >= kScBoundedCfgMinimumTextBytes;
    consan_detail::ConSanCfgBuildInputs cfg = consan_detail::build_consan_cfg_inputs(
        code_object, result.program_inventory.kernels(), result.program_inventory.functions());
    if (restrict_to_preflight_candidates) {
      cfg.leaders.clear();
      cfg.kernel_entries.clear();
      cfg.code_ranges.clear();
      for (const ConSanKernelInfo &kernel : result.program_inventory.kernels()) {
        if (!kernel.has_text_range || kernel.preflight_action != ConSanPreflightAction::Candidate)
          continue;
        cfg.leaders.push_back(kernel.entry_text_offset);
        cfg.kernel_entries.push_back(kernel.entry_text_offset);
        if (kernel.code_size != 0u) {
          cfg.code_ranges.push_back(
              {.start_offset = kernel.entry_text_offset, .size = kernel.code_size});
        }
      }
      std::ranges::sort(cfg.leaders);
      cfg.leaders.erase(std::ranges::unique(cfg.leaders).begin(), cfg.leaders.end());
      std::ranges::sort(cfg.kernel_entries);
      cfg.kernel_entries.erase(std::ranges::unique(cfg.kernel_entries).begin(),
                               cfg.kernel_entries.end());
    }
    blocks = BasicBlock::build(code_object, *decoder, arch, cfg.leaders, cfg.code_ranges);
    block_ptrs = block_ptrs_for(blocks);

    const BlockOffsetIndex block_index = build_block_offset_index(blocks);
    reservoir_owner_contexts.reserve(result.program_inventory.kernels().size());
    for (const ConSanKernelInfo &kernel : result.program_inventory.kernels()) {
      // With no shared functions, a preflight-skipped kernel cannot own a
      // supported access. Avoid constructing thousands of unused liveness
      // analyses for generated operator-library code objects.
      if (restrict_to_preflight_candidates &&
          kernel.preflight_action != ConSanPreflightAction::Candidate)
        continue;
      if (!kernel.has_text_range)
        continue;
      const auto descriptor = read_kernel_descriptor(original_bytes, kernel.descriptor_file_offset);
      if (!descriptor)
        continue;
      auto scope =
          build_kernel_cfg_scope(blocks, block_index,
                                 KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                                                    .additional_entry_offsets = {}},
                                 cfg.kernel_entries, pristine_text);
      if (!scope)
        continue;
      reservoir_owner_contexts.push_back(
          {.descriptor_file_offset = kernel.descriptor_file_offset,
           .scope = std::move(*scope),
           .liveness = nullptr});
      ScReservoirOwnerContext &context = reservoir_owner_contexts.back();
      context.liveness = std::make_unique<LivenessAnalysis>(KernelBlockScope(context.scope.blocks),
                                                            LivenessAnalysisOptions{},
                                                            context.scope.liveness_edges);
    }
  }

  struct LdsCheckTrapCandidate {
    const ConSanAccessInventorySite *site = nullptr;
    uint16_t scratch_vgpr = 0;
    uint16_t vcc_save_sgpr = 0;
    std::optional<uint16_t> required_vgpr_allocation_count;
    std::optional<uint16_t> selectable_vgpr_bank_mode;
    std::optional<ConSanSuperColliderScalarVccSpill> scalar_vcc_spill;
  };

  // The save and restore described by this type bracket one instrumentation
  // body and complete before control returns to guest code. That lifetime
  // contract permits independent owner sets to reuse byte offsets. A future
  // spill that survives a call, relay, or guest-visible control-flow edge
  // needs a separate plan type backed by a globally disjoint allocator.
  struct ScBracketLocalPrivateSpillPlan {
    std::vector<uint64_t> owner_descriptor_file_offsets;
    uint32_t original_private_bytes = 0;
    uint32_t private_limit = 0;
    uint16_t vgpr_base = 0;
    uint16_t vgpr_count = 0;
    bool uses_dynamic_stack_frame = false;
    uint16_t saved_frame_base_sgpr = 0;
    uint16_t saved_scc_sgpr = 0;
  };

  struct LdsSitePlanningContext {
    std::string_view container_name;
    uint64_t container_entry_text_offset = 0;
    const ConSanAccessInventorySite *site = nullptr;
    const Instruction *instruction = nullptr;
    std::vector<uint64_t> owner_descriptor_file_offsets;
    std::vector<const ScReservoirOwnerContext *> owner_contexts;
    std::optional<uint16_t> common_allocated_vgpr_count;
    std::optional<uint16_t> common_allocated_sgpr_count;
    std::optional<uint16_t> min_auto_scratch_vgpr;
    std::optional<uint16_t> min_preferred_vcc_save_sgpr;
    uint32_t max_owner_private_segment_size = 0;
    bool uses_dynamic_stack = false;
    bool every_owner_uses_dynamic_stack = true;
    bool every_owner_can_grow_ordinary_vgprs = true;
  };
  std::unordered_map<uint64_t, const ConSanKernelInfo *> kernels_by_descriptor;
  std::unordered_map<uint64_t, KernelMaxRegisterRefs> max_refs_by_descriptor;
  kernels_by_descriptor.reserve(result.program_inventory.kernels().size());
  max_refs_by_descriptor.reserve(result.program_inventory.kernels().size());
  for (const ConSanKernelInfo &kernel : result.program_inventory.kernels()) {
    kernels_by_descriptor.emplace(kernel.descriptor_file_offset, &kernel);
    const auto owner_context =
        std::ranges::find(reservoir_owner_contexts, kernel.descriptor_file_offset,
                          &ScReservoirOwnerContext::descriptor_file_offset);
    if (owner_context != reservoir_owner_contexts.end()) {
      max_refs_by_descriptor.emplace(
          kernel.descriptor_file_offset,
          max_register_refs_in_kernel(kernel, KernelBlockScope(owner_context->scope.blocks)));
    }
  }

  std::vector<LdsSitePlanningContext> site_contexts;
  const auto add_site_context = [&](const ConSanAccessInventorySite &site,
                                    std::vector<uint64_t> owners) {
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
    if (owners.empty()) {
      (void)publish_sc_access_rejection(result.coverage_ledger, result.errors, site.physical_id,
                                        ConSanLoweringOutcomeKind::ResourceRejected,
                                        "SuperCollider LDS lowerer found no execution owner");
      return;
    }

    LdsSitePlanningContext context;
    context.container_name = site.container.name;
    context.container_entry_text_offset = site.container.entry_text_offset;
    context.site = &site;
    context.owner_descriptor_file_offsets = std::move(owners);
    context.instruction = find_instruction_at_text_offset(KernelBlockScope(block_ptrs),
                                                          site.physical_id.original_text_offset);
    bool complete = true;
    bool first_owner = true;
    std::optional<uint16_t> max_owner_vgpr_ref;
    std::optional<uint16_t> max_owner_sgpr_ref;
    for (uint64_t owner : context.owner_descriptor_file_offsets) {
      const auto owner_kernel = kernels_by_descriptor.find(owner);
      const auto owner_context = std::ranges::find(
          reservoir_owner_contexts, owner, &ScReservoirOwnerContext::descriptor_file_offset);
      const auto descriptor = read_kernel_descriptor(original_bytes, owner);
      const Instruction *owner_site_inst =
          owner_context == reservoir_owner_contexts.end()
              ? nullptr
              : find_instruction_at_text_offset(KernelBlockScope(owner_context->scope.blocks),
                                                site.physical_id.original_text_offset);
      if (owner_kernel == kernels_by_descriptor.end() ||
          owner_context == reservoir_owner_contexts.end() || owner_context->liveness == nullptr ||
          context.instruction == nullptr || owner_site_inst != context.instruction || !descriptor ||
          owner_kernel->second->preflight_action == ConSanPreflightAction::Blocked) {
        complete = false;
        break;
      }
      context.owner_contexts.push_back(&*owner_context);
      const uint16_t allocated_vgprs = descriptor_ordinary_vgpr_allocation_count(*descriptor, arch);
      const uint16_t unified_vgprs = descriptor_vgpr_allocation_count(*descriptor, arch);
      const uint16_t allocated_sgprs = descriptor_sgpr_allocation_count(*descriptor, arch);
      context.max_owner_private_segment_size =
          std::max(context.max_owner_private_segment_size, descriptor->private_segment_fixed_size);
      context.common_allocated_vgpr_count =
          first_owner ? allocated_vgprs
                      : std::min(*context.common_allocated_vgpr_count, allocated_vgprs);
      context.common_allocated_sgpr_count =
          first_owner ? allocated_sgprs
                      : std::min(*context.common_allocated_sgpr_count, allocated_sgprs);
      const auto refs_it = max_refs_by_descriptor.find(owner);
      if (refs_it == max_refs_by_descriptor.end()) {
        complete = false;
        break;
      }
      const KernelMaxRegisterRefs &refs = refs_it->second;
      if (refs.vgpr)
        max_owner_vgpr_ref =
            max_owner_vgpr_ref ? std::max(*max_owner_vgpr_ref, *refs.vgpr) : *refs.vgpr;
      if (refs.sgpr)
        max_owner_sgpr_ref =
            max_owner_sgpr_ref ? std::max(*max_owner_sgpr_ref, *refs.sgpr) : *refs.sgpr;
      const bool owner_uses_dynamic_stack =
          owner_kernel->second->uses_dynamic_stack.value_or(false);
      context.uses_dynamic_stack |= owner_uses_dynamic_stack;
      context.every_owner_uses_dynamic_stack &= owner_uses_dynamic_stack;
      context.every_owner_can_grow_ordinary_vgprs &= allocated_vgprs == unified_vgprs;
      first_owner = false;
    }
    if (!complete) {
      (void)publish_sc_access_rejection(
          result.coverage_ledger, result.errors, site.physical_id,
          ConSanLoweringOutcomeKind::ResourceRejected,
          "SuperCollider LDS lowerer found no complete planning context");
      return;
    }
    if (max_owner_vgpr_ref && *max_owner_vgpr_ref < 255u)
      context.min_auto_scratch_vgpr = static_cast<uint16_t>(*max_owner_vgpr_ref + 1u);
    if (max_owner_sgpr_ref && *max_owner_sgpr_ref + 1u < REGISTER_SET_ALLOCATABLE_SGPRS)
      context.min_preferred_vcc_save_sgpr = static_cast<uint16_t>(*max_owner_sgpr_ref + 1u);
    site_contexts.push_back(std::move(context));
  };
  for (const auto &[file_offset, site] : admitted_sites) {
    (void)file_offset;
    add_site_context(*site, site->execution_and_container_owner_descriptor_file_offsets());
  }

  if (options.max_patches_is_expert_limit && options.max_patches == 0u) {
    result.warnings.emplace_back(
        "ConSan LDS check/trap proof skipped: shared patch budget consumed");
    return;
  }

  std::vector<LdsCheckTrapCandidate> candidates;
  size_t supported_candidate_count = 0;
  size_t scratchable_candidate_count = 0;
  size_t s_clause_blocked_candidate_count = 0;
  size_t missing_width_plan_site_count = 0;
  size_t missing_instruction_site_count = 0;
  size_t missing_vcc_save_site_count = 0;
  size_t missing_scratch_site_count = 0;
  size_t invalid_scratch_allocation_site_count = 0;
  size_t dynamic_stack_private_spill_rejection_count = 0;
  size_t private_spill_encoding_failure_count = 0;
  // Candidate discovery records immutable requirements only. Stable private
  // slots are assigned transactionally when a concrete placement is selected,
  // so rejected candidates cannot inflate any descriptor.
  std::unordered_map<const ConSanAccessInventorySite *, ScBracketLocalPrivateSpillPlan>
      private_spill_plans;
  for (const LdsSitePlanningContext &site_context : site_contexts) {
    if (!options.test_kernel_name_filter.empty() || !options.kernel_name_allowlist.empty()) {
      const bool container_matches =
          consan_container_selected(options, site_context.container_name);
      const bool owner_matches =
          !site_context.owner_descriptor_file_offsets.empty() &&
          (options.kernel_name_allowlist.empty()
               ? std::ranges::any_of(site_context.owner_descriptor_file_offsets,
                                     [&](uint64_t owner) {
                                       const auto kernel = kernels_by_descriptor.find(owner);
                                       return kernel != kernels_by_descriptor.end() &&
                                              consan_container_selected(options,
                                                                        kernel->second->name);
                                     })
               : std::ranges::all_of(
                     site_context.owner_descriptor_file_offsets, [&](uint64_t owner) {
                       const auto kernel = kernels_by_descriptor.find(owner);
                       return kernel != kernels_by_descriptor.end() &&
                              consan_container_selected(options, kernel->second->name);
                     }));
      if (!container_matches && !owner_matches)
        continue;
    }
    if (site_context.site == nullptr || site_context.owner_contexts.empty())
      continue;
    const KernelBlockScope kernel_scope(site_context.owner_contexts.front()->scope.blocks);
    std::vector<const LivenessAnalysis *> owner_liveness;
    owner_liveness.reserve(site_context.owner_contexts.size());
    for (const ScReservoirOwnerContext *owner : site_context.owner_contexts)
      owner_liveness.push_back(owner->liveness.get());
    const auto max_auto_scratch_vgpr = site_context.common_allocated_vgpr_count;
    const auto allocated_sgpr_count = site_context.common_allocated_sgpr_count;
    const auto min_auto_scratch_vgpr = site_context.min_auto_scratch_vgpr;
    const auto min_preferred_vcc_save_sgpr = site_context.min_preferred_vcc_save_sgpr;
    {
      const ConSanAccessInventorySite &site = *site_context.site;
      const ConSanAccessLoweringForm &form = *site.lowering.form;
      ++supported_candidate_count;
      const std::optional<uint16_t> selectable_vgpr_bank_mode =
          target.has_selectable_vgpr_bank &&
                  site.file_offset >= site.physical_id.original_text_offset
              ? consan_selectable_vgpr_bank_mode_at(
                    arch, original_bytes, site.file_offset - site.physical_id.original_text_offset,
                    site_context.container_entry_text_offset, site.file_offset)
              : std::nullopt;
      auto required_vgprs = access_dword_count(site);
      if (!required_vgprs) {
        ++missing_width_plan_site_count;
        continue;
      }
      const auto split_two_address =
          split_two_address_lds_expansion_plan(original_bytes, site, target);
      if (target.requires_split_two_address_lds_relocation && site.lowering.form &&
          site.lowering.form->kind == ConSanAccessLoweringFormKind::NativeTwoRange &&
          !split_two_address) {
        ++missing_width_plan_site_count;
        continue;
      }
      const uint32_t report_alignment_padding =
          options.report_buffer_address && target.requires_even_vgpr_tuples ? 1u : 0u;
      const Instruction *site_inst = site_context.instruction;
      if (site_inst == nullptr) {
        ++missing_instruction_site_count;
        continue;
      }
      // The complete physical-site owner set is known here, so a dead scalar
      // above one owner's current allocation is safe when every owner
      // descriptor is grown transactionally. Liveness across all owners,
      // rather than the smallest preexisting descriptor, defines the reusable
      // register set.
      constexpr uint16_t dead_vcc_selection_limit = REGISTER_SET_ALLOCATABLE_SGPRS;
      auto vcc_save =
          allocated_sgpr_count
              ? choose_sc_dead_vcc_save_sgpr(site_inst, owner_liveness, *allocated_sgpr_count,
                                             min_preferred_vcc_save_sgpr, dead_vcc_selection_limit,
                                             target, options)
              : std::nullopt;
      bool spill_vcc_save_to_vgpr = false;
      if (!vcc_save && consan_arch_is_rdna4_or_cdna5(target.arch) && allocated_sgpr_count) {
        vcc_save = choose_sc_borrowed_scalar_sgpr(*allocated_sgpr_count, options);
        spill_vcc_save_to_vgpr = vcc_save.has_value();
      }
      if (!vcc_save) {
        ++missing_vcc_save_site_count;
        continue;
      }
      const uint16_t vcc_save_sgpr_width = sc_vcc_save_sgpr_width(target);
      const bool reuse_readback_for_report =
          lds_report_action_reuses_readback(options, static_cast<uint16_t>(*required_vgprs));
      const uint32_t probe_scratch_vgprs =
          *required_vgprs + lds_check_trap_extra_scratch_vgprs(site, split_two_address) +
          (reuse_readback_for_report
               ? 0u
               : report_alignment_padding + report_action_scratch_vgprs(options)) +
          (lds_load_clobbers_address(site) ? 1u : 0u);
      const uint32_t scratch_vgprs = probe_scratch_vgprs + (spill_vcc_save_to_vgpr ? 1u : 0u);
      const auto choose_context_scratch =
          [&](std::optional<uint16_t> allocation_limit) -> std::optional<uint16_t> {
        if (options.scratch_vgpr) {
          return choose_scratch_vgpr(site, options, site_inst, owner_liveness.front(),
                                     min_auto_scratch_vgpr, allocation_limit, scratch_vgprs);
        }
        // A shared helper can reuse any run proved dead in every owner. An
        // owner-wide maximum-reference floor would discard those common holes
        // and needlessly grow every descriptor.
        RegisterSet live_before_all_owners;
        for (const LivenessAnalysis *liveness : owner_liveness)
          live_before_all_owners |= liveness->live_before(*site_inst);

        ConSanRegisterRequest request;
        request.reg_class = RegClass::VGPR;
        request.count = scratch_vgprs;
        request.alignment = access_scratch_tuple_base_is_valid(site, 1u) ? 1u : 2u;
        request.search_start = site_context.owner_descriptor_file_offsets.size() > 1u
                                   ? 0u
                                   : min_auto_scratch_vgpr.value_or(0u);
        // RegisterSet also represents selectable target banks, while these
        // probe operands directly encode only the ordinary low VGPR bank.
        request.current_allocation_count =
            std::min(allocation_limit.value_or(kConSanOrdinaryVgprLimit), kConSanOrdinaryVgprLimit);
        request.architecture_limit = request.current_allocation_count;
        request.allow_spill = false;
        if (form.address_vgpr)
          request.forbidden.expand(
              {RegClass::VGPR, *form.address_vgpr, static_cast<uint8_t>(form.address_vgpr_count)});
        if (form.destination_vgpr)
          request.forbidden.expand({RegClass::VGPR, *form.destination_vgpr,
                                    static_cast<uint8_t>(form.destination_register_count)});
        const uint8_t data_vgprs = static_cast<uint8_t>(
            form.second_data_vgpr ? form.element_register_count : form.data_register_count);
        if (form.data_vgpr)
          request.forbidden.expand({RegClass::VGPR, *form.data_vgpr, data_vgprs});
        if (form.second_data_vgpr) {
          request.forbidden.expand({RegClass::VGPR, *form.second_data_vgpr, data_vgprs});
        }

        const ConSanRegisterPlan plan = plan_consan_registers(request, live_before_all_owners);
        return plan.source == ConSanRegisterAllocationSource::LivenessDead ? plan.base
                                                                           : std::nullopt;
      };
      uint16_t scratch_vgpr = 0;
      bool scratch_selected = false;
      if (const auto candidate = choose_context_scratch(max_auto_scratch_vgpr)) {
        scratch_vgpr = *candidate;
        scratch_selected = true;
      }
      if (!scratch_selected && !options.scratch_vgpr && max_auto_scratch_vgpr &&
          site_context.every_owner_can_grow_ordinary_vgprs) {
        if (const auto candidate = choose_context_scratch(std::nullopt)) {
          scratch_vgpr = *candidate;
          scratch_selected = true;
        }
      }
      if (!scratch_selected && !options.scratch_vgpr && max_auto_scratch_vgpr) {
        if (site_context.uses_dynamic_stack) {
          std::optional<uint16_t> saved_frame_base_sgpr;
          std::optional<uint16_t> saved_scc_sgpr;
          const auto scalar_is_available = [&](uint16_t sgpr) {
            if (sgpr == kDynamicStackTopSgpr || sgpr == kDynamicFrameBaseSgpr ||
                (sgpr >= *vcc_save && sgpr < *vcc_save + vcc_save_sgpr_width))
              return false;
            const RegisterRef reg{RegClass::SGPR, sgpr, 1u};
            return std::ranges::all_of(owner_liveness, [&](const LivenessAnalysis *liveness) {
              return !liveness->any_live_before(*site_inst, reg);
            });
          };
          if (site_context.every_owner_uses_dynamic_stack) {
            for (uint16_t sgpr = 0u; sgpr < REGISTER_SET_ALLOCATABLE_SGPRS; ++sgpr) {
              if (!scalar_is_available(sgpr))
                continue;
              if (!saved_frame_base_sgpr) {
                saved_frame_base_sgpr = sgpr;
              } else {
                saved_scc_sgpr = sgpr;
                break;
              }
            }
          }
          const auto spill_scratch =
              choose_spill_scratch_vgpr(site, *max_auto_scratch_vgpr, scratch_vgprs);
          const auto private_limit = consan_address_free_private_limit(arch);
          if (spill_scratch && private_limit && saved_frame_base_sgpr && saved_scc_sgpr) {
            scratch_vgpr = *spill_scratch;
            auto spill = build_dynamic_stack_vgpr_spill_sequence(
                scratch_vgpr, static_cast<uint16_t>(scratch_vgprs), kDynamicStackTopSgpr,
                kDynamicFrameBaseSgpr, *saved_frame_base_sgpr, *saved_scc_sgpr, arch,
                /*additional_frame_bytes=*/0u);
            const uint64_t requested_private_bytes =
                static_cast<uint64_t>(site_context.max_owner_private_segment_size) +
                static_cast<uint64_t>(scratch_vgprs) * SpillManager::kSlotBytes;
            const auto required_private_bytes =
                requested_private_bytes <= std::numeric_limits<uint32_t>::max()
                    ? consan_normalize_address_free_private_size(
                          arch, static_cast<uint32_t>(requested_private_bytes))
                    : std::nullopt;
            if (spill && required_private_bytes && *required_private_bytes <= *private_limit) {
              spill->total_private_bytes = *required_private_bytes;
              private_spill_plans.insert_or_assign(
                  &site,
                  ScBracketLocalPrivateSpillPlan{
                      .owner_descriptor_file_offsets = site_context.owner_descriptor_file_offsets,
                      .original_private_bytes = site_context.max_owner_private_segment_size,
                      .private_limit = *private_limit,
                      .vgpr_base = scratch_vgpr,
                      .vgpr_count = static_cast<uint16_t>(scratch_vgprs),
                      .uses_dynamic_stack_frame = true,
                      .saved_frame_base_sgpr = *saved_frame_base_sgpr,
                      .saved_scc_sgpr = *saved_scc_sgpr,
                  });
              scratch_selected = true;
            } else {
              ++private_spill_encoding_failure_count;
            }
          } else {
            ++dynamic_stack_private_spill_rejection_count;
          }
        } else {
          const auto spill_scratch =
              choose_spill_scratch_vgpr(site, *max_auto_scratch_vgpr, scratch_vgprs);
          const auto private_limit = consan_address_free_private_limit(arch);
          if (spill_scratch && private_limit) {
            scratch_vgpr = *spill_scratch;
            SpillManager preview_manager(site_context.max_owner_private_segment_size,
                                         *private_limit);
            auto spill =
                build_vgpr_spill_sequence(preview_manager, scratch_vgpr, scratch_vgprs, arch);
            if (spill) {
              private_spill_plans.insert_or_assign(
                  &site,
                  ScBracketLocalPrivateSpillPlan{
                      .owner_descriptor_file_offsets = site_context.owner_descriptor_file_offsets,
                      .original_private_bytes = site_context.max_owner_private_segment_size,
                      .private_limit = *private_limit,
                      .vgpr_base = scratch_vgpr,
                      .vgpr_count = static_cast<uint16_t>(scratch_vgprs),
                  });
              scratch_selected = true;
            } else {
              ++private_spill_encoding_failure_count;
            }
          } else {
            ++private_spill_encoding_failure_count;
          }
        }
      }
      if (!scratch_selected) {
        ++missing_scratch_site_count;
        continue;
      }
      auto required_allocation =
          required_descriptor_vgpr_allocation_for_scratch(site, scratch_vgpr, scratch_vgprs);
      if (!required_allocation) {
        ++invalid_scratch_allocation_site_count;
        continue;
      }
      std::optional<uint16_t> descriptor_growth;
      if (max_auto_scratch_vgpr && *required_allocation > *max_auto_scratch_vgpr)
        descriptor_growth = *required_allocation;
      std::optional<ConSanSuperColliderScalarVccSpill> scalar_vcc_spill;
      if (spill_vcc_save_to_vgpr) {
        scalar_vcc_spill = ConSanSuperColliderScalarVccSpill{
            .vcc_save_sgpr = *vcc_save,
            .reservoir_vgpr = static_cast<uint16_t>(scratch_vgpr + probe_scratch_vgprs),
            .reservoir = ConSanSuperColliderScalarVccReservoir::PackedLanes,
        };
      }
      ++scratchable_candidate_count;
      if (text_offset_is_inside_s_clause(kernel_scope,
                                             site.physical_id.original_text_offset)) {
        ++s_clause_blocked_candidate_count;
        continue;
      }
      LdsCheckTrapCandidate candidate;
      candidate.site = &site;
      candidate.scratch_vgpr = scratch_vgpr;
      candidate.vcc_save_sgpr = *vcc_save;
      candidate.required_vgpr_allocation_count = descriptor_growth;
      candidate.selectable_vgpr_bank_mode = selectable_vgpr_bank_mode;
      candidate.scalar_vcc_spill = scalar_vcc_spill;
      candidates.push_back(std::move(candidate));
    }
  }
  auto candidate_order = [](const LdsCheckTrapCandidate &lhs, const LdsCheckTrapCandidate &rhs) {
    const bool lhs_grows_descriptor = lhs.required_vgpr_allocation_count.has_value();
    const bool rhs_grows_descriptor = rhs.required_vgpr_allocation_count.has_value();
    if (lhs_grows_descriptor != rhs_grows_descriptor)
      return !lhs_grows_descriptor;
    return lhs.site->file_offset < rhs.site->file_offset;
  };
  std::ranges::sort(candidates, candidate_order);

  const size_t max_patches =
      options.max_patches_is_expert_limit ? options.max_patches : supported_candidate_count;
  std::vector<LdsCheckTrapCandidate> selected_candidates;
  selected_candidates.reserve(std::min(max_patches, supported_candidate_count));
  std::unordered_set<const ConSanAccessInventorySite *> selected_sites;
  struct ScSelectedSpillState {
    // Within this bracket-local lifetime domain, an exact owner set shares
    // stable register-to-slot mappings while distinct sets may reuse offsets.
    std::map<std::vector<uint64_t>, SpillManager> managers;
    std::unordered_map<const ConSanAccessInventorySite *, VgprSpillSequence> sequences;
  };
  ScSelectedSpillState selected_spills;
  std::unordered_set<const ConSanAccessInventorySite *> selection_spill_failure_sites;
  const auto try_add_selected_spill = [&](ScSelectedSpillState &state,
                                          const LdsCheckTrapCandidate &candidate) {
    const auto plan = private_spill_plans.find(candidate.site);
    if (plan == private_spill_plans.end())
      return true;
    const std::vector<uint64_t> &key = plan->second.owner_descriptor_file_offsets;
    if (plan->second.uses_dynamic_stack_frame) {
      auto sequence = build_dynamic_stack_vgpr_spill_sequence(
          plan->second.vgpr_base, plan->second.vgpr_count, kDynamicStackTopSgpr,
          kDynamicFrameBaseSgpr, plan->second.saved_frame_base_sgpr, plan->second.saved_scc_sgpr,
          arch, /*additional_frame_bytes=*/0u);
      const uint64_t requested_private_bytes =
          static_cast<uint64_t>(plan->second.original_private_bytes) +
          static_cast<uint64_t>(plan->second.vgpr_count) * SpillManager::kSlotBytes;
      const auto required_private_bytes =
          requested_private_bytes <= std::numeric_limits<uint32_t>::max()
              ? consan_normalize_address_free_private_size(
                    arch, static_cast<uint32_t>(requested_private_bytes))
              : std::nullopt;
      if (!sequence || !required_private_bytes ||
          *required_private_bytes > plan->second.private_limit)
        return false;
      sequence->total_private_bytes = *required_private_bytes;
      state.sequences.insert_or_assign(candidate.site, std::move(*sequence));
      return true;
    }
    std::optional<SpillManager> manager;
    if (const auto existing = state.managers.find(key); existing != state.managers.end())
      manager = existing->second;
    else
      manager.emplace(plan->second.original_private_bytes, plan->second.private_limit);
    auto sequence =
        build_vgpr_spill_sequence(*manager, plan->second.vgpr_base, plan->second.vgpr_count, arch);
    if (!sequence)
      return false;
    state.managers.insert_or_assign(key, std::move(*manager));
    state.sequences.insert_or_assign(candidate.site, std::move(*sequence));
    return true;
  };
  const auto note_selection_spill_failure = [&](const ConSanAccessInventorySite *site) {
    if (selection_spill_failure_sites.insert(site).second)
      ++private_spill_encoding_failure_count;
  };
  for (const LdsCheckTrapCandidate &candidate : candidates) {
    if (selected_candidates.size() >= max_patches)
      break;
    if (!try_add_selected_spill(selected_spills, candidate)) {
      note_selection_spill_failure(candidate.site);
      continue;
    }
    selected_sites.insert(candidate.site);
    selected_candidates.push_back(candidate);
  }

  if (!selected_candidates.empty() && selected_candidates.size() < supported_candidate_count &&
      max_patches >= supported_candidate_count) {
    result.warnings.emplace_back(
        "ConSan LDS check/trap proof selected only " + std::to_string(selected_candidates.size()) +
        "/" + std::to_string(supported_candidate_count) +
        " supported sites; scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " missing_width_plan_sites=" + std::to_string(missing_width_plan_site_count) +
        " missing_instruction_sites=" + std::to_string(missing_instruction_site_count) +
        " missing_vcc_save_sites=" + std::to_string(missing_vcc_save_site_count) +
        " missing_scratch_sites=" + std::to_string(missing_scratch_site_count) +
        " invalid_scratch_allocation_sites=" +
        std::to_string(invalid_scratch_allocation_site_count) +
        " dynamic_stack_private_spill_rejections=" +
        std::to_string(dynamic_stack_private_spill_rejection_count) +
        " private_spill_encoding_failures=" +
        std::to_string(private_spill_encoding_failure_count));
  }

  if (selected_candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan LDS check/trap proof found no resource-feasible "
        "ds_load_b{32,64,128}, ds_load_tr{8_b64,16_b128}, ds_load_2addr_b{32,64}, "
        "ds_load_{i,u}{8,16}, ds_load_{i,u}8_d16(_hi), ds_load_u16_d16(_hi), or "
        "ds_read_{i,u}8_d16(_hi), ds_read_u16_d16(_hi), "
        "ds_store_b{32,64,96,128}, or ds_store_2addr_b{32,64} site; supported_candidates=" +
        std::to_string(supported_candidate_count) +
        " scratchable_candidates=" + std::to_string(scratchable_candidate_count) +
        " missing_width_plan_sites=" + std::to_string(missing_width_plan_site_count) +
        " missing_instruction_sites=" + std::to_string(missing_instruction_site_count) +
        " missing_vcc_save_sites=" + std::to_string(missing_vcc_save_site_count) +
        " missing_scratch_sites=" + std::to_string(missing_scratch_site_count) +
        " invalid_scratch_allocation_sites=" +
        std::to_string(invalid_scratch_allocation_site_count) +
        " dynamic_stack_private_spill_rejections=" +
        std::to_string(dynamic_stack_private_spill_rejection_count) +
        " private_spill_encoding_failures=" + std::to_string(private_spill_encoding_failure_count) +
        " s_clause_blocked_candidates=" + std::to_string(s_clause_blocked_candidate_count));
    return;
  }

  struct PlannedLdsCheckTrapPatch {
    const ConSanAccessInventorySite *site = nullptr;
    std::vector<ConSanProbeIntentId> intent_ids;
    std::vector<uint32_t> words;
    std::optional<uint32_t> guest_instruction_word_offset;
    ConSanPatchInfo info;
    std::optional<uint16_t> required_vgpr_allocation_count;
  };

  std::vector<PlannedLdsCheckTrapPatch> planned_patches;
  planned_patches.reserve(selected_candidates.size());
  for (const LdsCheckTrapCandidate &candidate : selected_candidates) {
    const ConSanAccessInventorySite &site = *candidate.site;
    const auto split_two_address =
        split_two_address_lds_expansion_plan(original_bytes, site, target);
    std::optional<uint32_t> guest_instruction_word_offset;
    auto words = build_lds_check_trap_words(original_bytes, site, target, options,
                                            candidate.scratch_vgpr, candidate.vcc_save_sgpr,
                                            candidate.selectable_vgpr_bank_mode.value_or(0u),
                                            guest_instruction_word_offset, result.errors);
    if (!words)
      return;
    const auto spill = selected_spills.sequences.find(&site);
    std::optional<std::array<uint32_t, 2>> scalar_vcc_save_words;
    std::optional<std::array<uint32_t, 2>> scalar_vcc_restore_words;
    if (candidate.scalar_vcc_spill) {
      scalar_vcc_save_words = instrumentation::build_v_writelane_b32(
          candidate.scalar_vcc_spill->reservoir_vgpr, candidate.scalar_vcc_spill->vcc_save_sgpr, 0u,
          arch);
      scalar_vcc_restore_words = instrumentation::build_v_readlane_b32(
          candidate.scalar_vcc_spill->vcc_save_sgpr, candidate.scalar_vcc_spill->reservoir_vgpr, 0u,
          arch);
      if (!scalar_vcc_save_words || !scalar_vcc_restore_words) {
        result.errors.emplace_back(
            "ConSan LDS check/trap could not encode its scalar VCC-save spill");
        return;
      }
    }
    if (candidate.selectable_vgpr_bank_mode.value_or(0u) != 0u) {
      const auto select_low = instrumentation::build_s_set_vgpr_msb_transition(
          static_cast<uint8_t>(*candidate.selectable_vgpr_bank_mode), 0u, arch);
      const auto restore_guest = instrumentation::build_s_set_vgpr_msb_transition(
          0u, static_cast<uint8_t>(*candidate.selectable_vgpr_bank_mode), arch);
      const size_t original_access_words =
          split_two_address ? split_two_address->access_word_count() : 2u;
      const size_t guest_prefix_words =
          original_access_words + (lds_load_clobbers_address(site) ? 1u : 0u);
      if (!select_low || !restore_guest || words->size() < guest_prefix_words) {
        result.errors.emplace_back(
            "ConSan LDS check/trap proof could not preserve selectable VGPR-bank mode");
        return;
      }
      std::vector<uint32_t> wrapped;
      const bool has_spill = spill != selected_spills.sequences.end();
      wrapped.reserve(
          words->size() +
          (has_spill ? spill->second.save_words.size() + spill->second.restore_words.size() + 4u
                     : 2u) +
          (scalar_vcc_save_words ? 6u : 0u));
      if (has_spill || scalar_vcc_save_words) {
        wrapped.push_back(*select_low);
        if (has_spill)
          wrapped.insert(wrapped.end(), spill->second.save_words.begin(),
                         spill->second.save_words.end());
        if (scalar_vcc_save_words)
          wrapped.insert(wrapped.end(), scalar_vcc_save_words->begin(),
                         scalar_vcc_save_words->end());
        wrapped.push_back(*restore_guest);
      }
      if (lds_load_clobbers_address(site)) {
        if (guest_instruction_word_offset) {
          if (*guest_instruction_word_offset == 0u) {
            result.errors.emplace_back(
                "ConSan LDS check/trap proof lost its preserved-address guest offset");
            return;
          }
          guest_instruction_word_offset =
              static_cast<uint32_t>(wrapped.size() + 3u + *guest_instruction_word_offset - 1u);
        }
        wrapped.push_back(*select_low);
        wrapped.push_back(words->front());
        wrapped.push_back(*restore_guest);
        wrapped.insert(wrapped.end(), words->begin() + 1u, words->begin() + guest_prefix_words);
      } else {
        if (guest_instruction_word_offset) {
          guest_instruction_word_offset =
              static_cast<uint32_t>(wrapped.size() + *guest_instruction_word_offset);
        }
        wrapped.insert(wrapped.end(), words->begin(), words->begin() + guest_prefix_words);
      }
      wrapped.push_back(*select_low);
      wrapped.insert(wrapped.end(), words->begin() + guest_prefix_words, words->end());
      if (scalar_vcc_restore_words)
        wrapped.insert(wrapped.end(), scalar_vcc_restore_words->begin(),
                       scalar_vcc_restore_words->end());
      if (has_spill)
        wrapped.insert(wrapped.end(), spill->second.restore_words.begin(),
                       spill->second.restore_words.end());
      wrapped.push_back(*restore_guest);
      words = std::move(wrapped);
    } else if (spill != selected_spills.sequences.end()) {
      std::vector<uint32_t> wrapped;
      wrapped.reserve(spill->second.save_words.size() + words->size() +
                      spill->second.restore_words.size() + (scalar_vcc_save_words ? 4u : 0u));
      wrapped.insert(wrapped.end(), spill->second.save_words.begin(),
                     spill->second.save_words.end());
      if (scalar_vcc_save_words)
        wrapped.insert(wrapped.end(), scalar_vcc_save_words->begin(), scalar_vcc_save_words->end());
      if (guest_instruction_word_offset) {
        guest_instruction_word_offset =
            static_cast<uint32_t>(wrapped.size() + *guest_instruction_word_offset);
      }
      wrapped.insert(wrapped.end(), words->begin(), words->end());
      if (scalar_vcc_restore_words)
        wrapped.insert(wrapped.end(), scalar_vcc_restore_words->begin(),
                       scalar_vcc_restore_words->end());
      wrapped.insert(wrapped.end(), spill->second.restore_words.begin(),
                     spill->second.restore_words.end());
      words = std::move(wrapped);
    } else if (scalar_vcc_save_words && scalar_vcc_restore_words) {
      std::vector<uint32_t> wrapped;
      wrapped.reserve(words->size() + 4u);
      wrapped.insert(wrapped.end(), scalar_vcc_save_words->begin(), scalar_vcc_save_words->end());
      if (guest_instruction_word_offset) {
        guest_instruction_word_offset =
            static_cast<uint32_t>(wrapped.size() + *guest_instruction_word_offset);
      }
      wrapped.insert(wrapped.end(), words->begin(), words->end());
      wrapped.insert(wrapped.end(), scalar_vcc_restore_words->begin(),
                     scalar_vcc_restore_words->end());
      words = std::move(wrapped);
    }

    PlannedLdsCheckTrapPatch planned;
    planned.site = &site;
    planned.intent_ids = result.coverage_ledger.pending_intent_ids(
        ConSanProbeIntentKind::RedundantAccessObservation, site.physical_id);
    if (planned.intent_ids.empty()) {
      result.errors.emplace_back(
          "ConSan LDS check/trap proof selected a site without a pending semantic intent");
      return;
    }
    planned.words = std::move(*words);
    planned.guest_instruction_word_offset = guest_instruction_word_offset;
    planned.required_vgpr_allocation_count = candidate.required_vgpr_allocation_count;
    planned.info.kind = lds_check_trap_patch_kind(site, /*use_local_cave=*/true);
    planned.info.anchor_offset = site.physical_id.original_text_offset;
    planned.info.original_size = site.instruction_size;
    planned.info.trampoline_size =
        static_cast<uint32_t>(planned.words.size() * sizeof(uint32_t));
    planned.info.scratch_vgpr = candidate.scratch_vgpr;
    planned.info.sc_scalar_vcc_spill = candidate.scalar_vcc_spill;
    planned.info.required_sgpr_count =
        static_cast<uint16_t>(candidate.vcc_save_sgpr + sc_vcc_save_sgpr_width(target));
    if (const auto plan = private_spill_plans.find(&site);
        plan != private_spill_plans.end() && plan->second.uses_dynamic_stack_frame) {
      planned.info.required_sgpr_count = std::max<uint16_t>(
          planned.info.required_sgpr_count,
          static_cast<uint16_t>(
              std::max(plan->second.saved_frame_base_sgpr, plan->second.saved_scc_sgpr) + 1u));
    }
    planned.info.owner_descriptor_file_offsets =
        site.execution_and_container_owner_descriptor_file_offsets();
    if (spill != selected_spills.sequences.end()) {
      planned.info.spilled_vgpr_count = spill->second.vgpr_count;
      planned.info.required_private_segment_size = spill->second.total_private_bytes;
      if (spill->second.uses_dynamic_stack_frame)
        planned.info.dynamic_private_segment_addend = spill->second.dynamic_frame_bytes;
    }
    if (planned.guest_instruction_word_offset) {
      planned.info.relocated_guest_instruction_offset =
          *planned.guest_instruction_word_offset * sizeof(uint32_t);
    }

    planned_patches.push_back(std::move(planned));
  }

  ConSanDescriptorRegisterRequirements descriptor_growths;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if (planned.required_vgpr_allocation_count) {
      for (uint64_t owner : planned.info.owner_descriptor_file_offsets)
        note_maximum_descriptor_extent(descriptor_growths, owner,
                                       *planned.required_vgpr_allocation_count);
    }
  }
  ConSanDescriptorRegisterRequirements descriptor_sgpr_growths;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    const uint16_t required = planned.info.required_sgpr_count;
    if (required != 0)
      for (uint64_t owner : planned.info.owner_descriptor_file_offsets)
        note_maximum_descriptor_extent(descriptor_sgpr_growths, owner, required);
  }
  std::unordered_map<uint64_t, uint32_t> descriptor_private_requirements;
  const ConSanDescriptorMemoryRequirements no_group_segment_requirements;
  for (const PlannedLdsCheckTrapPatch &planned : planned_patches) {
    if (planned.info.required_private_segment_size != 0) {
      for (uint64_t owner : planned.info.owner_descriptor_file_offsets) {
        uint32_t &required = descriptor_private_requirements[owner];
        required = std::max(required, planned.info.required_private_segment_size);
      }
    }
  }
  std::vector<ConSanStagedTextRewrite> rewrites;
  rewrites.reserve(planned_patches.size());
  for (PlannedLdsCheckTrapPatch &planned : planned_patches) {
    rewrites.push_back({
        .original_site = planned.site->physical_id,
        .intent_ids = std::move(planned.intent_ids),
        .words = std::move(planned.words),
        .patch = std::move(planned.info),
    });
  }
  (void)stage_consan_text_rewrites(
      code_object, arch,
      {descriptor_growths, descriptor_sgpr_growths, descriptor_private_requirements,
       no_group_segment_requirements},
      {.maximum_ordinary_vgpr_count = REGISTER_SET_MAX_VGPRS,
       .inventory_proves_empty_accumulator_bank = false,
       .maximum_group_segment_bytes = std::nullopt},
      "ConSan LDS check/trap proof", std::move(rewrites), result);
}
