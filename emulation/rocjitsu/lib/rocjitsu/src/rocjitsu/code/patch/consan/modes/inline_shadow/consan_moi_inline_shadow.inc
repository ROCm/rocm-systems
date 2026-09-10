void try_apply_inline_shadow_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   const ConSanMoiOperatingPoint &operating_point,
                                   rj_code_arch_t arch,
                                   std::span<const ConSanMoiCandidate> admitted,
                                   const MoiObjectModeSemantics &semantics,
                                   ConSanTransformArtifacts &result) {
  if (!options.moi_report_buffer_address)
    return;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe does not support this architecture");
    return;
  }

  const ConSanMoiReportBufferLayout &layout = semantics.report_layout;
  if (options.max_patches == 0u)
    return;
  if (!validate_inline_shadow_exec_save_sgpr(
          project_inline_shadow_scalar_state(operating_point), semantics.inline_access_present,
          moi_exec_save_sgpr_count(
              resolve_moi_exec_save_requirement(options, options, operating_point, semantics),
              arch),
          arch, result.warnings))
    return;
  if (!operating_point.moi_owner_epoch_vgprs.owner() &&
      result.moi_operating_point.owner_persistent_vgprs.empty() &&
      !operating_point.moi_persistent_sgprs.owner() &&
      !operating_point.automatic_moi_private_epoch) {
    result.warnings.emplace_back("ConSan MOI inline-shadow probe requires persistent owner state");
    return;
  }

  std::vector<const ConSanMoiCandidate *> candidates =
      consan_detail::make_moi_candidate_pointer_view(admitted);
  std::ranges::sort(candidates, [](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
    return lhs->anchor() < rhs->anchor();
  });
  if (candidates.empty()) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow probe found no supported native LDS or normalized group-flat "
        "load/store candidate");
    return;
  }

  const bool requires_wave_partitioned_exact_shadow =
      std::ranges::any_of(candidates, [&](const ConSanMoiCandidate *candidate) {
        return std::ranges::any_of(candidate->site().ranges, [](const ConSanAccessRange &range) {
          return range.byte_width < consan_moi_shadow_cell::granule_bytes;
        });
      });
  if (requires_wave_partitioned_exact_shadow) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow subword traffic uses the wave-partitioned external exact-shadow "
        "table");
  }

  struct PlannedInlineShadowFragment : MoiPlannedProbeResources {
    const ConSanMoiCandidate *candidate = nullptr;
    ConSanMoiOperatingPoint point;
    std::optional<MoiWorkitemOwnerDerivationPlan> owner_derivation;
    MoiInlineShadowEmissionPlan emission;
    uint32_t guest_instruction_word_count = 0u;
  };

  const std::span<const uint8_t> active_bytes = result.active_bytes(bytes);
  AmdGpuCodeObject code_object(active_bytes.data(), active_bytes.size());
  std::unordered_set<uint64_t> dynamic_lds_descriptors;
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    if (kernel.has_dynamic_lds)
      dynamic_lds_descriptors.insert(kernel.descriptor_file_offset);
  }
  std::unordered_map<uint64_t, std::optional<ConSanMoiOperatingPoint>> points_by_descriptor;
  std::unordered_map<uint64_t, std::optional<ConSanMoiWorkgroupShadowLayout>>
      workgroup_shadow_by_descriptor;
  MoiPrivateStateLayoutCache private_layout_cache;
  std::unordered_map<uint64_t, std::optional<uint16_t>> owner_shift_by_descriptor;
  std::unordered_map<uint64_t, std::optional<ConSanMoiWorkgroupSources>>
      workgroup_sources_by_descriptor;
  bool warned_dynamic_lds_external_shadow = false;
  MoiSpillManagers spill_managers;
  std::vector<PlannedInlineShadowFragment> planned;
  planned.reserve(candidates.size());

  for (const ConSanMoiCandidate *candidate_ptr : candidates) {
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const ConSanProgramContainer *container =
        result.program_inventory.container(candidate.site().container);
    if (container == nullptr)
      continue;
    const ConSanCandidateResourcePlan *resource_plan =
        resource_plan_for_candidate(result.resource_plans, candidate);
    if (resource_plan == nullptr)
      continue;
    const bool selected_reduced_spill_recovery =
        std::ranges::any_of(resource_plan->alternatives, [](const auto &alternative) {
          return alternative.kind ==
                     ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery &&
                 alternative.outcome == ConSanResourcePlanAlternativeOutcome::Selected;
        });
    const MoiAccessResourceFacts access_resource_facts =
        resolve_moi_access_resource_facts(operating_point, candidate, arch);
    const uint16_t expected_scratch_count =
        selected_reduced_spill_recovery
            ? inline_shadow_spill_backed_scratch_count(options.moi_track_atomics,
                                                       access_resource_facts, candidate)
            : inline_shadow_scratch_count(options.moi_track_atomics, access_resource_facts,
                                          candidate);
    const auto resources =
        resolve_moi_scratch_plan(*resource_plan, result.program_inventory, operating_point,
                                 result.moi_operating_point, expected_scratch_count);
    if (!resources)
      continue;

    const std::optional<uint64_t> direct_descriptor =
        resources->owner_descriptor_file_offsets.size() == 1u
            ? std::optional<uint64_t>(resources->owner_descriptor_file_offsets.front())
            : std::nullopt;
    ConSanMoiOperatingPoint candidate_point;
    if (direct_descriptor) {
      auto [cached, inserted] = points_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
      if (inserted) {
        ConSanMoiOperatingPoint resolved = operating_point;
        if (apply_moi_transient_sgpr_assignment(options, resolved, result.moi_operating_point,
                                                resources->owner_descriptor_file_offsets) &&
            apply_moi_persistent_vgpr_assignment(resolved, result.moi_operating_point,
                                                 resources->owner_descriptor_file_offsets)) {
          cached->second = std::move(resolved);
        }
      }
      if (!cached->second)
        continue;
      candidate_point = *cached->second;
    } else {
      candidate_point = operating_point;
      if (!apply_moi_transient_sgpr_assignment(options, candidate_point, result.moi_operating_point,
                                               resources->owner_descriptor_file_offsets) ||
          !apply_moi_persistent_vgpr_assignment(candidate_point, result.moi_operating_point,
                                                resources->owner_descriptor_file_offsets)) {
        continue;
      }
    }

    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    const bool has_unspilled_component_override = std::ranges::any_of(
        resources->owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
          const auto *assignment =
              moi_transient_sgpr_assignment(result.moi_operating_point, descriptor_offset);
          return assignment != nullptr && !assignment->spill_backed;
        });
    const bool spill_backed_scalar_state = candidate_point.has_inline_moi_scalar_spill();
    if (!requires_wave_partitioned_exact_shadow && !has_unspilled_component_override &&
        !spill_backed_scalar_state) {
      const bool has_dynamic_lds_owner = std::ranges::any_of(
          resources->owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
            return dynamic_lds_descriptors.contains(descriptor_offset);
          });
      if (!has_dynamic_lds_owner) {
        if (direct_descriptor) {
          auto [cached, inserted] =
              workgroup_shadow_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
          if (inserted) {
            cached->second = common_moi_workgroup_shadow_layout(bytes, *resources, arch, options,
                                                                result.warnings);
          }
          workgroup_shadow = cached->second;
        } else {
          workgroup_shadow =
              common_moi_workgroup_shadow_layout(bytes, *resources, arch, options, result.warnings);
        }
      }
      if (has_dynamic_lds_owner && !warned_dynamic_lds_external_shadow) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow dynamic-LDS owner uses the external exact-shadow table");
        warned_dynamic_lds_external_shadow = true;
      } else if (!workgroup_shadow) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow access falls back to the external exact-shadow table");
      }
    }

    std::optional<ConSanMoiPrivateStateLayout> private_layout;
    std::optional<MoiWorkitemOwnerDerivationPlan> owner_derivation;
    if (candidate_point.automatic_moi_private_epoch) {
      const bool persist_dispatch_id = !moi_has_runtime_hardware_dispatch_id(candidate_point) &&
                                       (candidate_point.moi_dispatch_identity.sgpr() ||
                                        candidate_point.moi_dispatch_identity.private_fallback());
      private_layout = private_layout_cache.resolve(
          direct_descriptor, result.program_inventory, *resources, arch, result.warnings,
          {.owner = options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
           .workgroup_key = true,
           .dispatch_id = persist_dispatch_id});
      if (!private_layout)
        continue;
      if (options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
        std::optional<uint16_t> owner_shift;
        if (direct_descriptor) {
          auto [cached, inserted] =
              owner_shift_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
          if (inserted) {
            cached->second =
                common_moi_workitem_owner_shift(bytes, *resources, arch, result.warnings);
          }
          owner_shift = cached->second;
        } else {
          owner_shift = common_moi_workitem_owner_shift(bytes, *resources, arch, result.warnings);
        }
        if (!owner_shift || !private_layout->owner_offset)
          continue;
        owner_derivation = MoiWorkitemOwnerDerivationPlan{
            .entry_workitem_x_private_offset = private_layout->owner_offset,
            .wave_size_shift = *owner_shift,
        };
      }
    }

    if (private_layout && private_layout->dispatch_id_offset) {
      const std::optional<uint16_t> reload_sgpr =
          candidate_point.moi_scalar_spill_setup
              ? std::optional{candidate_point.moi_scalar_spill_setup->temporaries.frame_base_sgpr}
          : candidate_point.moi_exec_save_sgpr
              ? std::optional{static_cast<uint16_t>(*candidate_point.moi_exec_save_sgpr + 12u)}
              : std::nullopt;
      if (!reload_sgpr) {
        result.warnings.emplace_back(
            "ConSan MOI private dispatch identity has no owner-local scalar reload pair");
        continue;
      }
      candidate_point.moi_dispatch_identity.set_sgpr(reload_sgpr);
    }

    std::vector<std::string> workgroup_errors;
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    const bool has_persistent_workgroup_key =
        (private_layout && private_layout->workgroup_key_offset) ||
        candidate_point.moi_workgroup_key_vgpr ||
        candidate_point.moi_persistent_sgprs.workgroup_key;
    if (has_persistent_workgroup_key) {
      workgroup_sources = ConSanMoiWorkgroupSources{};
    } else {
      std::optional<uint64_t> descriptor = direct_descriptor;
      if (!descriptor) {
        descriptor = common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      }
      if (!descriptor)
        continue;
      if (direct_descriptor) {
        auto [cached, inserted] =
            workgroup_sources_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
        if (inserted) {
          cached->second =
              moi_descriptor_workgroup_sources(bytes, *direct_descriptor, arch, workgroup_errors);
        }
        workgroup_sources = cached->second;
      } else {
        workgroup_sources =
            moi_descriptor_workgroup_sources(bytes, *descriptor, arch, workgroup_errors);
      }
    }
    if (!workgroup_sources) {
      result.warnings.insert(result.warnings.end(), workgroup_errors.begin(),
                             workgroup_errors.end());
      continue;
    }

    std::optional<VgprSpillSequence> spill;
    if (resources->source == ConSanRegisterAllocationSource::SpillRequired ||
        moi_scalar_spill_requires_dynamic_vgpr_frame(result.program_inventory, *resources,
                                                     candidate_point)) {
      if (candidate_point.moi_branch_only_spill &&
          candidate_point.moi_branch_only_spill->dynamic_stack_borrowed_sgpr) {
        constexpr uint16_t reservoir_count =
            DynamicStackBorrowedSgprSpillSequence::kScalarReservoirCount;
        const uint32_t scalar_frame_bytes =
            static_cast<uint32_t>(moi_exec_save_sgpr_count(
                resolve_moi_exec_save_requirement(options, options, candidate_point, semantics),
                arch)) *
            SpillManager::kSlotBytes;
        if (resources->count < reservoir_count) {
          result.warnings.emplace_back(
              "ConSan MOI dynamic-stack bootstrap has no scalar reservoir window");
          continue;
        }
        const uint16_t reservoir_base =
            static_cast<uint16_t>(resources->base + resources->count - reservoir_count);
        const auto borrowed = build_dynamic_stack_borrowed_sgpr_spill_sequence(
            resources->base, resources->count,
            *candidate_point.moi_branch_only_spill->dynamic_stack_borrowed_sgpr, reservoir_base,
            resources->original_private_segment_size, arch, scalar_frame_bytes);
        if (!borrowed) {
          result.warnings.emplace_back(
              "ConSan MOI could not encode the branch-only dynamic-stack bootstrap");
          continue;
        }
        spill = borrowed->as_vgpr_spill_sequence();
      } else {
        spill = build_moi_spill_sequence(
            result.program_inventory, *resources, options, options, candidate_point, semantics,
            spill_managers, arch, result.warnings,
            private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base)
                           : std::nullopt);
      }
      if (!spill)
        continue;
    }

    std::optional<SgprSpillSequence> scalar_spill;
    if (candidate_point.has_inline_moi_scalar_spill()) {
      scalar_spill = build_moi_sgpr_spill_sequence(
          result.program_inventory, *resources, options, options, candidate_point, semantics,
          spill_managers, arch, result.warnings,
          private_layout ? std::optional<uint32_t>(private_layout->ephemeral_base) : std::nullopt,
          spill ? &*spill : nullptr);
      if (!scalar_spill)
        continue;
    }

    MoiInlineShadowEmissionPlan emission{
        .report_buffer_address = *options.moi_report_buffer_address,
        .report_generation = options.moi_report_generation,
        .report_dispatch_id = options.moi_report_dispatch_id,
        .track_atomics = options.moi_track_atomics,
        .source_in_kernel = container->is_kernel(),
        .byte_granular_external_shadow = requires_wave_partitioned_exact_shadow,
        .spill_overlaps_guest_operands = spill.has_value(),
        .required_exec_save_sgpr_count = moi_exec_save_sgpr_count(
            resolve_moi_exec_save_requirement(options, options, candidate_point, semantics), arch),
        .scratch_vgpr = resources->base,
        .scratch_count = resources->count,
        .scalar_state = project_inline_shadow_scalar_state(candidate_point),
        .scalar_abi =
            plan_moi_scalar_abi(options.moi_engine, moi_scalar_preservation_state(candidate_point)),
        .dispatch_id =
            consan_moi_detail::moi_target_dispatch_id_sources({candidate_point, options}, arch),
        .owner_field =
            {
                .owner_vgpr = candidate_point.moi_owner_epoch_vgprs.owner(),
                .persistent_owner_sgpr = candidate_point.moi_persistent_sgprs.owner(),
                .automatic_private_epoch = candidate_point.automatic_moi_private_epoch,
                .private_owner_source = options.moi_owner_source,
                .resident_wave_owner_sgpr = candidate_point.moi_owner_sgpr.base(),
                .borrowed_resident_wave_owner_sgpr = candidate_point.moi_owner_sgpr.automatic(),
            },
        .epoch_field =
            {
                .epoch_vgpr = candidate_point.moi_owner_epoch_vgprs.epoch(),
                .persistent_epoch_sgpr = candidate_point.moi_persistent_sgprs.epoch(),
                .automatic_private_epoch = candidate_point.automatic_moi_private_epoch,
            },
        .workgroup_key_registers = moi_workgroup_key_register_plan(candidate_point),
        .access_resource_facts = access_resource_facts,
        .workgroup_sources = *workgroup_sources,
        .workgroup_shadow = workgroup_shadow,
        .private_epoch_offset =
            private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        .owner_derivation = owner_derivation,
        .private_workgroup_key_offset =
            private_layout ? private_layout->workgroup_key_offset : std::nullopt,
        .private_dispatch_id_offset =
            private_layout ? private_layout->dispatch_id_offset : std::nullopt,
    };
    uint32_t guest_instruction_word_count = 0u;
    std::vector<std::string> candidate_errors;
    if (!build_inline_shadow_words(bytes, candidate, emission, arch, layout,
                                   spill ? &*spill : nullptr, candidate_errors,
                                   &guest_instruction_word_count)) {
      for (std::string &error : candidate_errors) {
        error += " for " + std::string(candidate.site().mnemonic_view()) + " at text offset " +
                 std::to_string(candidate.anchor()) + " with scratch v" +
                 std::to_string(resources->base) + ":v" +
                 std::to_string(resources->base + resources->count - 1u);
      }
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint32_t required_private_bytes =
        std::max({private_layout ? private_layout->ephemeral_base : 0u,
                  spill ? spill->total_private_bytes : 0u,
                  scalar_spill ? scalar_spill->total_private_bytes : 0u});
    MoiPlannedProbeResources probe{
        .resources = *resources,
        .spill = std::move(spill),
        .scalar_spill = std::move(scalar_spill),
        .private_layout = std::move(private_layout),
        .required_private_bytes = required_private_bytes,
    };
    planned.push_back({std::move(probe), candidate_ptr, std::move(candidate_point),
                       std::move(owner_derivation), std::move(emission),
                       guest_instruction_word_count});
    if (options.max_patches_is_expert_limit && planned.size() == options.max_patches)
      break;
  }

  if (planned.empty())
    return;

  MoiDescriptorVgprRequirements vgprs;
  MoiDescriptorSgprRequirements sgprs;
  MoiDescriptorPrivateRequirements private_bytes;
  MoiDescriptorLdsRequirements lds;
  std::vector<ConSanTextFragment> fragments;
  fragments.reserve(planned.size());
  for (const PlannedInlineShadowFragment &entry : planned) {
    const ConSanMoiCandidate &candidate = *entry.candidate;
    note_descriptor_requirements(vgprs, entry.resources);
    note_moi_sgpr_requirements(sgprs, entry.resources, options, options, entry.point, semantics,
                               arch);
    note_moi_probe_private_requirements(private_bytes, entry);
    if (entry.emission.workgroup_shadow)
      note_moi_lds_requirements(lds, entry.resources, *entry.emission.workgroup_shadow);

    uint32_t guest_words = 0u;
    auto probe_words = build_inline_shadow_words(bytes, candidate, entry.emission, arch, layout,
                                                 entry.spill ? &*entry.spill : nullptr,
                                                 result.errors, &guest_words);
    if (!probe_words)
      return;
    if (guest_words == 0u || guest_words > probe_words->size()) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow probe lost its relocated guest instruction");
      return;
    }

    const bool defer_guest = candidate.site().kind == ConSanLdsAccessKind::Read ||
                             candidate.site().kind == ConSanLdsAccessKind::Write;
    MoiAppendedBodyOptions body_options(arch);
    body_options.trailing_guest_word_count = defer_guest ? guest_words : 0u;
    body_options.scalar_save_words = entry.scalar_spill
                                         ? std::span<const uint32_t>(entry.scalar_spill->save_words)
                                         : std::span<const uint32_t>{};
    body_options.scalar_restore_words =
        entry.scalar_spill ? std::span<const uint32_t>(entry.scalar_spill->restore_words)
                           : std::span<const uint32_t>{};
    body_options.incoming_vgpr_bank_mode = candidate.incoming_vgpr_bank_mode;
    body_options.probe_guest_instruction_offset =
        static_cast<uint32_t>((probe_words->size() - guest_words) * sizeof(uint32_t));
    body_options.guest_instruction_word_count = guest_words;
    uint32_t body_guest_offset = 0u;
    body_options.body_guest_instruction_offset = &body_guest_offset;
    body_options.restore_vgpr_spill_before_trailing_guest = defer_guest && entry.spill.has_value();
    body_options.wrap_embedded_guest_vgpr_bank =
        !defer_guest && candidate.incoming_vgpr_bank_mode.value_or(0u) != 0u;
    if (moi_access_requires_high_bank_address_capture(candidate, arch)) {
      body_options.high_bank_address_source = candidate.site().lowering.form->address_vgpr;
      body_options.high_bank_address_destination = static_cast<uint16_t>(
          consan_detail::inline_shadow_loop_counter_vgpr(
              entry.resources.base, entry.emission.scalar_state.exec_save_sgpr.has_value(),
              entry.emission.track_atomics) +
          inline_shadow_loop_scratch_count(candidate));
    }
    std::optional<uint32_t> empty_exec_guard;
    if (entry.scalar_spill) {
      empty_exec_guard = instrumentation::build_s_cbranch_execz(0, arch);
      if (!empty_exec_guard) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe could not encode its empty-wave guard");
        return;
      }
      body_options.entry_prefix_words = std::span<const uint32_t>(&*empty_exec_guard, 1u);
    }

    auto body = assemble_moi_appended_body({.spill = entry.spill ? &*entry.spill : nullptr,
                                            .displaced_tail_words = {},
                                            .body_size = 0u,
                                            .guest_instruction_size = candidate.size()},
                                           *probe_words, "inline-shadow probe", result.errors,
                                           body_options);
    if (!body)
      return;

    ConSanPatchInfo patch;
    patch.kind = ConSanPatchKind::TrampolineMoiExactShadowStore;
    patch.anchor_offset = candidate.anchor();
    patch.original_size = candidate.size();
    note_moi_probe_patch_info(patch, entry);
    patch.moi_dispatch_identity_source = entry.emission.dispatch_id;
    patch.workgroup_shadow = entry.emission.workgroup_shadow;
    auto fragment = make_consan_around_text_fragment(
        std::move(*body), body_guest_offset, candidate.size(), candidate.intent_ids, {},
        std::move(patch), result.errors, "MOI inline-shadow probe",
        guest_words * static_cast<uint32_t>(sizeof(uint32_t)));
    if (!fragment)
      return;
    if (entry.scalar_spill) {
      fragment->branch_fixups.push_back(
          {.word_index = 0u, .target = ConSanTextFragmentBranchTarget::Guest});
    }
    fragments.push_back(std::move(*fragment));
  }

  (void)stage_consan_text_rewrites(code_object, arch, {vgprs, sgprs, private_bytes, lds},
                                   moi_descriptor_mutation_policy(&options, arch),
                                   "MOI inline-shadow probe", std::move(fragments), result);
}
