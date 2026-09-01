void try_apply_inline_shadow_patch(std::span<const uint8_t> bytes, const ConSanOptions &options,
                                   const ConSanMoiOperatingPoint &operating_point,
                                   rj_code_arch_t arch, MoiResourcePlanningState &resource_state,
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
  std::unique_ptr<Decoder> relocation_decoder = Decoder::create(arch);
  if (!relocation_decoder) {
    result.warnings.emplace_back("ConSan MOI inline-shadow could not create a relocation decoder");
    return;
  }

  const ConSanMoiReportBufferLayout &layout = semantics.report_layout;
  if (options.max_patches == 0)
    return;
  if (!validate_inline_shadow_exec_save_sgpr(options, options, operating_point, semantics, arch,
                                             result.warnings))
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

  // A workgroup-local shadow cell is updated with one B64 DS exchange per
  // participating lane. That is exact when a lane owns the whole four-byte
  // cell, but adjacent B8/B16 lanes can serialize through the same cell with
  // different byte masks. The first consumer may then replace the producer
  // and subsequent read/read exchanges erase its evidence. Keep every access
  // in the object in the wave-partitioned external table whenever subword
  // traffic is present, so mixed-width pairs also share one coherent shadow
  // domain. A future local representation may lift this restriction by
  // retaining independent provenance for every byte.
  const bool requires_wave_partitioned_exact_shadow =
      std::ranges::any_of(candidates, [&](const ConSanMoiCandidate *candidate) {
        return std::ranges::any_of(candidate->ranges, [](const ConSanAccessRange &range) {
          return range.byte_width < consan_moi_exact_shadow::granule_bytes;
        });
      });
  if (requires_wave_partitioned_exact_shadow) {
    result.warnings.emplace_back(
        "ConSan MOI inline-shadow subword traffic uses the wave-partitioned external exact-shadow "
        "table");
  }

  struct PlannedInlineShadowPatch : MoiPlannedAccessPatch {
    std::optional<uint64_t> branch_only_relay_bank_offset;
    uint64_t branch_only_relay_bank_word_count = 0;
    ConSanMoiWorkgroupSources workgroup_sources;
    std::optional<uint32_t> private_workgroup_key_offset;
    std::optional<uint32_t> private_dispatch_id_offset;
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow;
    uint16_t workgroup_shadow_compact_token = 0;
  };
  std::vector<PlannedInlineShadowPatch> planned_patches;
  planned_patches.reserve(candidates.size());
  AmdGpuCodeObject code_object(bytes.data(), bytes.size());
  const uint64_t original_text_size =
      code_object.text_sections().size() == 1 ? code_object.text_sections().front()->size() : 0;
  constexpr uint64_t kIndirectIslandWords = kMoiInlineShadowIndirectIslandWords;
  // A gfx1250 dispatcher can contain a stranded singleton. Its worst case is
  // a 12-word relocated-host arm, a 15-word two-word-access arm (return-PC
  // match, guest-return adjustment, conditional branch, and long jump), plus
  // the final terminator. Keep RDNA4's compact spacing: its explicit-key
  // groups do not carry the gfx1250 call-return and adjustment sequences.
  const uint64_t access_relay_words =
      target->direct_call_form == ConSanDirectCallForm::SCallI64 ? 28u : 20u;
  const bool has_function_candidate =
      std::ranges::any_of(candidates, [](const ConSanMoiCandidate *candidate) {
        return !candidate->container.is_kernel();
      });
  std::unordered_set<uint64_t> dynamic_lds_descriptors;
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    if (kernel.has_dynamic_lds)
      dynamic_lds_descriptors.insert(kernel.descriptor_file_offset);
  }
  std::unordered_map<uint64_t, std::optional<ConSanMoiOperatingPoint>> direct_points_by_descriptor;
  std::unordered_map<uint64_t, std::optional<ConSanMoiWorkgroupShadowLayout>>
      direct_workgroup_shadow_by_descriptor;
  MoiPrivateEpochLayoutCache direct_private_layout_cache;
  std::unordered_map<uint64_t, std::optional<uint16_t>> direct_owner_shift_by_descriptor;
  std::unordered_map<uint64_t, std::optional<ConSanMoiWorkgroupSources>>
      direct_workgroup_sources_by_descriptor;
  bool warned_dynamic_lds_external_shadow = false;
  const bool has_branch_only_scalar_assignment =
      std::ranges::any_of(result.moi_operating_point.owner_transient_sgprs,
                          [](const ConSanMoiTransientSgprAssignment &assignment) {
                            return assignment.branch_only_spill.has_value();
                          });
  const bool use_indirect_appended =
      operating_point.moi_exec_save_sgpr.has_value() &&
      (candidates.size() >= 8u || has_function_candidate || has_branch_only_scalar_assignment);
  const uint64_t reserved_barrier_island_count =
      use_indirect_appended ? moi_reserved_access_sync_island_count(result.program_inventory,
                                                                    result.observation_plan(),
                                                                    options, options, semantics)
                            : 0u;
  const uint64_t indirect_body_begin =
      original_text_size + reserved_barrier_island_count * kIndirectIslandWords * sizeof(uint32_t) +
      candidates.size() * access_relay_words * sizeof(uint32_t);
  const uint64_t access_island_begin =
      original_text_size + reserved_barrier_island_count * kIndirectIslandWords * sizeof(uint32_t);
  DbiPatchPlacementPlanner placement_planner(arch, use_indirect_appended ? indirect_body_begin
                                                                         : original_text_size);

  std::span<const uint8_t> original_text;
  if (code_object.text_sections().size() == 1) {
    const Section *text = code_object.text_sections().front();
    original_text =
        std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(text->data()), text->size());
  }
  const std::vector<std::pair<uint64_t, uint64_t>> synchronization_ranges =
      collect_moi_synchronization_ranges(result.program_inventory);
  const auto relocation_overlaps_synchronization = [&](uint64_t begin, uint64_t end) {
    const auto after = std::ranges::lower_bound(synchronization_ranges, end, {},
                                                [](const auto &range) { return range.first; });
    return after != synchronization_ranges.begin() && std::prev(after)->second > begin;
  };
  std::optional<MoiLocalNopIslandAllocator> local_entry_islands;
  if (use_indirect_appended && !original_text.empty())
    local_entry_islands.emplace(original_text, result.program_inventory, result.patches, arch,
                                kIndirectIslandWords, moi_resource_reserved_ranges(resource_state));
  BranchOnlyRelayRouter branch_router;
  std::optional<uint64_t> latest_generated_frontier_word;
  if (local_entry_islands) {
    for (uint64_t offset : local_entry_islands->available_word_offsets())
      (void)branch_router.offer(offset, BranchOnlyRelayProvenance::PristineNop);
  }
  const auto retire_branch_relay_range = [&](uint64_t offset, uint32_t words) {
    branch_router.retire_range(offset, static_cast<uint64_t>(words) * sizeof(uint32_t));
  };

  MoiDenseAccessRoutePlan dense_routes = plan_moi_dense_access_routes(
      result.program_inventory, result.resource_plans, options, operating_point, arch,
      resource_state, original_text, candidates, target->moi_dense_route_group_capacity,
      access_island_begin, access_relay_words, use_indirect_appended,
      MoiDenseAccessRouteAbi::InlineShadow, local_entry_islands ? &*local_entry_islands : nullptr,
      &branch_router);
  const auto &dense_candidates = dense_routes.partition;
  const auto &candidate_indices = dense_candidates.index_by_candidate;
  auto &dense_entry_islands = dense_routes.entry_islands;
  auto &dense_dispatchers = dense_routes.dispatchers;
  auto &dense_entry_hosts = dense_routes.entry_hosts;
  if (use_indirect_appended) {
    // Every ordinary access owns an eight-word entry-island prefix. The
    // remaining per-site bank is explicit branch-relay capacity unless a
    // dense dispatcher owns that complete slice.
    for (const ConSanMoiCandidate *candidate : candidates) {
      if (dense_entry_islands.contains(candidate))
        continue;
      const uint64_t slice_begin = access_island_begin + candidate_indices.at(candidate) *
                                                             access_relay_words * sizeof(uint32_t);
      for (uint64_t word = kIndirectIslandWords; word < access_relay_words; ++word) {
        const uint64_t offset = slice_begin + word * sizeof(uint32_t);
        if (branch_router.offer(offset, BranchOnlyRelayProvenance::GeneratedBank)) {
          latest_generated_frontier_word =
              std::max(latest_generated_frontier_word.value_or(0u), offset);
        }
      }
    }
  }
  const auto uses_branch_only_scalar_spill = [&](const ConSanMoiCandidate *candidate) {
    return moi_candidate_uses_branch_only_scalar_spill(result.resource_plans,
                                                       result.moi_operating_point, *candidate);
  };
  BranchOnlyDirectRelayReservoirSet direct_branch_reservoirs;
  if (has_branch_only_scalar_assignment && use_indirect_appended && !original_text.empty()) {
    std::vector<std::pair<uint64_t, uint64_t>> candidate_ranges;
    candidate_ranges.reserve(candidates.size());
    for (const ConSanMoiCandidate *candidate : candidates)
      candidate_ranges.emplace_back(candidate->anchor(), candidate->anchor() + candidate->size());
    uint64_t route_frontier_source = std::numeric_limits<uint64_t>::max();
    for (const ConSanMoiCandidate *candidate : candidates) {
      if (uses_branch_only_scalar_spill(candidate))
        route_frontier_source = std::min(route_frontier_source, candidate->anchor());
    }
    if (route_frontier_source == std::numeric_limits<uint64_t>::max()) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow probe lost its branch-only routing frontier");
      return;
    }
    std::string relay_error;
    if (!plan_moi_access_direct_reservoirs(result.program_inventory, options, arch, resource_state,
                                           original_text, synchronization_ranges, candidate_ranges,
                                           dense_entry_hosts, route_frontier_source,
                                           2u * candidates.size(), placement_planner, branch_router,
                                           direct_branch_reservoirs, &relay_error)) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe " + relay_error);
      return;
    }
  }

  MoiSpillManagers spill_managers;
  std::vector<std::pair<uint64_t, uint64_t>> planned_anchor_ranges;
  bool warned_access_relay_sync_overlap = false;
  std::unordered_map<uint64_t, uint16_t> next_compact_token_by_descriptor;
  std::unordered_set<uint64_t> selected_inline_anchor_relay_offsets;
  const auto publish_inline_anchor_relays = [&](const ConSanMoiCandidate &candidate,
                                                uint32_t original_size) {
    if (original_size <= sizeof(uint32_t))
      return;
    branch_router.retire_range(candidate.anchor(), original_size);
    for (uint64_t offset = candidate.anchor() + sizeof(uint32_t);
         offset < candidate.anchor() + original_size; offset += sizeof(uint32_t)) {
      if (selected_inline_anchor_relay_offsets.insert(offset).second)
        (void)branch_router.offer(offset, BranchOnlyRelayProvenance::OwnedAnchor);
    }
  };
  std::vector<const ConSanMoiCandidate *> planning_candidates = candidates;
  if (has_branch_only_scalar_assignment && options.max_patches >= candidates.size()) {
    // Plan branch-only sites nearest appended text first. Their unconditional
    // source branch makes every remaining original-instruction word a proven
    // capacity-one relay host. Publishing those hosts before planning earlier
    // sites avoids depending on padding or workload-specific NOP caves while
    // preserving ordinary candidate order when a caller requests a subset.
    std::ranges::stable_sort(planning_candidates,
                             [&](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
                               const bool lhs_branch_only = uses_branch_only_scalar_spill(lhs);
                               const bool rhs_branch_only = uses_branch_only_scalar_spill(rhs);
                               if (lhs_branch_only != rhs_branch_only)
                                 return lhs_branch_only;
                               if (lhs_branch_only)
                                 return lhs->anchor() > rhs->anchor();
                               return lhs->anchor() < rhs->anchor();
                             });
  }
  for (size_t planning_index = 0; planning_index < planning_candidates.size(); ++planning_index) {
    const ConSanMoiCandidate *candidate_ptr = planning_candidates[planning_index];
    const ConSanMoiCandidate &candidate = *candidate_ptr;
    const bool use_dense_call = dense_entry_islands.contains(candidate_ptr);
    const uint64_t candidate_end = candidate.anchor() + candidate.size();
    if (use_indirect_appended && !use_dense_call &&
        std::ranges::any_of(planned_anchor_ranges, [&](const auto &range) {
          return candidate.anchor() < range.second && range.first < candidate_end;
        })) {
      result.warnings.emplace_back(
          "ConSan MOI inline-shadow probe skipped access site inside a relocated prefix");
      continue;
    }
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
    // A normal full-width plan may itself require spilling. The selected
    // operand-recovery alternative proves that the smaller window can either
    // reload an overlapping address or leave a disjoint address intact until
    // the deferred guest load.
    const uint16_t expected_scratch_count =
        selected_reduced_spill_recovery
            ? inline_shadow_spill_backed_scratch_count(options, operating_point, candidate, arch)
            : inline_shadow_scratch_count(options, operating_point, candidate, arch);
    const auto resources = resolve_moi_scratch_plan(
        *resource_plan, operating_point, result.moi_operating_point, expected_scratch_count);
    if (!resources)
      continue;
    const std::optional<uint64_t> direct_descriptor =
        resources->owner_descriptor_file_offsets.size() == 1u
            ? std::optional<uint64_t>(resources->owner_descriptor_file_offsets.front())
            : std::nullopt;
    ConSanMoiOperatingPoint candidate_point;
    if (direct_descriptor) {
      auto [cached, inserted] =
          direct_points_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
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
    uint16_t workgroup_shadow_compact_token = 0u;
    const bool has_unspilled_component_override = std::ranges::any_of(
        resources->owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
          const auto *assignment =
              moi_transient_sgpr_assignment(result.moi_operating_point, descriptor_offset);
          return assignment != nullptr && !assignment->spill_backed;
        });
    // The spill-backed scalar window is preserved only after the access body
    // establishes its private frame. Kernel-entry local-shadow initialization
    // cannot borrow that guest window. Keep the semantically equivalent
    // external exact-shadow table for every spill-backed component, whether
    // or not it also found a fresh visible-evidence latch.
    const bool spill_backed_scalar_state = candidate_point.has_inline_moi_scalar_spill();
    if (!requires_wave_partitioned_exact_shadow && !has_unspilled_component_override &&
        !spill_backed_scalar_state) {
      uint16_t next_compact_token = 0u;
      if (candidate.container.is_kernel() && candidate.kernel_descriptor_file_offset) {
        uint16_t &next = next_compact_token_by_descriptor[*candidate.kernel_descriptor_file_offset];
        if (next == 0u)
          next = 1u;
        next_compact_token = next;
      }
      const bool has_dynamic_lds_owner = std::ranges::any_of(
          resources->owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
            return dynamic_lds_descriptors.contains(descriptor_offset);
          });
      if (!has_dynamic_lds_owner) {
        if (direct_descriptor) {
          auto [cached, inserted] =
              direct_workgroup_shadow_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
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
      } else if (workgroup_shadow->compact) {
        workgroup_shadow_compact_token = next_compact_token;
      }
    }

    std::optional<MoiPrivateEpochLayout> private_layout;
    std::optional<MoiWorkitemOwnerDerivationPlan> owner_derivation;
    if (candidate_point.automatic_moi_private_epoch) {
      // Every supported target exposes workgroup identity as an entry ABI
      // value. Preserve the complete key before arbitrary guest code can reuse
      // the ordinary SGPR or TTMP inputs.
      const bool persist_workgroup_key = true;
      // The object-wide hardware pair may overlap one full-pressure owner's
      // guest state. Capture that owner's identity in private storage and
      // reload it into the access router's dead PC pair after the router has
      // entered its body.
      const bool persist_dispatch_id = !moi_has_runtime_hardware_dispatch_id(candidate_point) &&
                                       (candidate_point.moi_dispatch_identity.sgpr() ||
                                        candidate_point.moi_dispatch_identity.private_fallback());
      private_layout = direct_private_layout_cache.resolve(
          direct_descriptor, result.program_inventory, *resources, arch, result.warnings,
          {.owner = options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId,
           .workgroup_key = persist_workgroup_key,
           .dispatch_id = persist_dispatch_id});
      if (!private_layout)
        continue;
      if (options.moi_owner_source == ConSanMoiOwnerSource::WorkitemId) {
        std::optional<uint16_t> owner_shift;
        if (direct_descriptor) {
          auto [cached, inserted] =
              direct_owner_shift_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
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
      const auto reload = moi_indirect_jump_sgprs(options, candidate_point);
      if (!reload) {
        result.warnings.emplace_back(
            "ConSan MOI private dispatch identity has no owner-local scalar reload pair");
        continue;
      }
      candidate_point.moi_dispatch_identity.set_sgpr(reload->pc_sgpr);
    }

    std::vector<std::string> workgroup_errors;
    std::optional<ConSanMoiWorkgroupSources> workgroup_sources;
    const bool has_persistent_workgroup_key =
        (private_layout && private_layout->workgroup_key_offset) ||
        candidate_point.moi_workgroup_key_vgpr ||
        candidate_point.moi_persistent_sgprs.workgroup_key;
    if (has_persistent_workgroup_key) {
      // The owner-specific entry prologue captured the descriptor-valid
      // coordinates. Shared physical text can therefore consume one stable
      // key even when its owning descriptors request different coordinate
      // subsets.
      workgroup_sources = ConSanMoiWorkgroupSources{};
    } else {
      std::optional<uint64_t> workgroup_descriptor = direct_descriptor;
      if (!workgroup_descriptor) {
        workgroup_descriptor =
            common_moi_record_owner_descriptor(bytes, *resources, arch, result.warnings);
      }
      if (!workgroup_descriptor)
        continue;
      if (direct_descriptor) {
        auto [cached, inserted] =
            direct_workgroup_sources_by_descriptor.try_emplace(*direct_descriptor, std::nullopt);
        if (inserted) {
          cached->second =
              moi_descriptor_workgroup_sources(bytes, *direct_descriptor, arch, workgroup_errors);
        }
        workgroup_sources = cached->second;
      } else {
        workgroup_sources =
            moi_descriptor_workgroup_sources(bytes, *workgroup_descriptor, arch, workgroup_errors);
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

    std::vector<std::string> candidate_errors;
    uint32_t guest_instruction_word_count = 0;
    auto words = build_inline_shadow_words(
        bytes, candidate, options, options, candidate_point, resources->base, semantics,
        options.moi_partition_mask_debug, arch, layout, *workgroup_sources, workgroup_shadow,
        requires_wave_partitioned_exact_shadow,
        private_layout ? std::optional<uint32_t>(private_layout->epoch_offset) : std::nullopt,
        owner_derivation, private_layout ? private_layout->workgroup_key_offset : std::nullopt,
        private_layout ? private_layout->dispatch_id_offset : std::nullopt,
        workgroup_shadow_compact_token, resources->count, spill.has_value(),
        spill ? &*spill : nullptr, candidate_errors, &guest_instruction_word_count);
    if (!words) {
      for (std::string &error : candidate_errors) {
        error += " for " + candidate.mnemonic + " at text offset " +
                 std::to_string(candidate.anchor()) + " with scratch v" +
                 std::to_string(resources->base) + ":v" +
                 std::to_string(resources->base + resources->count - 1u);
      }
      result.warnings.insert(result.warnings.end(), candidate_errors.begin(),
                             candidate_errors.end());
      continue;
    }

    const uint64_t spill_bytes =
        spill ? static_cast<uint64_t>((spill->save_words.size() + spill->restore_words.size()) *
                                      sizeof(uint32_t))
              : 0;
    const uint64_t scalar_spill_bytes =
        scalar_spill ? static_cast<uint64_t>(scalar_spill->save_words.size() +
                                             scalar_spill->restore_words.size()) *
                           sizeof(uint32_t)
                     : 0;
    // Per-lane private stores cannot preserve a borrowed scalar window when
    // EXEC is empty. Reserve a leading guard for every scalar-spill body so
    // emission can bypass the save/probe/restore sequence and continue with
    // the displaced guest instruction (which is itself inert under EXEC=0).
    const uint64_t scalar_spill_guard_bytes = scalar_spill ? sizeof(uint32_t) : 0u;
    const bool capture_high_bank_address =
        moi_access_requires_high_bank_address_capture(candidate, arch);
    const uint64_t high_bank_capture_bytes =
        capture_high_bank_address
            ? (1u + (spill                                                            ? 2u
                     : (candidate.incoming_vgpr_bank_mode.value_or(0u) & 0xfcu) != 0u ? 1u
                                                                                      : 0u)) *
                  sizeof(uint32_t)
            : 0u;
    const uint64_t vgpr_msb_bytes =
        (target->has_selectable_vgpr_bank && candidate.incoming_vgpr_bank_mode.value_or(0u) != 0u
             ? (spill ? 4u : 2u) * sizeof(uint32_t)
             : 0u) +
        high_bank_capture_bytes;
    const uint32_t required_private_bytes =
        std::max({private_layout ? private_layout->ephemeral_base : 0u,
                  spill ? spill->total_private_bytes : 0u,
                  scalar_spill ? scalar_spill->total_private_bytes : 0u});
    const uint32_t available_padding =
        count_nop_padding(bytes, candidate.file_offset + candidate.size(), arch);
    const uint64_t available_bytes =
        candidate.size() + static_cast<uint64_t>(available_padding) * sizeof(uint32_t);
    std::optional<DbiPatchPlacement> placement;
    std::optional<uint64_t> entry_island_offset;
    bool entry_island_at_anchor = false;
    std::vector<uint32_t> displaced_tail_words;
    std::optional<BranchOnlyRelayRoute> branch_only_route;
    std::optional<uint64_t> branch_only_relay_bank_offset;
    uint64_t branch_only_relay_bank_word_count = 0u;
    uint32_t displaced_original_size = candidate.size();
    bool placed_through_indirect_island = false;
    if (use_indirect_appended && candidate_point.moi_branch_only_spill) {
      const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t)) +
                                   spill_bytes + scalar_spill_bytes + vgpr_msb_bytes +
                                   scalar_spill_guard_bytes;
      const uint64_t remaining_candidate_count = planning_candidates.size() - planning_index;
      const uint64_t remaining_patch_budget =
          options.max_patches > planned_patches.size()
              ? static_cast<uint64_t>(options.max_patches - planned_patches.size())
              : 0u;
      const uint64_t remaining_pair_capacity =
          std::min(remaining_candidate_count, remaining_patch_budget);
      if (remaining_pair_capacity == 0u ||
          remaining_pair_capacity >
              std::numeric_limits<uint64_t>::max() / (2u * sizeof(uint32_t)) ||
          patch_bytes > std::numeric_limits<uint64_t>::max() - sizeof(uint32_t)) {
      } else {
        DbiPatchPlacementPlanner tentative_planner = placement_planner;
        BranchOnlyRelayRouter tentative_router = branch_router;
        std::string relay_error;
        std::optional<uint64_t> tentative_latest_frontier_word = latest_generated_frontier_word;
        const uint64_t body_reservation_bytes = patch_bytes + sizeof(uint32_t);
        const uint64_t appended_begin = tentative_planner.appended_end();
        const uint64_t future_pair_capacity =
            std::min(remaining_candidate_count - 1u, remaining_patch_budget - 1u);
        const uint64_t future_bank_words = future_pair_capacity * 2u;
        const bool lookahead_overflows =
            future_bank_words > std::numeric_limits<uint64_t>::max() / sizeof(uint32_t) ||
            body_reservation_bytes >
                std::numeric_limits<uint64_t>::max() - future_bank_words * sizeof(uint32_t) ||
            appended_begin > std::numeric_limits<uint64_t>::max() - body_reservation_bytes -
                                 future_bank_words * sizeof(uint32_t);
        const uint64_t lookahead_end =
            lookahead_overflows
                ? 0u
                : appended_begin + body_reservation_bytes + future_bank_words * sizeof(uint32_t);
        const bool needs_spatial_frontier =
            !tentative_latest_frontier_word || lookahead_overflows ||
            !compute_sopp_branch_simm16(*tentative_latest_frontier_word, lookahead_end) ||
            !compute_sopp_branch_simm16(lookahead_end, *tentative_latest_frontier_word);
        bool frontier_ready = !lookahead_overflows;
        if (needs_spatial_frontier && frontier_ready) {
          const uint64_t relay_bank_words = remaining_pair_capacity * 2u;
          const uint64_t relay_bank_bytes = relay_bank_words * sizeof(uint32_t);
          const uint64_t relay_bank_offset = tentative_planner.appended_end();
          const bool bank_overflows =
              relay_bank_bytes > std::numeric_limits<uint64_t>::max() - relay_bank_offset ||
              body_reservation_bytes >
                  std::numeric_limits<uint64_t>::max() - relay_bank_offset - relay_bank_bytes;
          const uint64_t relay_bank_last_word =
              bank_overflows ? 0u : relay_bank_offset + relay_bank_bytes - sizeof(uint32_t);
          const uint64_t body_end =
              bank_overflows ? 0u : relay_bank_offset + relay_bank_bytes + body_reservation_bytes;
          const bool bank_reaches_prior_frontier =
              !tentative_latest_frontier_word ||
              (compute_sopp_branch_simm16(*tentative_latest_frontier_word, relay_bank_last_word) &&
               compute_sopp_branch_simm16(relay_bank_last_word, *tentative_latest_frontier_word));
          const bool bank_spans_body = !bank_overflows &&
                                       compute_sopp_branch_simm16(relay_bank_offset, body_end) &&
                                       compute_sopp_branch_simm16(body_end, relay_bank_offset);
          frontier_ready =
              bank_reaches_prior_frontier && bank_spans_body &&
              tentative_planner.reserve_appended_prefix(relay_bank_bytes, &relay_error);
          if (frontier_ready) {
            for (uint64_t word = 0; word < relay_bank_words; ++word) {
              (void)tentative_router.offer(relay_bank_offset + word * sizeof(uint32_t),
                                           BranchOnlyRelayProvenance::GeneratedBank);
            }
            branch_only_relay_bank_offset = relay_bank_offset;
            branch_only_relay_bank_word_count = relay_bank_words;
            tentative_latest_frontier_word = relay_bank_last_word;
          }
        }
        const auto reservation =
            frontier_ready ? tentative_planner.plan_indirect_appended(
                                 candidate.anchor(), candidate.size(), body_reservation_bytes)
                           : std::nullopt;
        if (!reservation) {
        } else {
          BranchOnlyRelaySearchLimits relay_limits;
          // Inline routing exposes banks shared by the complete object, so its
          // useful fallback window is larger than the generic router default.
          // Fixed ceilings keep every pair bounded independently of inventory.
          relay_limits.batch.fallback_setup = 1'000'000u;
          relay_limits.pair.greedy = 1'000'000u;
          branch_only_route = tentative_router.plan_pair(
              tentative_planner, candidate.anchor(), reservation->body_offset,
              reservation->body_offset + patch_bytes, candidate.anchor() + candidate.size(),
              &relay_error, nullptr, relay_limits);
          if (branch_only_route && !tentative_router.commit(*branch_only_route, &relay_error)) {
            branch_only_route.reset();
          } else if (branch_only_route) {
            for (const BranchOnlyRelayClaim &claim : branch_only_route->claims) {
              if (local_entry_islands &&
                  claim.provenance == BranchOnlyRelayProvenance::PristineNop) {
                local_entry_islands->reserve(claim.offset, 1u);
              }
            }
            if (!direct_branch_reservoirs.mark_claims_used(branch_only_route->claims,
                                                           &relay_error)) {
              result.errors.emplace_back("ConSan MOI inline-shadow probe " + relay_error);
              return;
            }
            branch_router = std::move(tentative_router);
            placement_planner = std::move(tentative_planner);
            latest_generated_frontier_word = tentative_latest_frontier_word;
            placement = *reservation;
            placement->body_size = patch_bytes;
            placement->return_target = candidate.anchor() + candidate.size();
            placed_through_indirect_island = true;
          }
        }
      }
    }
    if (use_indirect_appended && !placed_through_indirect_island) {
      if (candidate_point.moi_branch_only_spill) {
        result.warnings.emplace_back(
            "ConSan MOI inline-shadow probe could not route its branch-only scalar-spill body");
        continue;
      }
      const uint64_t preferred_entry_island =
          use_dense_call ? dense_entry_islands.at(candidate_ptr)
                         : access_island_begin + candidate_indices.at(candidate_ptr) *
                                                     access_relay_words * sizeof(uint32_t);
      auto entry_plan = plan_moi_access_entry_island_with_policy(
          original_text, candidate, kIndirectIslandWords, preferred_entry_island,
          kIndirectIslandWords,
          /*use_preferred_without_direct_branch=*/false,
          !use_dense_call && local_entry_islands ? &*local_entry_islands : nullptr, resource_state,
          relocation_decoder.get(), arch, candidate_errors, [&](uint64_t begin, uint64_t end) {
            if (use_dense_call)
              return false;
            const bool overlaps = relocation_overlaps_synchronization(begin, end);
            if (overlaps && !warned_access_relay_sync_overlap) {
              result.warnings.emplace_back(
                  "ConSan MOI inline-shadow probe did not relocate a synchronization site through "
                  "an uninstrumented access entry relay");
              warned_access_relay_sync_overlap = true;
            }
            return !overlaps;
          });
      if (entry_plan) {
        entry_island_offset = entry_plan->offset;
        entry_island_at_anchor = entry_plan->at_anchor;
        displaced_original_size = entry_plan->original_size;
        displaced_tail_words = std::move(entry_plan->displaced_tail_words);
        if (!entry_plan->uses_preferred_island && !entry_plan->at_anchor)
          retire_branch_relay_range(*entry_plan->offset, kIndirectIslandWords);
        const uint64_t patch_bytes =
            static_cast<uint64_t>((words->size() + displaced_tail_words.size()) *
                                  sizeof(uint32_t)) +
            spill_bytes + scalar_spill_bytes + vgpr_msb_bytes + scalar_spill_guard_bytes;
        // The long return is getpc, a three-word signed delta builder, SCC
        // restore, a scalar-destination dependency wait, and setpc. The fixed
        // reservation keeps later body addresses stable while their
        // instruction words are regenerated during emission.
        const uint64_t return_bytes =
            (use_dense_call && target->direct_call_form == ConSanDirectCallForm::SCallI64 ? 1u
                                                                                          : 7u) *
            sizeof(uint32_t);
        std::string placement_error;
        const auto reservation =
            placement_planner.plan_indirect_appended(candidate.anchor(), displaced_original_size,
                                                     patch_bytes + return_bytes, &placement_error);
        if (reservation) {
          placement = *reservation;
          placement->body_size = patch_bytes;
          placement->return_target = candidate.anchor() + displaced_original_size;
          placed_through_indirect_island = true;
        }
      }
    }
    if (!placed_through_indirect_island) {
      const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t)) +
                                   spill_bytes + scalar_spill_bytes + vgpr_msb_bytes +
                                   scalar_spill_guard_bytes;
      const bool needs_chainable_entry_patch =
          moi_initializes_owner_epoch(options, operating_point) &&
          std::ranges::any_of(
              resources->owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
                const ConSanKernelInfo *kernel =
                    result.program_inventory.find_kernel_by_descriptor(descriptor_offset);
                if (kernel == nullptr || kernel->entry_text_offset != candidate.anchor()) {
                  return false;
                }
                return kernel->uses_dynamic_stack.value_or(false) ||
                       operating_point.automatic_moi_persistent_vgprs ||
                       operating_point.moi_persistent_sgprs.complete();
              });
      DbiPatchPlacementRequest placement_request;
      placement_request.anchor_offset = candidate.anchor();
      placement_request.original_size = candidate.size();
      placement_request.body_size = patch_bytes;
      placement_request.inline_capacity = spill || scalar_spill || private_layout ||
                                                  workgroup_shadow || needs_chainable_entry_patch ||
                                                  capture_high_bank_address
                                              ? 0u
                                              : available_bytes;
      std::string placement_error;
      placement = placement_planner.plan(placement_request, &placement_error);
      if (!placement) {
        result.warnings.emplace_back("ConSan MOI inline-shadow probe skipped access site: " +
                                     placement_error);
        continue;
      }
    }

    PlannedInlineShadowPatch planned;
    planned.candidate = candidate_ptr;
    planned.placement = *placement;
    planned.entry_island_offset = entry_island_offset;
    planned.entry_island_at_anchor = entry_island_at_anchor;
    planned.dense_call_anchor = use_dense_call;
    planned.branch_only_route = std::move(branch_only_route);
    planned.branch_only_relay_bank_offset = branch_only_relay_bank_offset;
    planned.branch_only_relay_bank_word_count = branch_only_relay_bank_word_count;
    if (use_dense_call)
      planned.dense_dispatcher_offset = dense_dispatchers.at(candidate_ptr);
    planned.displaced_tail_words = std::move(displaced_tail_words);
    planned.resources = *resources;
    planned.workgroup_sources = *workgroup_sources;
    planned.spill = spill;
    planned.scalar_spill = scalar_spill;
    if (private_layout) {
      planned.private_epoch_offset = private_layout->epoch_offset;
      planned.private_workgroup_key_offset = private_layout->workgroup_key_offset;
      planned.private_dispatch_id_offset = private_layout->dispatch_id_offset;
      planned.persistent_private_state_end = private_layout->persistent_state_end;
    }
    planned.owner_derivation = owner_derivation;
    planned.workgroup_shadow = workgroup_shadow;
    planned.workgroup_shadow_compact_token = workgroup_shadow_compact_token;
    planned.required_private_bytes = required_private_bytes;
    planned.guest_instruction_word_count = guest_instruction_word_count;
    const bool routed_through_branch_only = planned.branch_only_route.has_value();
    planned_patches.push_back(std::move(planned));
    if (routed_through_branch_only)
      publish_inline_anchor_relays(candidate, placement->original_size);
    if (use_indirect_appended && !use_dense_call)
      planned_anchor_ranges.emplace_back(candidate.anchor(),
                                         candidate.anchor() + placement->original_size);
    if (workgroup_shadow && workgroup_shadow->compact && candidate.kernel_descriptor_file_offset) {
      ++next_compact_token_by_descriptor[*candidate.kernel_descriptor_file_offset];
    }
    if (planned_patches.size() == options.max_patches)
      break;
  }

  for (auto &[dispatcher, host] : dense_entry_hosts) {
    (void)dispatcher;
    host.body_offset = placement_planner.appended_end();
    std::string reservation_error;
    if (!placement_planner.reserve_appended_prefix(
            (host.displaced_words.size() + kIndirectIslandWords) * sizeof(uint32_t),
            &reservation_error)) {
      result.errors.emplace_back("ConSan MOI inline-shadow probe could not reserve a dense "
                                 "entry-host body: " +
                                 reservation_error);
      return;
    }
  }

  if (planned_patches.empty())
    return;

  std::unordered_map<uint64_t, size_t> inline_anchor_relay_donors;
  for (size_t index = 0; index < planned_patches.size(); ++index) {
    const PlannedInlineShadowPatch &planned = planned_patches[index];
    for (uint64_t offset = planned.candidate->anchor() + sizeof(uint32_t);
         offset < planned.candidate->anchor() + planned.placement.original_size;
         offset += sizeof(uint32_t)) {
      if (selected_inline_anchor_relay_offsets.contains(offset))
        inline_anchor_relay_donors.emplace(offset, index);
    }
  }
  for (size_t index = 0; index < planned_patches.size(); ++index) {
    const PlannedInlineShadowPatch &planned = planned_patches[index];
    const auto donor_precedes_borrower = [&](uint64_t relay) {
      if (!selected_inline_anchor_relay_offsets.contains(relay))
        return true;
      const auto donor = inline_anchor_relay_donors.find(relay);
      return donor != inline_anchor_relay_donors.end() && donor->second < index;
    };
    if (planned.branch_only_route &&
        (!std::ranges::all_of(planned.branch_only_route->entry_relay_offsets,
                              donor_precedes_borrower) ||
         !std::ranges::all_of(planned.branch_only_route->return_relay_offsets,
                              donor_precedes_borrower))) {
      result.errors.emplace_back(
          "ConSan MOI inline-shadow probe planned an anchor-tail relay before its donor");
      return;
    }
  }

  MoiDescriptorVgprRequirements descriptor_requirements;
  MoiDescriptorSgprRequirements scalar_requirements;
  MoiDescriptorPrivateRequirements private_requirements;
  MoiDescriptorLdsRequirements lds_requirements;
  for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
    note_descriptor_requirements(descriptor_requirements, planned_patch.resources);
    ConSanMoiOperatingPoint candidate_point = operating_point;
    if (!apply_moi_transient_sgpr_assignment(options, candidate_point, result.moi_operating_point,
                                             planned_patch.resources.owner_descriptor_file_offsets))
      continue;
    note_moi_sgpr_requirements(scalar_requirements, planned_patch.resources, options, options,
                               candidate_point, semantics, arch);
    if (planned_patch.workgroup_shadow) {
      note_moi_lds_requirements(lds_requirements, planned_patch.resources,
                                *planned_patch.workgroup_shadow);
    }
    note_moi_access_private_requirements(private_requirements, planned_patch);
  }

  const bool uses_appended_cave =
      std::ranges::any_of(planned_patches, [](const PlannedInlineShadowPatch &patch) {
        return patch.placement.kind == DbiPatchPlacementKind::AppendedCave;
      });
  if (uses_appended_cave) {
    CodeObjectPatcher patcher(code_object);
    if (!apply_moi_appended_access_descriptor_requirements(
            patcher, code_object, options, descriptor_requirements, scalar_requirements,
            private_requirements, &lds_requirements, arch, result))
      return;
    std::vector<uint8_t> new_text;
    std::vector<ConSanPatchInfo> patches;
    std::vector<ConSanCommittedLowering> lowering_commits;
    patches.reserve(planned_patches.size());
    lowering_commits.reserve(planned_patches.size());
    if (!initialize_moi_appended_access_text(patcher, use_indirect_appended, indirect_body_begin,
                                             direct_branch_reservoirs, "inline-shadow probe", arch,
                                             new_text, patches, result))
      return;
    const auto dense_groups = group_moi_dense_access_patches(planned_patches);
    if (!emit_moi_dense_access_groups(
            dense_groups, dense_entry_hosts, options, result.moi_operating_point,
            access_relay_words,
            /*collapse_spill_router_for_explicit_key=*/false, "inline-shadow probe", arch, new_text,
            patches, result.errors,
            [](const PlannedInlineShadowPatch &, const ConSanMoiOperatingPoint &) { return true; }))
      return;
    for (const PlannedInlineShadowPatch &planned_patch : planned_patches) {
      const ConSanMoiCandidate &candidate = *planned_patch.candidate;
      ConSanMoiOperatingPoint candidate_point = operating_point;
      if (!apply_moi_transient_sgpr_assignment(
              options, candidate_point, result.moi_operating_point,
              planned_patch.resources.owner_descriptor_file_offsets)) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow access lost its transient scalar assignment");
        return;
      }
      if (!apply_moi_persistent_vgpr_assignment(
              candidate_point, result.moi_operating_point,
              planned_patch.resources.owner_descriptor_file_offsets)) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow access lost its persistent VGPR assignment");
        return;
      }
      if (planned_patch.private_dispatch_id_offset) {
        const auto reload = moi_indirect_jump_sgprs(options, candidate_point);
        if (!reload) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow access lost its private dispatch reload pair");
          return;
        }
        candidate_point.moi_dispatch_identity.set_sgpr(reload->pc_sgpr);
      }
      auto words = build_inline_shadow_words(
          bytes, candidate, options, options, candidate_point, planned_patch.resources.base,
          semantics, options.moi_partition_mask_debug, arch, layout,
          planned_patch.workgroup_sources, planned_patch.workgroup_shadow,
          requires_wave_partitioned_exact_shadow, planned_patch.private_epoch_offset,
          planned_patch.owner_derivation, planned_patch.private_workgroup_key_offset,
          planned_patch.private_dispatch_id_offset, planned_patch.workgroup_shadow_compact_token,
          planned_patch.resources.count, planned_patch.spill.has_value(),
          planned_patch.spill ? &*planned_patch.spill : nullptr, result.errors);
      if (!words)
        return;

      // Keep every displaced memory access as close as possible to the
      // continuation. In particular, appended caves must restore spills and
      // prepare a long return before touching application data; otherwise
      // another wave can observe the access while this wave is still in
      // trampoline bookkeeping.
      std::vector<uint32_t> deferred_guest_words;
      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave &&
          (candidate.kind == ConSanLdsAccessKind::Read ||
           candidate.kind == ConSanLdsAccessKind::Write)) {
        const size_t guest_word_count = planned_patch.guest_instruction_word_count;
        if (words->size() < guest_word_count) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe lost its deferred guest write");
          return;
        }
        deferred_guest_words.assign(words->end() - guest_word_count, words->end());
        words->erase(words->end() - guest_word_count, words->end());
      }

      ConSanPatchInfo info;
      info.kind = planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave
                      ? ConSanPatchKind::TrampolineMoiExactShadowStore
                      : ConSanPatchKind::InlineMoiExactShadowStore;
      info.anchor_offset = candidate.anchor();
      note_moi_access_patch_info(info, planned_patch);
      info.persistent_workgroup_key_private_offset = planned_patch.private_workgroup_key_offset;
      info.persistent_dispatch_id_private_offset = planned_patch.private_dispatch_id_offset;
      if (planned_patch.workgroup_shadow) {
        info.workgroup_shadow_base = planned_patch.workgroup_shadow->base;
        info.workgroup_shadow_size = planned_patch.workgroup_shadow->size;
        info.workgroup_shadow_validity_base = planned_patch.workgroup_shadow->validity_base;
        info.workgroup_shadow_validity_size = planned_patch.workgroup_shadow->validity_size;
        info.workgroup_shadow_lazy_initialization =
            planned_patch.workgroup_shadow->lazy_initialization;
        info.workgroup_shadow_compact = planned_patch.workgroup_shadow->compact;
        info.workgroup_shadow_compact_token = planned_patch.workgroup_shadow_compact_token;
        info.required_group_segment_size =
            planned_patch.workgroup_shadow->required_group_segment_size;
      }

      if (planned_patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
        const uint64_t cave_text_offset = planned_patch.placement.body_offset;
        if (planned_patch.branch_only_relay_bank_offset.has_value() !=
            (planned_patch.branch_only_relay_bank_word_count != 0u)) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe has inconsistent branch-only relay-bank metadata");
          return;
        }
        if (planned_patch.branch_only_relay_bank_offset) {
          if (new_text.size() != *planned_patch.branch_only_relay_bank_offset) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe emitted a stale branch-only relay-bank mapping");
            return;
          }
          for (uint64_t word = 0; word < planned_patch.branch_only_relay_bank_word_count; ++word)
            append_word_bytes(new_text, build_s_nop(0, arch));
        }
        if (new_text.size() != cave_text_offset) {
          result.errors.emplace_back(
              "ConSan MOI inline-shadow probe emitted a stale appended-cave mapping: expected " +
              std::to_string(cave_text_offset) + " actual " + std::to_string(new_text.size()));
          return;
        }
        const uint64_t entry_target =
            planned_patch.branch_only_route &&
                    !planned_patch.branch_only_route->entry_relay_offsets.empty()
                ? planned_patch.branch_only_route->entry_relay_offsets.front()
                : planned_patch.entry_island_offset.value_or(planned_patch.placement.body_offset);
        if (!planned_patch.dense_call_anchor &&
            !write_moi_appended_anchor(new_text, planned_patch, entry_target, arch,
                                       "inline-shadow probe", result.errors))
          return;

        if (planned_patch.entry_island_offset && !planned_patch.dense_call_anchor) {
          const auto jump_sgprs = moi_indirect_jump_sgprs(options, candidate_point);
          if (!jump_sgprs) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe has no indirect jump scalar state");
            return;
          }
          std::vector<uint32_t> island_words;
          if (!append_moi_scc_preserving_indirect_jump(
                  island_words, *planned_patch.entry_island_offset, cave_text_offset,
                  jump_sgprs->pc_sgpr, jump_sgprs->scc_save_sgpr,
                  /*capture_scc=*/true, arch) ||
              island_words.empty()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its entry island");
            return;
          }
          const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
          if (!wait_pc) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its entry dependency wait");
            return;
          }
          island_words.insert(island_words.end() - 1, *wait_pc);
          if (island_words.size() != kIndirectIslandWords) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe encoded an invalid entry island size");
            return;
          }
          if (!write_moi_entry_island(new_text, planned_patch, island_words, patches,
                                      "inline-shadow probe", result.errors))
            return;
        }

        MoiAppendedBodyOptions body_options(arch);
        const std::array<uint32_t, 1> empty_exec_guard_placeholder = {
            build_s_nop(0, arch),
        };
        if (planned_patch.scalar_spill)
          body_options.entry_prefix_words = empty_exec_guard_placeholder;
        body_options.deferred_guest_word_count = deferred_guest_words.size();
        body_options.scalar_save_words = planned_patch.scalar_spill
                                             ? planned_patch.scalar_spill->save_words
                                             : std::span<const uint32_t>{};
        body_options.scalar_restore_words = planned_patch.scalar_spill
                                                ? planned_patch.scalar_spill->restore_words
                                                : std::span<const uint32_t>{};
        body_options.incoming_vgpr_bank_mode = candidate.incoming_vgpr_bank_mode;
        if (moi_access_requires_high_bank_address_capture(candidate, arch)) {
          body_options.high_bank_address_source = candidate.lowering.form->address_vgpr;
          body_options.high_bank_address_destination = static_cast<uint16_t>(
              consan_detail::inline_shadow_loop_counter_vgpr(
                  planned_patch.resources.base, candidate_point.moi_exec_save_sgpr.has_value(),
                  options.moi_track_atomics) +
              inline_shadow_loop_scratch_count(candidate));
        }
        auto cave_words = assemble_moi_appended_body(planned_patch, *words, "inline-shadow probe",
                                                     result.errors, body_options);
        if (!cave_words)
          return;
        std::optional<size_t> empty_exec_continuation_word_offset;
        const auto insert_deferred_guest = [&](size_t word_offset) {
          if (deferred_guest_words.empty())
            return;
          info.relocated_guest_instruction_offset =
              cave_text_offset + word_offset * sizeof(uint32_t);
          cave_words->insert(cave_words->begin() + static_cast<ptrdiff_t>(word_offset),
                             deferred_guest_words.begin(), deferred_guest_words.end());
        };
        // An anchor-resident indirect island may displace instructions after
        // the memory access. Those instructions are the access's continuation
        // and must remain after the deferred guest operation. In particular,
        // a load's relocated consumers may overwrite its address VGPRs.
        std::vector<uint32_t> displaced_tail_words;
        if (!deferred_guest_words.empty() && !planned_patch.displaced_tail_words.empty()) {
          if (cave_words->size() < planned_patch.displaced_tail_words.size()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe lost its displaced continuation");
            return;
          }
          const auto displaced_begin =
              cave_words->end() - planned_patch.displaced_tail_words.size();
          displaced_tail_words.assign(displaced_begin, cave_words->end());
          cave_words->erase(displaced_begin, cave_words->end());
        }
        if (planned_patch.branch_only_route) {
          if (!candidate_point.moi_branch_only_spill || planned_patch.entry_island_offset ||
              planned_patch.dense_call_anchor || !planned_patch.displaced_tail_words.empty()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe found inconsistent branch-only state");
            return;
          }
          if (cave_words->empty()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow branch-only body lost its guard slot");
            return;
          }
          const uint64_t guarded_word_count =
              cave_words->size() - 1u + deferred_guest_words.size() + displaced_tail_words.size();
          if (guarded_word_count > std::numeric_limits<int16_t>::max()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow branch-only guard exceeds conditional-branch range");
            return;
          }
          const auto skip_empty_wave = instrumentation::build_s_cbranch_execz(
              static_cast<int16_t>(guarded_word_count), arch);
          if (!skip_empty_wave) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its empty-wave guard");
            return;
          }
          cave_words->front() = *skip_empty_wave;
          insert_deferred_guest(cave_words->size());
          cave_words->insert(cave_words->end(), displaced_tail_words.begin(),
                             displaced_tail_words.end());
          if (cave_words->size() * sizeof(uint32_t) != planned_patch.placement.body_size) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow branch-only body size changed after placement");
            return;
          }
          const uint64_t return_source = cave_text_offset + cave_words->size() * sizeof(uint32_t);
          const uint64_t return_target =
              planned_patch.branch_only_route->return_relay_offsets.empty()
                  ? planned_patch.placement.return_target
                  : planned_patch.branch_only_route->return_relay_offsets.front();
          const auto ret = compute_sopp_branch_simm16(return_source, return_target);
          if (!ret) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow branch-only return is unreachable");
            return;
          }
          cave_words->push_back(build_s_branch(*ret, arch));
        } else if (planned_patch.dense_call_anchor) {
          if (target->direct_call_form == ConSanDirectCallForm::SCallI64) {
            empty_exec_continuation_word_offset = cave_words->size();
            insert_deferred_guest(cave_words->size());
            cave_words->insert(cave_words->end(), displaced_tail_words.begin(),
                               displaced_tail_words.end());
            const uint16_t call_return_sgpr =
                candidate_point.has_inline_moi_scalar_spill()
                    ? candidate_point.moi_router_call->call_return_sgpr
                    : static_cast<uint16_t>(*candidate_point.moi_exec_save_sgpr + 28u);
            cave_words->push_back(build_s_setpc_b64(call_return_sgpr, arch));
          } else {
            const auto jump_sgprs = moi_indirect_jump_sgprs(options, candidate_point);
            empty_exec_continuation_word_offset = cave_words->size();
            if (!jump_sgprs ||
                !append_moi_scc_preserving_indirect_jump(
                    *cave_words, cave_text_offset, planned_patch.placement.return_target,
                    jump_sgprs->pc_sgpr, jump_sgprs->scc_save_sgpr,
                    /*capture_scc=*/false, arch)) {
              result.errors.emplace_back(
                  "ConSan MOI inline-shadow dense explicit-key probe could not encode its long "
                  "return");
              return;
            }
            const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
            if (!wait_pc) {
              result.errors.emplace_back(
                  "ConSan MOI inline-shadow dense explicit-key probe could not encode its return "
                  "dependency wait");
              return;
            }
            cave_words->insert(cave_words->end() - 1, *wait_pc);
            insert_deferred_guest(cave_words->size() - 1u);
            cave_words->insert(cave_words->end() - 1, displaced_tail_words.begin(),
                               displaced_tail_words.end());
          }
        } else if (planned_patch.entry_island_offset) {
          const auto jump_sgprs = moi_indirect_jump_sgprs(options, candidate_point);
          empty_exec_continuation_word_offset = cave_words->size();
          if (!jump_sgprs) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe has no return jump scalar state");
            return;
          }
          if (!append_moi_scc_preserving_indirect_jump(
                  *cave_words, cave_text_offset, planned_patch.placement.return_target,
                  jump_sgprs->pc_sgpr, jump_sgprs->scc_save_sgpr,
                  /*capture_scc=*/false, arch)) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its long return");
            return;
          }
          const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
          if (!wait_pc) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its return dependency wait");
            return;
          }
          cave_words->insert(cave_words->end() - 1, *wait_pc);
          insert_deferred_guest(cave_words->size() - 1u);
          cave_words->insert(cave_words->end() - 1, displaced_tail_words.begin(),
                             displaced_tail_words.end());
        } else {
          const auto ret = compute_sopp_branch_simm16(planned_patch.placement.return_branch_offset,
                                                      planned_patch.placement.return_target);
          if (!ret) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow probe appended cave return is out of range");
            return;
          }
          empty_exec_continuation_word_offset = cave_words->size();
          insert_deferred_guest(cave_words->size());
          cave_words->insert(cave_words->end(), displaced_tail_words.begin(),
                             displaced_tail_words.end());
          cave_words->push_back(build_s_branch(*ret, arch));
        }
        if (planned_patch.scalar_spill && !planned_patch.branch_only_route) {
          if (!empty_exec_continuation_word_offset || *empty_exec_continuation_word_offset == 0u ||
              *empty_exec_continuation_word_offset - 1u >
                  static_cast<size_t>(std::numeric_limits<int16_t>::max())) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow scalar-spill empty-wave guard cannot reach the "
                "guest continuation");
            return;
          }
          const auto skip_empty_wave = instrumentation::build_s_cbranch_execz(
              static_cast<int16_t>(*empty_exec_continuation_word_offset - 1u), arch);
          if (!skip_empty_wave || cave_words->empty()) {
            result.errors.emplace_back(
                "ConSan MOI inline-shadow scalar-spill body could not encode its empty-wave "
                "guard");
            return;
          }
          cave_words->front() = *skip_empty_wave;
        }
        append_words_bytes(new_text, *cave_words);
        info.trampoline_offset = cave_text_offset;
        info.original_size = planned_patch.placement.original_size;
        info.trampoline_size = static_cast<uint32_t>(cave_words->size() * sizeof(uint32_t));

      } else {
        const uint64_t patch_bytes = static_cast<uint64_t>(words->size() * sizeof(uint32_t));
        if (patch_bytes != planned_patch.placement.body_size) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe final patch size changed");
          return;
        }
        if (candidate.anchor() > new_text.size() ||
            patch_bytes > new_text.size() - candidate.anchor()) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe exceeds .text");
          return;
        }
        std::memcpy(new_text.data() + candidate.anchor(), words->data(),
                    static_cast<size_t>(patch_bytes));
        info.trampoline_offset = candidate.anchor() + candidate.size();
        info.original_size = static_cast<uint32_t>(patch_bytes);
        info.trampoline_size = 0;
        if (planned_patch.guest_instruction_word_count != 0u &&
            planned_patch.guest_instruction_word_count <= words->size()) {
          info.relocated_guest_instruction_offset =
              candidate.anchor() +
              (words->size() - planned_patch.guest_instruction_word_count) * sizeof(uint32_t);
        }
      }
      auto commit = make_moi_access_lowering_commit(result.observation_plan(), candidate, info);
      if (!commit) {
        result.errors.emplace_back(
            "ConSan MOI inline-shadow probe produced an invalid intent-bound lowering");
        return;
      }
      lowering_commits.push_back(std::move(*commit));
      patches.push_back(info);
      if (planned_patch.branch_only_route) {
        std::string relay_error;
        if (!BranchOnlyRelayRouter::emit_and_record(
                new_text, *planned_patch.branch_only_route, info.trampoline_offset,
                planned_patch.placement.return_target, arch, patches, &relay_error)) {
          result.errors.emplace_back("ConSan MOI inline-shadow probe " + relay_error);
          return;
        }
      }
    }

    if (!emit_moi_dense_entry_hosts(
            dense_entry_hosts, dense_groups, options, result.moi_operating_point,
            kIndirectIslandWords,
            /*pad_to_planned_offset=*/false,
            /*use_indirect_jump_scc_save=*/true, "inline-shadow probe",
            "ConSan MOI inline-shadow probe lost its dense host scalar assignment",
            "ConSan MOI inline-shadow probe could not encode its dense entry-host return", arch,
            new_text, result.errors,
            [](const PlannedInlineShadowPatch &, const ConSanMoiOperatingPoint &) { return true; },
            [&](const MoiDenseEntryHost &host, std::vector<uint32_t> &host_words) {
              const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
              if (!wait_pc)
                return false;
              host_words.insert(host_words.end() - 1, *wait_pc);
              return host_words.size() == host.displaced_words.size() + kIndirectIslandWords;
            }))
      return;

    if (!replace_consan_text(patcher, new_text, options.patched_image_growth_limit,
                             "MOI inline-shadow probe", result.program_inventory.code_object_id(),
                             result.errors)) {
      return;
    }
    if (!result.publish_access_lowering(std::move(patcher).emit(), "ConSan MOI inline-shadow probe",
                                        std::move(lowering_commits), std::move(patches))) {
      return;
    }
    return;
  }

  if (!apply_inline_moi_access_patches(
          bytes, planned_patches, descriptor_requirements, scalar_requirements,
          private_requirements, &lds_requirements, &options, "inline-shadow probe", arch,
          [&](const PlannedInlineShadowPatch &planned_patch) {
            ConSanMoiOperatingPoint candidate_point = operating_point;
            if (!apply_moi_transient_sgpr_assignment(
                    options, candidate_point, result.moi_operating_point,
                    planned_patch.resources.owner_descriptor_file_offsets)) {
              result.errors.emplace_back(
                  "ConSan MOI inline-shadow access lost its transient scalar assignment");
              return std::optional<std::vector<uint32_t>>{};
            }
            if (!apply_moi_persistent_vgpr_assignment(
                    candidate_point, result.moi_operating_point,
                    planned_patch.resources.owner_descriptor_file_offsets)) {
              result.errors.emplace_back(
                  "ConSan MOI inline-shadow access lost its persistent VGPR assignment");
              return std::optional<std::vector<uint32_t>>{};
            }
            if (planned_patch.private_dispatch_id_offset) {
              const auto reload = moi_indirect_jump_sgprs(options, candidate_point);
              if (!reload) {
                result.errors.emplace_back(
                    "ConSan MOI inline-shadow access lost its private dispatch reload pair");
                return std::optional<std::vector<uint32_t>>{};
              }
              candidate_point.moi_dispatch_identity.set_sgpr(reload->pc_sgpr);
            }
            return build_inline_shadow_words(
                bytes, *planned_patch.candidate, options, options, candidate_point,
                planned_patch.resources.base, semantics, options.moi_partition_mask_debug, arch,
                layout, planned_patch.workgroup_sources, planned_patch.workgroup_shadow,
                requires_wave_partitioned_exact_shadow, planned_patch.private_epoch_offset,
                planned_patch.owner_derivation, planned_patch.private_workgroup_key_offset,
                planned_patch.private_dispatch_id_offset,
                planned_patch.workgroup_shadow_compact_token, planned_patch.resources.count,
                planned_patch.spill.has_value(),
                planned_patch.spill ? &*planned_patch.spill : nullptr, result.errors);
          },
          [](const PlannedInlineShadowPatch &planned_patch) {
            const ConSanMoiCandidate &candidate = *planned_patch.candidate;
            ConSanPatchInfo info;
            info.kind = ConSanPatchKind::InlineMoiExactShadowStore;
            info.anchor_offset = candidate.anchor();
            info.trampoline_offset = candidate.anchor() + candidate.size();
            info.original_size = static_cast<uint32_t>(planned_patch.placement.body_size);
            note_moi_access_patch_info(info, planned_patch);
            info.persistent_workgroup_key_private_offset =
                planned_patch.private_workgroup_key_offset;
            info.persistent_dispatch_id_private_offset = planned_patch.private_dispatch_id_offset;
            if (planned_patch.workgroup_shadow) {
              info.workgroup_shadow_base = planned_patch.workgroup_shadow->base;
              info.workgroup_shadow_size = planned_patch.workgroup_shadow->size;
              info.workgroup_shadow_validity_base = planned_patch.workgroup_shadow->validity_base;
              info.workgroup_shadow_validity_size = planned_patch.workgroup_shadow->validity_size;
              info.workgroup_shadow_lazy_initialization =
                  planned_patch.workgroup_shadow->lazy_initialization;
              info.workgroup_shadow_compact = planned_patch.workgroup_shadow->compact;
              info.workgroup_shadow_compact_token = planned_patch.workgroup_shadow_compact_token;
              info.required_group_segment_size =
                  planned_patch.workgroup_shadow->required_group_segment_size;
            }
            return info;
          },
          result))
    return;
}
