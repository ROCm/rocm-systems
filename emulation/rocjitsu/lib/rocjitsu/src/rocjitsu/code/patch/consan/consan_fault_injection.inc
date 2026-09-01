[[nodiscard]] BarrierSite barrier_site_from_inventory(const ConSanFaultSite &site);

/// Private candidate transaction for one complete set of typed fault plans.
/// Mutation mechanisms can compose through this state, but incomplete bytes,
/// proof, and tallies do not become visible through the transform artifact.
struct FaultApplicationState {
  explicit FaultApplicationState(const ConSanTransformArtifacts &result)
      : program_inventory(result.program_inventory), fault_sites(result.fault_sites),
        barrier_move_destinations(result.barrier_move_destinations),
        replacement(result.replacement), patches(result.patches), mutation(result.mutation),
        outcome(result.outcome), transform_failure_cause(result.transform_failure_cause),
        warnings(result.warnings), errors(result.errors) {}

  const ProgramInventory &program_inventory;
  std::span<const ConSanFaultSite> fault_sites;
  std::span<const ConSanBarrierMoveDestination> barrier_move_destinations;
  std::vector<uint8_t> replacement;
  std::vector<ConSanPatchInfo> patches;
  ConSanMutationOutcome mutation;
  ConSanTransformOutcome outcome = ConSanTransformOutcome::Unchanged;
  std::optional<ConSanTransformFailureCause> transform_failure_cause;
  std::vector<std::string> warnings;
  std::vector<std::string> errors;

  [[nodiscard]] bool modified() const {
    return outcome == ConSanTransformOutcome::ModifiedValid || !patches.empty();
  }
  void mark_modified() {
    if (outcome == ConSanTransformOutcome::Unchanged)
      outcome = ConSanTransformOutcome::ModifiedValid;
  }
  [[nodiscard]] bool successful() const { return errors.empty(); }
};

[[nodiscard]] ConSanFaultSelectionView fault_selection_view(const FaultApplicationState &state) {
  return {.program_inventory = state.program_inventory, .fault_sites = state.fault_sites};
}

[[nodiscard]] bool owned_by_requested_kernel(std::span<const ConSanExecutionOwner> owners,
                                             const ConSanFaultSelectionView &inventory,
                                             const ConSanOptions &options);

[[nodiscard]] const ConSanFaultSite *
select_fault_site_for_plan(const ConSanFaultSelectionView &inventory, const ConSanOptions &options,
                           ConSanFaultSiteKind kind, uint32_t legacy_index);

struct AtomicOrderBoundary {
  const ConSanSyncSequence *sequence = nullptr;
  const ConSanSyncEvent *fence = nullptr;
  std::optional<uint64_t> release_wait_text_offset;
};

[[nodiscard]] AtomicOrderBoundary resolve_atomic_order_boundary(const ProgramInventory &inventory,
                                                                const ConSanFaultSite &site,
                                                                AtomicFaultEncoding encoding,
                                                                ConSanAtomicOrderEdge edge) {
  AtomicOrderBoundary result;
  if (!site.sync_event_identity)
    return result;
  const SynchronizationInventoryView sync = inventory.sync();
  result.sequence = sync.find_unique_sequence_containing(*site.sync_event_identity);
  if (result.sequence == nullptr ||
      !consan_sync_confidence_meets(result.sequence->memory_role_confidence,
                                    ConSanSemanticConfidence::Conservative)) {
    return {};
  }
  if (edge != ConSanAtomicOrderEdge::Acquire && encoding != AtomicFaultEncoding::CdnaFlat)
    result.release_wait_text_offset = result.sequence->release_wait_text_offset;
  for (const std::string &identity : result.sequence->member_event_identities) {
    const auto event = std::ranges::find_if(sync.sync_events, [&](const auto &candidate) {
      return candidate.identity == identity && candidate.kind == ConSanSyncEventKind::Fence;
    });
    if (event != sync.sync_events.end() &&
        (edge == ConSanAtomicOrderEdge::Any ||
         (edge == ConSanAtomicOrderEdge::Release && is_release_cache_event(&*event)) ||
         (edge == ConSanAtomicOrderEdge::Acquire && is_acquire_cache_event(&*event)))) {
      result.fence = &*event;
      break;
    }
  }
  return result;
}

[[nodiscard]] std::optional<OrdinaryAcquireMutationTarget>
select_ordinary_acquire_mutation_target(const ConSanFaultSelectionView &inventory,
                                        const ConSanOptions &options) {
  return select_ordinary_acquire_mutation_target(
      inventory, ConSanFaultSelection{
                     .primary_site_identity = options.fault_site_identity,
                     .primary_sequence_identity = {},
                     .companion_site_identity = {},
                     .companion_sequence_identity = {},
                     .kernel_name_filter = options.test_kernel_name_filter,
                     .ordinal = options.fault_ordinary_index,
                 });
}

[[nodiscard]] bool
is_member_of_qualified_logical_barrier_pair(const ConSanFaultSelectionView &inventory,
                                            const ConSanFaultSite &site) {
  if (!site.sync_event_identity)
    return false;
  const ConSanSyncSequence *sequence =
      inventory.program_inventory.sync().find_unique_sequence_containing(*site.sync_event_identity);
  return sequence != nullptr && sequence->operation == ConSanSyncOperation::BarrierFull &&
         consan_sync_confidence_meets(sequence->confidence,
                                      ConSanSemanticConfidence::Conservative) &&
         sequence->member_event_identities.size() == 2u;
}

[[nodiscard]] ExactBarrierDropPairResolution
resolve_exact_barrier_drop_pair(const ConSanFaultSelectionView &inventory,
                                const ConSanOptions &options) {
  return resolve_exact_barrier_drop_pair(
      inventory, ConSanFaultSelection{
                     .primary_site_identity = options.fault_site_identity,
                     .primary_sequence_identity = options.fault_barrier_sequence_identity,
                     .companion_site_identity = {},
                     .companion_sequence_identity = {},
                     .kernel_name_filter = options.test_kernel_name_filter,
                     .ordinal = 0,
                 });
}

[[nodiscard]] ExactBarrierDropGroupResolution
resolve_exact_barrier_drop_group(const ConSanFaultSelectionView &inventory,
                                 const ConSanOptions &options) {
  return resolve_exact_barrier_drop_group(
      inventory,
      ConSanFaultSelection{
          .primary_site_identity = options.fault_site_identity,
          .primary_sequence_identity = options.fault_barrier_sequence_identity,
          .companion_site_identity = options.fault_barrier_companion_site_identity,
          .companion_sequence_identity = options.fault_barrier_companion_sequence_identity,
          .kernel_name_filter = options.test_kernel_name_filter,
          .ordinal = 0,
      });
}

[[nodiscard]] const ConSanBarrierLifecycleGroup *
find_lifecycle_group_for_sequence(const ProgramInventory &program_inventory,
                                  const ConSanSyncSequence &sequence) {
  const SynchronizationInventoryView sync = program_inventory.sync();
  const auto group = std::ranges::find_if(sync.barrier_lifecycle_groups, [&](const auto &item) {
    return std::ranges::all_of(sequence.member_event_identities, [&](const std::string &identity) {
      return std::ranges::find(item.member_event_identities, identity) !=
             item.member_event_identities.end();
    });
  });
  return group == sync.barrier_lifecycle_groups.end() ? nullptr : &*group;
}

[[nodiscard]] const ConSanBarrierSite *
find_static_participant_init(const ProgramInventory &program_inventory,
                             const ConSanBarrierLifecycleGroup &group) {
  const SynchronizationInventoryView sync = program_inventory.sync();
  const auto init_event = std::ranges::find_if(sync.sync_events, [&](const auto &event) {
    return event.operation == ConSanSyncOperation::BarrierInit &&
           event.container_name == group.container_name && event.in_kernel == group.in_kernel &&
           std::ranges::find(group.member_event_identities, event.identity) !=
               group.member_event_identities.end();
  });
  if (init_event == sync.sync_events.end())
    return nullptr;
  const auto find_in = [&](const auto &containers) -> const ConSanBarrierSite * {
    for (const auto &container : containers) {
      if (container.name != group.container_name)
        continue;
      const auto site = std::ranges::find(container.barrier_sites, init_event->text_offset,
                                          &ConSanBarrierSite::text_offset);
      if (site != container.barrier_sites.end())
        return &*site;
    }
    return nullptr;
  };
  return group.in_kernel ? find_in(program_inventory.kernels())
                         : find_in(program_inventory.functions());
}

void mark_participant_mutation_unsupported(ConSanFaultPlanningResult &result,
                                           std::string_view reason) {
  result.unsupported = true;
  result.warnings.emplace_back("ConSan barrier participant mutation is unsupported: " +
                               std::string(reason));
}

void mark_participant_mutation_unsupported(FaultApplicationState &result, std::string_view reason) {
  result.outcome = ConSanTransformOutcome::Unsupported;
  result.warnings.emplace_back("ConSan barrier participant mutation is unsupported: " +
                               std::string(reason));
}

[[nodiscard]] size_t applied_fault_mutation_count(std::span<const ConSanPatchInfo> patches) {
  size_t count = 0;
  size_t move_targets = 0;
  size_t barrier_id_scope_members = 0;
  size_t exact_barrier_drop_members = 0;
  std::vector<std::string> atomic_order_mutations;
  const auto note_patch = [&](const ConSanCommittedPatchGeometry &patch,
                              const ConSanPatchFaultProof &fault) {
    if (patch.phase != ConSanPatchPhase::Mutation)
      return;
    switch (patch.kind) {
    case ConSanPatchKind::InlineBarrierNopRewrite:
      if (!fault.fault_sequence_identity.empty()) {
        ++exact_barrier_drop_members;
      } else {
        ++count;
      }
      break;
    case ConSanPatchKind::InlineBarrierParticipantCountRewrite:
    case ConSanPatchKind::InlineAtomicAddressRewrite:
      ++count;
      break;
    case ConSanPatchKind::InlineAtomicOrderRewrite:
      if (fault.fault_sequence_identity.empty()) {
        ++count;
      } else if (std::ranges::find(atomic_order_mutations, fault.fault_sequence_identity) ==
                 atomic_order_mutations.end()) {
        atomic_order_mutations.push_back(fault.fault_sequence_identity);
      }
      break;
    case ConSanPatchKind::InlineAtomicScopeRewrite:
    case ConSanPatchKind::InlineLdsAddressRewrite:
    case ConSanPatchKind::InlineOrdinaryAddressRewrite:
    case ConSanPatchKind::InlineOrdinaryOrderRewrite:
    case ConSanPatchKind::InlineOrdinaryScopeRewrite:
      ++count;
      break;
    case ConSanPatchKind::InlineBarrierIdScopeRewrite:
      ++barrier_id_scope_members;
      break;
    case ConSanPatchKind::InlineBarrierMoveTargetRewrite:
      ++move_targets;
      break;
    default:
      break;
    }
  };
  for (const auto &patch : patches)
    note_patch(patch, patch);
  // One whole-barrier movement currently has two target records.
  count += (move_targets + 1u) / 2u;
  // ID/scope rewriting is transactional: all member patch records represent
  // one selected logical pair or lifecycle mutation.
  count += static_cast<size_t>(barrier_id_scope_members != 0u);
  // An exact whole-barrier drop rewrites both physical members transactionally.
  count += static_cast<size_t>(exact_barrier_drop_members != 0u);
  count += atomic_order_mutations.size();
  return count;
}

ConSanFaultPlanningResult build_fault_mutation_plan(const ConSanOptions &options,
                                                    rj_code_arch_t arch,
                                                    const ConSanFaultPlanningInput &input,
                                                    bool require_applicable_plan) {
  ConSanFaultPlanningResult result;
  const ProgramInventory &program_inventory = input.selection.program_inventory;
  const SynchronizationInventoryView sync = program_inventory.sync();
  const auto reject_application = [&](std::string dry_run_warning, std::string application_error) {
    if (require_applicable_plan)
      result.errors.push_back(std::move(application_error));
    else
      result.warnings.push_back(std::move(dry_run_warning));
  };
  const auto make_named_plan = [&](ConSanFaultMutationKind kind, std::string primary_identity) {
    ConSanFaultMutationPlan plan;
    plan.source_code_object = program_inventory.code_object_id();
    plan.kind = kind;
    plan.primary_identity = std::move(primary_identity);
    return plan;
  };
  const auto make_plan = [&](ConSanFaultMutationKind kind, const ConSanFaultSite &site) {
    return make_named_plan(kind, site.identity);
  };
  auto append_primary = [&](ConSanFaultMutationKind kind, const ConSanFaultSite *site) {
    if (site == nullptr) {
      result.warnings.emplace_back("ConSan fault dry run found no site for the requested mutation");
      return;
    }
    result.fault_plans.push_back(make_plan(kind, *site));
  };

  if (options.fault_drop_barrier && has_only_rocclr_runtime_kernels(program_inventory)) {
    result.warnings.emplace_back("ConSan barrier fault skipped ROCclr runtime helper code object");
  }

  if (options.fault_drop_barrier && !has_only_rocclr_runtime_kernels(program_inventory)) {
    const ConSanFaultSite *primary = select_fault_site_for_plan(
        input.selection, options, ConSanFaultSiteKind::Barrier, options.fault_barrier_index);
    if (!options.fault_barrier_sequence_identity.empty() ||
        !options.fault_barrier_companion_site_identity.empty() ||
        !options.fault_barrier_companion_sequence_identity.empty()) {
      const bool grouped = !options.fault_barrier_companion_site_identity.empty() ||
                           !options.fault_barrier_companion_sequence_identity.empty();
      std::optional<ExactBarrierDropGroup> group;
      std::optional<ExactBarrierDropPair> pair;
      std::string reason;
      if (grouped) {
        const auto resolution = resolve_exact_barrier_drop_group(input.selection, options);
        group = resolution.group;
        reason = exact_barrier_drop_group_issue_message(resolution.issue, resolution.member_issue);
      } else {
        const auto resolution = resolve_exact_barrier_drop_pair(input.selection, options);
        pair = resolution.pair;
        reason = exact_barrier_drop_pair_issue_message(resolution.issue);
      }
      if ((grouped && !group) || (!grouped && !pair)) {
        result.warnings.emplace_back("ConSan exact whole-barrier drop dry run rejected: " + reason);
      } else {
        ConSanFaultMutationPlan plan = make_plan(ConSanFaultMutationKind::DropBarrier,
                                                 group ? *group->first.primary : *pair->primary);
        if (group) {
          // For a grouped drop this identity selects the second logical pair;
          // every physical member of both pairs remains in semantic order
          // below, so the plan needs no second copy of either pair's inferred
          // completing member.
          plan.companion_identity = group->second.primary->identity;
          plan.logical_sequence_identity = exact_barrier_drop_group_identity(*group);
          plan.selection_sequence_identity = group->first.sequence->identity;
          plan.companion_selection_sequence_identity = group->second.sequence->identity;
          plan.ordered_member_identities = group->first.sequence->member_event_identities;
          plan.ordered_member_identities.insert(
              plan.ordered_member_identities.end(),
              group->second.sequence->member_event_identities.begin(),
              group->second.sequence->member_event_identities.end());
        } else {
          plan.companion_identity = pair->companion->identity;
          plan.logical_sequence_identity = pair->sequence->identity;
          plan.selection_sequence_identity = pair->sequence->identity;
          plan.ordered_member_identities = pair->sequence->member_event_identities;
        }
        result.fault_plans.push_back(std::move(plan));
      }
    } else if (primary != nullptr &&
               is_member_of_qualified_logical_barrier_pair(input.selection, *primary) &&
               !options.fault_allow_destructive_incomplete_barrier_drop) {
      result.warnings.emplace_back(
          "ConSan fault dry run rejected dropping one half of a qualified logical barrier pair; "
          "explicit destructive incomplete-drop mode is required");
    } else {
      if (primary != nullptr) {
        ConSanFaultMutationPlan plan = make_plan(ConSanFaultMutationKind::DropBarrier, *primary);
        plan.destructive_incomplete_barrier_drop =
            is_member_of_qualified_logical_barrier_pair(input.selection, *primary);
        result.fault_plans.push_back(std::move(plan));
      } else {
        append_primary(ConSanFaultMutationKind::DropBarrier, primary);
      }
    }
  }
  if (options.fault_move_barrier) {
    const ConSanFaultSite *primary = select_fault_site_for_plan(
        input.selection, options, ConSanFaultSiteKind::Barrier, options.fault_barrier_index);
    const ConSanSyncSequence *sequence =
        primary != nullptr && primary->sync_event_identity
            ? program_inventory.sync().find_unique_sequence_containing(
                  *primary->sync_event_identity)
            : nullptr;
    if (primary == nullptr || sequence == nullptr ||
        !owned_by_requested_kernel(sequence->execution_owners, input.selection, options) ||
        sequence->operation != ConSanSyncOperation::BarrierFull ||
        !consan_sync_confidence_meets(sequence->confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        sequence->member_event_identities.size() != 2u) {
      result.warnings.emplace_back("ConSan fault dry run found no barrier pair to move");
    } else {
      const auto companion =
          std::ranges::find_if(input.selection.fault_sites, [&](const auto &candidate) {
            return candidate.kind == ConSanFaultSiteKind::Barrier &&
                   candidate.sync_event_identity &&
                   owned_by_requested_kernel(candidate.execution_owners, input.selection,
                                             options) &&
                   candidate.identity != primary->identity &&
                   std::ranges::find(sequence->member_event_identities,
                                     *candidate.sync_event_identity) !=
                       sequence->member_event_identities.end();
          });
      if (companion == input.selection.fault_sites.end()) {
        result.warnings.emplace_back(
            "ConSan fault dry run found no completing barrier in the selected container");
      } else {
        ConSanFaultMutationPlan plan =
            make_plan(ConSanFaultMutationKind::MoveBarrierPair, *primary);
        plan.companion_identity = companion->identity;
        plan.logical_sequence_identity = sequence->identity;
        plan.selection_sequence_identity = sequence->identity;
        plan.ordered_member_identities = sequence->member_event_identities;
        plan.barrier_move_direction = options.fault_barrier_move_direction;
        if (options.fault_barrier_move_direction == ConSanBarrierMoveDirection::LegacyMarker) {
          result.fault_plans.push_back(std::move(plan));
        } else if (options.fault_barrier_destination_identity.empty()) {
          result.warnings.emplace_back(
              "ConSan fault dry run markerless barrier move requires an exact destination "
              "identity");
        } else {
          const auto destination = std::ranges::find(input.barrier_move_destinations,
                                                     options.fault_barrier_destination_identity,
                                                     &ConSanBarrierMoveDestination::identity);
          if (destination == input.barrier_move_destinations.end()) {
            result.warnings.emplace_back(
                "ConSan fault dry run found no barrier-move destination with exact identity '" +
                options.fault_barrier_destination_identity + "'");
          } else if (!owned_by_requested_kernel(destination->execution_owners, input.selection,
                                                options)) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected barrier-move destination outside the requested "
                "dispatch owner");
          } else if (!sequence->basic_block_index ||
                     destination->container_name != sequence->container_name ||
                     destination->in_kernel != sequence->in_kernel) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected barrier-move destination across a control-flow "
                "or container boundary");
          } else if (destination->basic_block_index != *sequence->basic_block_index &&
                     !consan_fault_admits_cross_block_barrier_move(*destination, options)) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected control-flow or container boundary: cross-block "
                "barrier move lacks a matching proven completing or destructive structured "
                "diamond opt-in");
          } else if (destination->text_offset < sequence->end_text_offset &&
                     sequence->begin_text_offset < destination->text_offset + destination->size) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected barrier-move destination overlapping the "
                "selected logical barrier");
          } else if (!destination->suitable()) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected unsuitable barrier-move destination: " +
                consan_barrier_move_destination_issue_message(destination->issue,
                                                              destination->issue_detail));
          } else if (options.fault_barrier_move_direction == ConSanBarrierMoveDirection::Earlier &&
                     destination->text_offset + destination->size > sequence->begin_text_offset) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected a destination that is not earlier than the "
                "selected logical barrier");
          } else if (options.fault_barrier_move_direction == ConSanBarrierMoveDirection::Later &&
                     destination->text_offset < sequence->end_text_offset) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected a destination that is not later than the "
                "selected logical barrier");
          } else {
            plan.destination_identity = destination->identity;
            if (destination->basic_block_index != *sequence->basic_block_index) {
              plan.barrier_move_cfg_contract = destination->cfg_contract;
              plan.structured_guard_block_index = destination->structured_guard_block_index;
              plan.structured_destination_block_index = destination->basic_block_index;
              plan.structured_source_block_index = destination->structured_source_block_index;
              plan.structured_guard_offset = destination->structured_guard_offset;
              plan.structured_destination_offset = destination->text_offset;
              plan.structured_source_offset = destination->structured_source_offset;
            }
            result.fault_plans.push_back(std::move(plan));
          }
        }
      }
    }
  }
  if (options.fault_mutate_barrier_id_scope) {
    const ConSanSyncSequence *sequence = nullptr;
    if (options.fault_barrier_sequence_identity.empty()) {
      reject_application(
          "ConSan barrier ID/scope dry run requires an exact logical sequence identity",
          "ConSan barrier ID/scope mutation requires an exact sequence identity and target ID");
    } else {
      const auto selected =
          std::ranges::find(sync.sync_sequences, options.fault_barrier_sequence_identity,
                            &ConSanSyncSequence::identity);
      if (selected == sync.sync_sequences.end()) {
        reject_application(
            "ConSan barrier ID/scope dry run found no logical sequence with exact identity '" +
                options.fault_barrier_sequence_identity + "'",
            "ConSan barrier ID/scope mutation requires an exact qualified two-member sequence");
      } else {
        sequence = &*selected;
      }
    }

    const ConSanBarrierSite::Scope target_scope =
        options.fault_barrier_target_id ? barrier_scope_for_id(*options.fault_barrier_target_id)
                                        : ConSanBarrierSite::Scope::Unknown;
    if (sequence == nullptr) {
      // The exact-identity diagnostic above is sufficient.
    } else if (sequence->kind != ConSanSyncSequenceKind::Barrier ||
               sequence->operation != ConSanSyncOperation::BarrierFull ||
               !consan_sync_confidence_meets(sequence->confidence,
                                             ConSanSemanticConfidence::Conservative) ||
               sequence->member_event_identities.size() != 2u || !sequence->barrier_id ||
               sequence->barrier_scope == ConSanBarrierSite::Scope::Unknown) {
      result.warnings.emplace_back(
          "ConSan barrier ID/scope dry run rejected an unqualified logical barrier sequence");
    } else if (const ConSanBarrierLifecycleGroup *group =
                   find_lifecycle_group_for_sequence(program_inventory, *sequence);
               group != nullptr && !group->admissible()) {
      result.warnings.emplace_back(
          "ConSan barrier ID/scope dry run rejected a partial unresolved lifecycle mutation: " +
          std::string(consan_barrier_lifecycle_issue_message(group->issue)));
    } else if (!options.fault_barrier_target_id) {
      result.warnings.emplace_back("ConSan barrier ID/scope dry run requires a target barrier ID");
    } else if (target_scope == ConSanBarrierSite::Scope::Unknown) {
      result.warnings.emplace_back(
          "ConSan barrier ID/scope dry run rejected a target ID with unknown scope");
    } else if (*options.fault_barrier_target_id == *sequence->barrier_id) {
      result.warnings.emplace_back(
          "ConSan barrier ID/scope dry run requires a target different from the original ID");
    } else if (const ConSanBarrierLifecycleGroup *group =
                   find_lifecycle_group_for_sequence(program_inventory, *sequence);
               group != nullptr &&
               (*options.fault_barrier_target_id < 1 || *options.fault_barrier_target_id > 16)) {
      result.warnings.emplace_back(
          "ConSan barrier lifecycle retarget requires a named-barrier target ID in [1, 16]");
    } else {
      std::array<const ConSanFaultSite *, 2> members{};
      for (size_t i = 0; i < members.size(); ++i) {
        const std::string &event_identity = sequence->member_event_identities[i];
        const auto member =
            std::ranges::find_if(input.selection.fault_sites, [&](const auto &candidate) {
              return candidate.kind == ConSanFaultSiteKind::Barrier &&
                     candidate.sync_event_identity == event_identity;
            });
        if (member != input.selection.fault_sites.end())
          members[i] = &*member;
      }
      if (members[0] == nullptr || members[1] == nullptr) {
        result.warnings.emplace_back(
            "ConSan barrier ID/scope dry run could not resolve both sequence members");
      } else {
        const ConSanBarrierLifecycleGroup *group =
            find_lifecycle_group_for_sequence(program_inventory, *sequence);
        ConSanFaultMutationPlan plan =
            make_plan(ConSanFaultMutationKind::BarrierIdScope, *members[0]);
        plan.companion_identity = members[1]->identity;
        plan.logical_sequence_identity = group != nullptr ? group->identity : sequence->identity;
        plan.selection_sequence_identity = sequence->identity;
        plan.ordered_member_identities =
            group != nullptr ? group->member_event_identities : sequence->member_event_identities;
        plan.original_barrier_id = sequence->barrier_id;
        plan.target_barrier_id = options.fault_barrier_target_id;
        plan.original_barrier_scope = sequence->barrier_scope;
        plan.target_barrier_scope = target_scope;
        result.fault_plans.push_back(std::move(plan));
      }
    }
  }
  if (options.fault_mutate_barrier_participants) {
    const auto selected =
        std::ranges::find(sync.sync_sequences, options.fault_barrier_sequence_identity,
                          &ConSanSyncSequence::identity);
    const ConSanBarrierLifecycleGroup *group =
        selected == sync.sync_sequences.end()
            ? nullptr
            : find_lifecycle_group_for_sequence(program_inventory, *selected);
    const ConSanBarrierSite *init = group != nullptr && group->admissible()
                                        ? find_static_participant_init(program_inventory, *group)
                                        : nullptr;
    if (options.fault_barrier_target_participant_mask) {
      mark_participant_mutation_unsupported(
          result, "the GFX12.5 lifecycle exposes a six-bit count in M0, not a participant mask");
    } else if (group == nullptr || !group->admissible() || init == nullptr ||
               init->participant_encoding != ConSanBarrierSite::ParticipantEncoding::M0Literal32 ||
               !init->participant_count || !init->participant_setup_text_offset ||
               !init->participant_setup_file_offset || !init->participant_setup_literal) {
      mark_participant_mutation_unsupported(
          result, "no complete lifecycle has an immediately preceding literal M0 count setup");
    } else if (!options.fault_barrier_target_participant_count ||
               *options.fault_barrier_target_participant_count == 0u ||
               *options.fault_barrier_target_participant_count > 63u ||
               *options.fault_barrier_target_participant_count == *init->participant_count) {
      mark_participant_mutation_unsupported(
          result, "the target count must be a distinct value in the encodable range [1, 63]");
    } else {
      ConSanFaultMutationPlan plan = make_named_plan(
          ConSanFaultMutationKind::BarrierParticipantCount, group->member_event_identities.front());
      plan.logical_sequence_identity = group->identity;
      plan.selection_sequence_identity = selected->identity;
      plan.ordered_member_identities = group->member_event_identities;
      plan.original_barrier_id = group->barrier_id;
      plan.target_barrier_id = group->barrier_id;
      plan.original_barrier_scope = group->barrier_scope;
      plan.target_barrier_scope = group->barrier_scope;
      plan.original_participant_count = init->participant_count;
      plan.target_participant_count = options.fault_barrier_target_participant_count;
      result.fault_plans.push_back(std::move(plan));
    }
  }
  const ConSanFaultSite *atomic = nullptr;
  if (options.fault_atomic_wrong_address || options.fault_atomic_weaken_order ||
      options.fault_atomic_weaken_scope) {
    atomic = select_fault_site_for_plan(input.selection, options, ConSanFaultSiteKind::Atomic,
                                        options.fault_atomic_index);
    if (atomic == nullptr) {
      if (!options.fault_site_identity.empty()) {
        result.warnings.emplace_back("ConSan atomic fault found no site with exact identity '" +
                                     options.fault_site_identity + "'");
      } else {
        result.warnings.emplace_back(
            "ConSan atomic fault found no patchable flat/global atomic at index " +
            std::to_string(options.fault_atomic_index));
      }
    }
  }
  if (options.fault_atomic_wrong_address) {
    const AtomicFaultEncoding encoding =
        atomic == nullptr
            ? AtomicFaultEncoding::Unsupported
            : classify_consan_atomic_fault_encoding(atomic->mnemonic, atomic->size, arch);
    if (atomic == nullptr) {
      // The exact missing-site diagnostic above is authoritative.
    } else if (!consan_atomic_fault_supports_address(encoding)) {
      reject_application(
          "ConSan fault dry run cannot rewrite the address for the selected atomic encoding",
          "ConSan atomic address fault is unsupported for the selected atomic encoding");
    } else if (options.fault_atomic_address_delta == 0u ||
               options.fault_atomic_address_delta % sizeof(uint32_t) != 0u ||
               options.fault_atomic_address_delta > 0x7fffffu) {
      result.errors.emplace_back(
          "ConSan atomic address delta must be a positive aligned signed-24-bit offset");
    } else {
      ConSanFaultMutationPlan plan =
          make_plan(ConSanFaultMutationKind::AtomicWrongAddress, *atomic);
      plan.address_delta_bytes = options.fault_atomic_address_delta;
      result.fault_plans.push_back(std::move(plan));
    }
  }
  if (options.fault_atomic_weaken_order) {
    const AtomicFaultEncoding encoding =
        atomic == nullptr
            ? AtomicFaultEncoding::Unsupported
            : classify_consan_atomic_fault_encoding(atomic->mnemonic, atomic->size, arch);
    const AtomicOrderBoundary boundary =
        atomic != nullptr && consan_atomic_fault_supports_order(encoding)
            ? resolve_atomic_order_boundary(program_inventory, *atomic, encoding,
                                            options.fault_atomic_order_edge)
            : AtomicOrderBoundary{};
    if (atomic == nullptr) {
      // The exact missing-site diagnostic above is authoritative.
    } else if (!consan_atomic_fault_supports_order(encoding)) {
      reject_application(
          "ConSan fault dry run cannot weaken order for the selected atomic encoding",
          "ConSan atomic order fault is unsupported for the selected atomic encoding");
    } else if (boundary.fence == nullptr && !boundary.release_wait_text_offset) {
      result.warnings.emplace_back(
          "ConSan fault dry run found no qualified atomic ordering boundary to weaken");
    } else {
      ConSanFaultMutationPlan plan = make_plan(ConSanFaultMutationKind::AtomicWeakenOrder, *atomic);
      plan.companion_identity =
          boundary.fence != nullptr
              ? std::optional<std::string>(boundary.fence->identity)
              : std::optional<std::string>(boundary.sequence->identity + "|release-wait-boundary");
      plan.logical_sequence_identity = boundary.sequence->identity;
      plan.selection_sequence_identity = boundary.sequence->identity;
      plan.ordered_member_identities = boundary.sequence->member_event_identities;
      plan.atomic_order_edge = options.fault_atomic_order_edge;
      result.fault_plans.push_back(std::move(plan));
    }
  }
  if (options.fault_atomic_weaken_scope) {
    const AtomicFaultEncoding encoding =
        atomic == nullptr
            ? AtomicFaultEncoding::Unsupported
            : classify_consan_atomic_fault_encoding(atomic->mnemonic, atomic->size, arch);
    if (atomic == nullptr) {
      // The exact missing-site diagnostic above is authoritative.
    } else if (!consan_atomic_fault_supports_scope(encoding)) {
      reject_application(
          "ConSan fault dry run cannot weaken scope for the selected atomic encoding",
          "ConSan atomic scope fault is unsupported for the selected atomic encoding");
    } else {
      append_primary(ConSanFaultMutationKind::AtomicWeakenScope, atomic);
    }
  }
  if (options.fault_lds_wrong_address) {
    const ConSanFaultSite *lds = select_fault_site_for_plan(
        input.selection, options, ConSanFaultSiteKind::LdsAccess, options.fault_lds_index);
    if (lds == nullptr) {
      append_primary(ConSanFaultMutationKind::LdsWrongAddress, lds);
    } else if (!options.fault_lds_address_vgpr) {
      result.errors.emplace_back("ConSan LDS address fault requires an explicit replacement VGPR");
    } else if (!lds->address_vgpr || lds->address_vgpr == options.fault_lds_address_vgpr) {
      result.errors.emplace_back(
          "ConSan LDS address fault requires a distinct 8-bit replacement VGPR");
    } else {
      ConSanFaultMutationPlan plan = make_plan(ConSanFaultMutationKind::LdsWrongAddress, *lds);
      plan.original_address_vgpr = lds->address_vgpr;
      plan.target_address_vgpr = options.fault_lds_address_vgpr;
      result.fault_plans.push_back(std::move(plan));
    }
  }
  if (options.fault_ordinary_wrong_address || options.fault_ordinary_weaken_order ||
      options.fault_ordinary_weaken_scope) {
    const auto target = select_ordinary_acquire_mutation_target(input.selection, options);
    if (!target) {
      result.warnings.emplace_back(
          "ConSan fault dry run found no exact supported ordinary acquire-load sequence");
    } else {
      const auto append_ordinary = [&](ConSanFaultMutationKind kind,
                                       std::optional<std::string> companion) {
        if (kind == ConSanFaultMutationKind::OrdinaryWrongAddress &&
            (options.fault_ordinary_address_delta == 0u ||
             options.fault_ordinary_address_delta % sizeof(uint32_t) != 0u ||
             options.fault_ordinary_address_delta > 0x7fffffu)) {
          result.errors.emplace_back(
              "ConSan ordinary address delta must be a positive aligned signed-24-bit offset");
          return;
        }
        ConSanFaultMutationPlan plan = make_plan(kind, *target->site);
        plan.companion_identity = std::move(companion);
        plan.logical_sequence_identity = target->sequence->identity;
        plan.selection_sequence_identity = target->sequence->identity;
        plan.ordered_member_identities = target->sequence->member_event_identities;
        if (kind == ConSanFaultMutationKind::OrdinaryWrongAddress)
          plan.address_delta_bytes = options.fault_ordinary_address_delta;
        result.fault_plans.push_back(std::move(plan));
      };
      if (options.fault_ordinary_wrong_address)
        append_ordinary(ConSanFaultMutationKind::OrdinaryWrongAddress, std::nullopt);
      if (options.fault_ordinary_weaken_order)
        append_ordinary(ConSanFaultMutationKind::OrdinaryWeakenOrder, target->cache->identity);
      if (options.fault_ordinary_weaken_scope)
        append_ordinary(ConSanFaultMutationKind::OrdinaryWeakenScope, std::nullopt);
    }
  }
  if (std::ranges::any_of(result.fault_plans, [](const ConSanFaultMutationPlan &plan) {
        return !plan.well_formed();
      })) {
    result.errors.emplace_back(
        "ConSan fault planner produced an internally inconsistent typed mutation plan");
  }
  if (options.fault_require_exactly_one && !input.prior_attempt_unsupported &&
      !result.unsupported && result.fault_plans.size() != 1u) {
    result.errors.emplace_back("ConSan fault dry run required exactly one planned mutation, got " +
                               std::to_string(result.fault_plans.size()));
  }
  return result;
}

[[nodiscard]] bool has_proof_nop_candidate(const ProgramInventory &inventory) {
  for (const ConSanKernelInfo &kernel : inventory.kernels()) {
    if (kernel.preflight_action == ConSanPreflightAction::Candidate)
      return true;
  }
  return false;
}

void try_apply_proof_nop_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                               bool force_trampoline, ConSanTransformArtifacts &result) {
  if (has_only_rocclr_runtime_kernels(result.program_inventory)) {
    result.warnings.emplace_back(
        "ConSan proof NOP patch skipped ROCclr runtime helper code object");
    return;
  }
  if (!has_proof_nop_candidate(result.program_inventory)) {
    result.warnings.emplace_back(
        "ConSan proof NOP patch skipped code object without supported DBI candidate "
        "sites");
    return;
  }

  if (!force_trampoline) {
    if (auto nop_site = find_existing_nop_site(code_object, arch)) {
      const uint32_t replacement_nop = build_s_nop(1, arch);
      if (!rewrite_word_in_place(code_object, nop_site->file_offset, replacement_nop, result))
        return;

      ConSanPatchInfo info;
      info.kind = ConSanPatchKind::InlineNopRewrite;
      info.anchor_offset = nop_site->text_offset;
      info.trampoline_offset = nop_site->text_offset;
      info.original_size = sizeof(replacement_nop);
      result.patches.push_back(info);
      result.mark_modified();
      return;
    }
  }

  std::string error;
  auto anchor = find_first_relocatable_anchor(code_object, arch, &error);
  if (!anchor) {
    result.errors.push_back("ConSan proof NOP patch skipped: " + error);
    return;
  }

  Instrumentor instrumentor(code_object, arch);
  instrumentor.add_point_by_offset(*anchor);
  auto patched = instrumentor.patch_with_debug_summaries();
  if (!patched.errors.empty()) {
    result.errors.push_back("ConSan proof NOP patch failed: " + patched.errors.front());
    return;
  }
  if (patched.elf_bytes.empty()) {
    result.errors.emplace_back("ConSan proof NOP patch produced empty ELF bytes");
    return;
  }

  result.replacement = std::move(patched.elf_bytes);
  result.mark_modified();
  result.patches.reserve(patched.patches.size());
  for (const InstrumentationPatch &patch : patched.patches) {
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineNop;
    info.anchor_offset = patch.anchor_offset;
    info.trampoline_offset = patch.trampoline_offset;
    info.original_size = patch.original_size;
    info.trampoline_size = patch.trampoline_size;
    result.patches.push_back(info);
  }
}

static void try_apply_barrier_drop_fault_patch(const AmdGpuCodeObject &code_object,
                                               rj_code_arch_t arch,
                                               const ConSanFaultMutationPlan &plan,
                                               FaultApplicationState &result) {
  if (has_only_rocclr_runtime_kernels(result.program_inventory)) {
    result.warnings.emplace_back("ConSan barrier fault skipped ROCclr runtime helper code object");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());

  if (plan.selection_sequence_identity) {
    const bool grouped = plan.companion_selection_sequence_identity.has_value();
    const ConSanFaultSelection selection = {
        .primary_site_identity = plan.primary_identity,
        .primary_sequence_identity = *plan.selection_sequence_identity,
        .companion_site_identity = plan.companion_identity
                                       ? std::string_view(*plan.companion_identity)
                                       : std::string_view{},
        .companion_sequence_identity =
            plan.companion_selection_sequence_identity
                ? std::string_view(*plan.companion_selection_sequence_identity)
                : std::string_view{},
        .kernel_name_filter = {},
        .ordinal = 0,
    };
    std::optional<ExactBarrierDropGroup> group;
    std::optional<ExactBarrierDropPair> pair;
    std::string reason;
    if (grouped) {
      const auto resolution =
          resolve_exact_barrier_drop_group(fault_selection_view(result), selection);
      group = resolution.group;
      reason = exact_barrier_drop_group_issue_message(resolution.issue, resolution.member_issue);
    } else {
      const auto resolution =
          resolve_exact_barrier_drop_pair(fault_selection_view(result), selection);
      pair = resolution.pair;
      reason = exact_barrier_drop_pair_issue_message(resolution.issue);
    }
    if ((grouped && !group) || (!grouped && !pair)) {
      result.warnings.emplace_back("ConSan exact whole-barrier drop rejected: " + reason);
      return;
    }
    std::vector<const ConSanFaultSite *> members;
    if (group) {
      members = {group->first.primary, group->first.companion, group->second.primary,
                 group->second.companion};
    } else {
      members = {pair->primary, pair->companion};
    }
    std::ranges::sort(members, {}, &ConSanFaultSite::text_offset);
    for (const ConSanFaultSite *member : members) {
      if (member->file_offset > original_bytes.size() ||
          sizeof(uint32_t) > original_bytes.size() - member->file_offset) {
        result.errors.emplace_back("ConSan exact whole-barrier drop member is outside ELF bytes");
        return;
      }
    }

    if (result.replacement.empty())
      result.replacement.assign(original_bytes.begin(), original_bytes.end());
    const uint32_t replacement_nop = build_s_nop(0, arch);
    for (const ConSanFaultSite *member : members) {
      const ExactBarrierDropPair &owning_pair =
          group && (member == group->second.primary || member == group->second.companion)
              ? group->second
              : (group ? group->first : *pair);
      std::memcpy(result.replacement.data() + member->file_offset, &replacement_nop,
                  sizeof(replacement_nop));
      ConSanPatchInfo info;
      info.phase = ConSanPatchPhase::Mutation;
      info.kind = ConSanPatchKind::InlineBarrierNopRewrite;
      info.anchor_offset = member->text_offset;
      info.trampoline_offset = member->text_offset;
      info.original_size = member->size;
      info.fault_primary_identity = owning_pair.primary->identity;
      info.fault_companion_identity = owning_pair.companion->identity;
      info.fault_sequence_identity = owning_pair.sequence->identity;
      result.patches.push_back(std::move(info));
    }
    result.mark_modified();
    result.mutation.applied_fault_logical_identity =
        group ? exact_barrier_drop_group_identity(*group) : pair->sequence->identity;
    result.warnings.emplace_back(group ? "ConSan exact grouped-barrier fault transactionally "
                                         "rewrote four physical members"
                                       : "ConSan exact whole-barrier fault transactionally rewrote "
                                         "both logical members");
    return;
  }

  const ConSanFaultSite *selected_inventory_site = find_fault_site_by_identity(
      fault_selection_view(result), plan.primary_identity, ConSanFaultSiteKind::Barrier);
  if (selected_inventory_site == nullptr) {
    result.errors.emplace_back("ConSan barrier fault plan no longer matches its pristine site");
    return;
  }
  const BarrierSite site = barrier_site_from_inventory(*selected_inventory_site);
  if (is_member_of_qualified_logical_barrier_pair(fault_selection_view(result),
                                                  *selected_inventory_site) &&
      !plan.destructive_incomplete_barrier_drop) {
    result.warnings.emplace_back(
        "ConSan barrier fault rejected dropping one half of a qualified logical barrier pair; "
        "set explicit destructive incomplete-drop mode only in a contained run");
    return;
  }

  if (result.replacement.empty())
    result.replacement.assign(original_bytes.begin(), original_bytes.end());

  const uint32_t replacement_nop = build_s_nop(0, arch);
  if (site.file_offset > result.replacement.size() ||
      sizeof(replacement_nop) > result.replacement.size() - site.file_offset) {
    result.errors.emplace_back("ConSan barrier fault offset is outside ELF bytes");
    if (!result.modified())
      result.replacement.clear();
    return;
  }

  std::memcpy(result.replacement.data() + site.file_offset, &replacement_nop,
              sizeof(replacement_nop));

  ConSanPatchInfo info;
  info.phase = ConSanPatchPhase::Mutation;
  info.kind = ConSanPatchKind::InlineBarrierNopRewrite;
  info.anchor_offset = site.text_offset;
  info.trampoline_offset = site.text_offset;
  info.original_size = site.size;
  result.patches.push_back(info);
  result.mark_modified();
  result.warnings.emplace_back("ConSan barrier fault rewrote " + site.mnemonic + " in " +
                               site.range_name);
}

[[nodiscard]] std::optional<uint8_t> inline_barrier_selector_for_id(int32_t barrier_id) {
  if (barrier_id >= 0 && barrier_id <= 64)
    return static_cast<uint8_t>(128 + barrier_id);
  if (barrier_id >= -16 && barrier_id < 0)
    return static_cast<uint8_t>(192 - barrier_id);
  return std::nullopt;
}

static void try_apply_barrier_id_scope_fault_patch(const AmdGpuCodeObject &code_object,
                                                   const ConSanFaultMutationPlan &plan,
                                                   FaultApplicationState &result) {
  const SynchronizationInventoryView sync = result.program_inventory.sync();
  if (!plan.selection_sequence_identity || !plan.target_barrier_id) {
    result.errors.emplace_back(
        "ConSan barrier ID/scope mutation requires an exact sequence identity and target ID");
    return;
  }
  const auto selected = std::ranges::find(sync.sync_sequences, *plan.selection_sequence_identity,
                                          &ConSanSyncSequence::identity);
  if (selected == sync.sync_sequences.end() || selected->kind != ConSanSyncSequenceKind::Barrier ||
      selected->operation != ConSanSyncOperation::BarrierFull ||
      !consan_sync_confidence_meets(selected->confidence, ConSanSemanticConfidence::Conservative) ||
      selected->member_event_identities.size() != 2u || !selected->barrier_id ||
      selected->barrier_scope == ConSanBarrierSite::Scope::Unknown) {
    result.errors.emplace_back(
        "ConSan barrier ID/scope mutation requires an exact qualified two-member sequence");
    return;
  }
  const ConSanBarrierLifecycleGroup *lifecycle_group =
      find_lifecycle_group_for_sequence(result.program_inventory, *selected);
  if (lifecycle_group != nullptr && !lifecycle_group->admissible()) {
    result.errors.emplace_back(
        "ConSan barrier ID/scope mutation refused a partial unresolved lifecycle mutation: " +
        std::string(consan_barrier_lifecycle_issue_message(lifecycle_group->issue)));
    return;
  }
  const int32_t target_id = *plan.target_barrier_id;
  if (target_id == *selected->barrier_id ||
      barrier_scope_for_id(target_id) == ConSanBarrierSite::Scope::Unknown) {
    result.errors.emplace_back(
        "ConSan barrier ID/scope mutation requires a distinct target with known scope");
    return;
  }
  if (lifecycle_group != nullptr && (target_id < 1 || target_id > 16)) {
    result.errors.emplace_back(
        "ConSan barrier lifecycle retarget requires a named-barrier target ID in [1, 16]");
    return;
  }

  const std::vector<std::string> &member_identities = lifecycle_group != nullptr
                                                          ? lifecycle_group->member_event_identities
                                                          : selected->member_event_identities;
  std::vector<const ConSanSyncEvent *> members;
  members.reserve(member_identities.size());
  for (const std::string &identity : member_identities) {
    const ConSanSyncEvent *event = sync.find_event(identity);
    if (event == nullptr) {
      result.errors.emplace_back(
          "ConSan barrier ID/scope mutation could not resolve every logical member");
      return;
    }
    members.push_back(event);
  }
  if (lifecycle_group == nullptr &&
      (members.size() != 2u || members[0]->operation != ConSanSyncOperation::BarrierSignal ||
       members[1]->operation != ConSanSyncOperation::BarrierWait)) {
    result.errors.emplace_back(
        "ConSan barrier ID/scope mutation rejected an unsupported pair shape");
    return;
  }

  const std::optional<uint8_t> inline_selector = inline_barrier_selector_for_id(target_id);
  if (!inline_selector || target_id < std::numeric_limits<int16_t>::min() ||
      target_id > std::numeric_limits<int16_t>::max()) {
    result.errors.emplace_back(
        "ConSan barrier ID/scope mutation cannot preserve the selected operand encodings");
    return;
  }

  for (const ConSanSyncEvent *member : members) {
    if (member->operation == ConSanSyncOperation::BarrierLeave) {
      if (lifecycle_group == nullptr || member->size != sizeof(uint32_t) ||
          member->barrier_raw_simm16 != 0u) {
        result.errors.emplace_back(
            "ConSan barrier ID/scope mutation found an unsupported leave member");
        return;
      }
      continue;
    }
    const bool selector_member = member->operation == ConSanSyncOperation::BarrierInit ||
                                 member->operation == ConSanSyncOperation::BarrierJoin ||
                                 member->operation == ConSanSyncOperation::BarrierSignal;
    const bool inline_member =
        selector_member &&
        member->barrier_operand_source == ConSanBarrierSite::OperandSource::Immediate &&
        member->size == sizeof(uint32_t);
    const bool literal32_member =
        selector_member &&
        member->barrier_operand_source == ConSanBarrierSite::OperandSource::Literal32 &&
        member->size == 2u * sizeof(uint32_t);
    const bool wait_member =
        member->operation == ConSanSyncOperation::BarrierWait &&
        member->barrier_operand_source == ConSanBarrierSite::OperandSource::Immediate &&
        member->size == sizeof(uint32_t);
    if (!inline_member && !literal32_member && !wait_member) {
      result.errors.emplace_back(
          "ConSan barrier ID/scope mutation cannot preserve a logical member encoding");
      return;
    }
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  for (const ConSanSyncEvent *member : members) {
    if (member->file_offset > original_bytes.size() ||
        member->size > original_bytes.size() - member->file_offset) {
      result.errors.emplace_back("ConSan barrier ID/scope mutation exceeds ELF bytes");
      return;
    }
  }
  std::vector<uint8_t> replacement(original_bytes.begin(), original_bytes.end());
  for (const ConSanSyncEvent *member : members) {
    if (member->operation == ConSanSyncOperation::BarrierLeave)
      continue;
    if (member->operation == ConSanSyncOperation::BarrierWait) {
      uint32_t word = 0;
      std::memcpy(&word, replacement.data() + member->file_offset, sizeof(word));
      word = (word & 0xffff0000u) | static_cast<uint16_t>(target_id);
      std::memcpy(replacement.data() + member->file_offset, &word, sizeof(word));
      continue;
    }
    if (member->barrier_operand_source == ConSanBarrierSite::OperandSource::Immediate) {
      uint32_t word = 0;
      std::memcpy(&word, replacement.data() + member->file_offset, sizeof(word));
      word = (word & ~0xffu) | *inline_selector;
      std::memcpy(replacement.data() + member->file_offset, &word, sizeof(word));
    } else {
      const uint32_t literal = static_cast<uint32_t>(target_id);
      std::memcpy(replacement.data() + member->file_offset + sizeof(uint32_t), &literal,
                  sizeof(literal));
    }
  }

  result.replacement = std::move(replacement);
  for (const ConSanSyncEvent *member : members) {
    if (member->operation == ConSanSyncOperation::BarrierLeave)
      continue;
    ConSanPatchInfo patch;
    patch.phase = ConSanPatchPhase::Mutation;
    patch.kind = ConSanPatchKind::InlineBarrierIdScopeRewrite;
    patch.anchor_offset = member->text_offset;
    patch.trampoline_offset = member->text_offset;
    patch.original_size = member->size;
    result.patches.push_back(std::move(patch));
  }
  result.mark_modified();
  result.mutation.applied_fault_logical_identity =
      lifecycle_group != nullptr ? lifecycle_group->identity : selected->identity;
  result.warnings.emplace_back(
      std::string("ConSan barrier ID/scope mutation rewrote one logical ") +
      (lifecycle_group != nullptr ? "lifecycle" : "pair") + " from " +
      std::to_string(*selected->barrier_id) + " to " + std::to_string(target_id));
}

static void try_apply_barrier_participant_fault_patch(const AmdGpuCodeObject &code_object,
                                                      const ConSanFaultMutationPlan &plan,
                                                      FaultApplicationState &result) {
  const SynchronizationInventoryView sync = result.program_inventory.sync();
  const auto selected =
      plan.selection_sequence_identity
          ? std::ranges::find(sync.sync_sequences, *plan.selection_sequence_identity,
                              &ConSanSyncSequence::identity)
          : sync.sync_sequences.end();
  const ConSanBarrierLifecycleGroup *group =
      selected == sync.sync_sequences.end()
          ? nullptr
          : find_lifecycle_group_for_sequence(result.program_inventory, *selected);
  const ConSanBarrierSite *init =
      group != nullptr && group->admissible()
          ? find_static_participant_init(result.program_inventory, *group)
          : nullptr;
  if (group == nullptr || !group->admissible() || init == nullptr ||
      init->participant_encoding != ConSanBarrierSite::ParticipantEncoding::M0Literal32 ||
      !init->participant_count || !init->participant_setup_text_offset ||
      !init->participant_setup_file_offset || !init->participant_setup_literal) {
    mark_participant_mutation_unsupported(
        result, "no complete lifecycle has an immediately preceding literal M0 count setup");
    return;
  }
  if (!plan.target_participant_count || *plan.target_participant_count == 0u ||
      *plan.target_participant_count > 63u ||
      *plan.target_participant_count == *init->participant_count) {
    mark_participant_mutation_unsupported(
        result, "the target count must be a distinct value in the encodable range [1, 63]");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  const uint64_t literal_file_offset = *init->participant_setup_file_offset + sizeof(uint32_t);
  if (literal_file_offset > original_bytes.size() ||
      sizeof(uint32_t) > original_bytes.size() - literal_file_offset) {
    result.errors.emplace_back("ConSan barrier participant count setup exceeds ELF bytes");
    return;
  }
  constexpr uint32_t kParticipantCountMask = 0x3fu << 16u;
  uint32_t replacement_literal = *init->participant_setup_literal;
  replacement_literal =
      (replacement_literal & ~kParticipantCountMask) | (*plan.target_participant_count << 16u);
  result.replacement.assign(original_bytes.begin(), original_bytes.end());
  std::memcpy(result.replacement.data() + literal_file_offset, &replacement_literal,
              sizeof(replacement_literal));

  ConSanPatchInfo patch;
  patch.phase = ConSanPatchPhase::Mutation;
  patch.kind = ConSanPatchKind::InlineBarrierParticipantCountRewrite;
  patch.anchor_offset = *init->participant_setup_text_offset;
  patch.trampoline_offset = patch.anchor_offset;
  patch.original_size = 2u * sizeof(uint32_t);
  patch.original_participant_count = init->participant_count;
  patch.target_participant_count = plan.target_participant_count;
  result.patches.push_back(std::move(patch));
  result.mark_modified();
  result.mutation.applied_fault_logical_identity = group->identity;
  result.warnings.emplace_back("ConSan barrier participant mutation rewrote one proven static "
                               "lifecycle count from " +
                               std::to_string(*init->participant_count) + " to " +
                               std::to_string(*plan.target_participant_count));
}

static void try_apply_barrier_move_fault_patch(const AmdGpuCodeObject &code_object,
                                               rj_code_arch_t arch,
                                               const ConSanFaultMutationPlan &plan,
                                               const ConSanPatchedImageGrowthLimit &growth_limit,
                                               FaultApplicationState &result) {
  if (has_only_rocclr_runtime_kernels(result.program_inventory)) {
    result.warnings.emplace_back(
        "ConSan barrier move fault skipped ROCclr runtime helper code object");
    return;
  }
  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  std::optional<BarrierSite> site;
  const ConSanFaultSelectionView selection = fault_selection_view(result);
  const ConSanFaultSite *selected_inventory_site =
      find_fault_site_by_identity(selection, plan.primary_identity, ConSanFaultSiteKind::Barrier);
  if (selected_inventory_site != nullptr)
    site = barrier_site_from_inventory(*selected_inventory_site);
  if (!site) {
    result.errors.emplace_back(
        "ConSan barrier move fault plan no longer matches its pristine primary site");
    return;
  }
  const ConSanFaultSite *companion_inventory_site =
      plan.companion_identity ? find_fault_site_by_identity(selection, *plan.companion_identity,
                                                            ConSanFaultSiteKind::Barrier)
                              : nullptr;
  const SynchronizationInventoryView sync = result.program_inventory.sync();
  const auto sequence =
      plan.selection_sequence_identity
          ? std::ranges::find(sync.sync_sequences, *plan.selection_sequence_identity,
                              &ConSanSyncSequence::identity)
          : sync.sync_sequences.end();
  const ConSanSyncSequence *selected_sequence =
      sequence == sync.sync_sequences.end() ? nullptr : &*sequence;
  const bool pair_matches_plan =
      companion_inventory_site != nullptr && selected_inventory_site->sync_event_identity &&
      companion_inventory_site->sync_event_identity && selected_sequence != nullptr &&
      selected_sequence->operation == ConSanSyncOperation::BarrierFull &&
      consan_sync_confidence_meets(selected_sequence->confidence,
                                   ConSanSemanticConfidence::Conservative) &&
      selected_sequence->member_event_identities == plan.ordered_member_identities &&
      std::ranges::find(selected_sequence->member_event_identities,
                        *selected_inventory_site->sync_event_identity) !=
          selected_sequence->member_event_identities.end() &&
      std::ranges::find(selected_sequence->member_event_identities,
                        *companion_inventory_site->sync_event_identity) !=
          selected_sequence->member_event_identities.end();
  if (!pair_matches_plan) {
    result.errors.emplace_back(
        "ConSan barrier move fault plan no longer matches its pristine logical pair");
    return;
  }
  const BarrierSite second_site = barrier_site_from_inventory(*companion_inventory_site);

  if (plan.barrier_move_direction != ConSanBarrierMoveDirection::LegacyMarker) {
    if (!plan.destination_identity) {
      result.warnings.emplace_back(
          "ConSan markerless barrier move requires an exact destination identity");
      return;
    }
    const auto destination =
        std::ranges::find(result.barrier_move_destinations, *plan.destination_identity,
                          &ConSanBarrierMoveDestination::identity);
    if (destination == result.barrier_move_destinations.end()) {
      result.warnings.emplace_back(
          "ConSan barrier move fault found no destination with exact identity '" +
          *plan.destination_identity + "'");
      return;
    }
    if (!selected_sequence->basic_block_index ||
        destination->container_name != selected_sequence->container_name ||
        destination->in_kernel != selected_sequence->in_kernel) {
      result.warnings.emplace_back(
          "ConSan barrier move fault rejected destination across a control-flow or container "
          "boundary");
      return;
    }
    const bool cross_block =
        destination->basic_block_index != *selected_sequence->basic_block_index;
    const bool destination_matches_cross_block_plan =
        plan.barrier_move_direction == ConSanBarrierMoveDirection::Earlier &&
        destination->cfg_contract == plan.barrier_move_cfg_contract &&
        destination->structured_guard_block_index == plan.structured_guard_block_index &&
        destination->basic_block_index == plan.structured_destination_block_index &&
        destination->structured_source_block_index == plan.structured_source_block_index &&
        destination->structured_guard_offset == plan.structured_guard_offset &&
        destination->text_offset == plan.structured_destination_offset &&
        destination->structured_source_offset == plan.structured_source_offset;
    if (cross_block && !destination_matches_cross_block_plan) {
      result.warnings.emplace_back(
          "ConSan barrier move fault rejected control-flow or container boundary: cross-block "
          "destination lacks a matching proven completing or destructive structured diamond "
          "opt-in");
      return;
    }
    if (destination->text_offset < selected_sequence->end_text_offset &&
        selected_sequence->begin_text_offset < destination->text_offset + destination->size) {
      result.warnings.emplace_back(
          "ConSan barrier move fault rejected destination overlapping the selected logical "
          "barrier");
      return;
    }
    if (!destination->suitable()) {
      result.warnings.emplace_back("ConSan barrier move fault rejected unsuitable destination: " +
                                   consan_barrier_move_destination_issue_message(
                                       destination->issue, destination->issue_detail));
      return;
    }
    if (plan.barrier_move_direction == ConSanBarrierMoveDirection::Earlier &&
        destination->text_offset + destination->size > selected_sequence->begin_text_offset) {
      result.warnings.emplace_back(
          "ConSan barrier move fault rejected a destination that is not earlier than the "
          "selected logical barrier");
      return;
    }
    if (plan.barrier_move_direction == ConSanBarrierMoveDirection::Later &&
        destination->text_offset < selected_sequence->end_text_offset) {
      result.warnings.emplace_back(
          "ConSan barrier move fault rejected a destination that is not later than the selected "
          "logical barrier");
      return;
    }

    std::array<BarrierSite, 2> ordered_sources = {*site, second_site};
    std::ranges::sort(ordered_sources, {}, &BarrierSite::text_offset);
    if (ordered_sources[0].size != sizeof(uint32_t) ||
        ordered_sources[1].size != sizeof(uint32_t) ||
        ordered_sources[0].text_offset + sizeof(uint32_t) != ordered_sources[1].text_offset) {
      result.errors.emplace_back(
          "ConSan markerless barrier move requires an adjacent two-instruction barrier pair");
      return;
    }

    CodeObjectPatcher patcher(code_object);
    const std::span<const uint8_t> text = patcher.text_bytes();
    if (destination->text_offset > text.size() ||
        destination->size > text.size() - destination->text_offset) {
      result.errors.emplace_back("ConSan markerless barrier move destination exceeds .text");
      return;
    }
    std::array<uint32_t, 2> barrier_words{};
    for (size_t i = 0; i < ordered_sources.size(); ++i) {
      std::memcpy(&barrier_words[i], text.data() + ordered_sources[i].text_offset,
                  sizeof(uint32_t));
    }
    std::vector<uint32_t> destination_words(destination->size / sizeof(uint32_t));
    std::memcpy(destination_words.data(), text.data() + destination->text_offset,
                destination->size);

    const uint64_t trampoline_size = 2u * sizeof(uint32_t) + destination->size + sizeof(uint32_t);
    std::optional<DbiPatchLocalCave> local_cave;
    for (const LocalNopCave &candidate :
         find_uncovered_nop_caves(code_object, result.program_inventory, arch)) {
      if (static_cast<uint64_t>(candidate.word_count) * sizeof(uint32_t) < trampoline_size)
        continue;
      const uint64_t return_pc = candidate.text_offset + trampoline_size - sizeof(uint32_t);
      if (compute_sopp_branch_simm16(destination->text_offset, candidate.text_offset) &&
          compute_sopp_branch_simm16(return_pc, destination->text_offset + destination->size)) {
        local_cave = DbiPatchLocalCave{
            candidate.text_offset, static_cast<uint64_t>(candidate.word_count) * sizeof(uint32_t)};
        break;
      }
    }
    DbiPatchPlacementPlanner placement_planner(arch, text.size());
    DbiPatchPlacementRequest placement_request;
    placement_request.anchor_offset = destination->text_offset;
    placement_request.original_size = destination->size;
    placement_request.body_size = 2u * sizeof(uint32_t) + destination->size;
    placement_request.local_cave = local_cave;
    placement_request.allow_appended_cave = true;
    std::string placement_error;
    const auto placement = placement_planner.plan(placement_request, &placement_error);
    if (!placement) {
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          "ConSan markerless barrier move has no reachable local or appended cave: " +
          placement_error);
      return;
    }
    const auto forward =
        compute_sopp_branch_simm16(placement->anchor_offset, placement->body_offset);
    const auto ret =
        compute_sopp_branch_simm16(placement->return_branch_offset, placement->return_target);
    if (!forward || !ret) {
      result.errors.emplace_back(
          "ConSan markerless barrier move trampoline branch exceeds s_branch simm16");
      return;
    }

    std::vector<uint32_t> cave_words;
    cave_words.reserve(2u + destination_words.size() + 1u);
    if (plan.barrier_move_direction == ConSanBarrierMoveDirection::Earlier) {
      cave_words.insert(cave_words.end(), barrier_words.begin(), barrier_words.end());
      cave_words.insert(cave_words.end(), destination_words.begin(), destination_words.end());
    } else {
      cave_words.insert(cave_words.end(), destination_words.begin(), destination_words.end());
      cave_words.insert(cave_words.end(), barrier_words.begin(), barrier_words.end());
    }
    cave_words.push_back(build_s_branch(*ret, arch));

    std::vector<uint8_t> new_text(text.begin(), text.end());
    const uint32_t nop = build_s_nop(0, arch);
    for (const BarrierSite &source : ordered_sources)
      std::memcpy(new_text.data() + source.text_offset, &nop, sizeof(nop));
    const uint32_t branch = build_s_branch(*forward, arch);
    std::memcpy(new_text.data() + destination->text_offset, &branch, sizeof(branch));
    for (uint32_t offset = sizeof(uint32_t); offset < destination->size; offset += sizeof(uint32_t))
      std::memcpy(new_text.data() + destination->text_offset + offset, &nop, sizeof(nop));
    if (placement->kind == DbiPatchPlacementKind::AppendedCave) {
      if (new_text.size() != placement->body_offset) {
        result.errors.emplace_back(
            "ConSan markerless barrier move emitted a stale appended-cave mapping");
        return;
      }
      append_consan_patch_words(new_text, cave_words);
    } else {
      std::memcpy(new_text.data() + placement->body_offset, cave_words.data(),
                  cave_words.size() * sizeof(uint32_t));
    }
    if (!replace_consan_text(patcher, new_text, growth_limit, "markerless barrier move",
                             result.program_inventory.code_object_id(), result.errors,
                             &result.transform_failure_cause)) {
      return;
    }
    result.replacement = std::move(patcher).emit();

    for (const BarrierSite &source : ordered_sources) {
      ConSanPatchInfo source_info;
      source_info.phase = ConSanPatchPhase::Mutation;
      source_info.kind = ConSanPatchKind::InlineBarrierMoveSourceRewrite;
      source_info.anchor_offset = source.text_offset;
      source_info.trampoline_offset = source.text_offset;
      source_info.original_size = source.size;
      source_info.barrier_move_direction = plan.barrier_move_direction;
      result.patches.push_back(source_info);
    }
    ConSanPatchInfo target_info;
    target_info.phase = ConSanPatchPhase::Mutation;
    target_info.kind = ConSanPatchKind::InlineBarrierMoveTargetRewrite;
    target_info.anchor_offset = destination->text_offset;
    target_info.trampoline_offset = placement->body_offset;
    target_info.original_size = destination->size;
    target_info.trampoline_size = static_cast<uint32_t>(cave_words.size() * sizeof(uint32_t));
    target_info.barrier_move_direction = plan.barrier_move_direction;
    if (cross_block) {
      target_info.barrier_move_cfg_contract = destination->cfg_contract;
      target_info.structured_guard_block_index = destination->structured_guard_block_index;
      target_info.structured_destination_block_index = destination->basic_block_index;
      target_info.structured_source_block_index = destination->structured_source_block_index;
      target_info.structured_guard_offset = destination->structured_guard_offset;
      target_info.structured_destination_offset = destination->text_offset;
      target_info.structured_source_offset = destination->structured_source_offset;
    }
    result.patches.push_back(target_info);
    result.mark_modified();
    result.warnings.emplace_back(
        "ConSan barrier move fault relocated an adjacent logical pair " +
        std::string(plan.barrier_move_direction == ConSanBarrierMoveDirection::Earlier ? "earlier"
                                                                                       : "later") +
        " at " + destination->mnemonic + " in " + destination->container_name);
    return;
  }

  constexpr uint16_t marker_imm = 42;
  const uint32_t marker = build_s_nop(marker_imm, arch);
  const uint32_t second_marker = build_s_nop(marker_imm + 1u, arch);
  std::optional<uint64_t> marker_file_offset;
  std::optional<uint64_t> marker_text_offset;
  std::optional<uint64_t> second_marker_file_offset;
  std::optional<uint64_t> second_marker_text_offset;
  for (const ConSanTextSection &section : result.program_inventory.text_sections()) {
    for (uint64_t offset = 0; offset + sizeof(uint32_t) <= section.size;
         offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, original_bytes.data() + section.file_offset + offset, sizeof(word));
      if (word != marker)
        if (word != second_marker)
          continue;
      if (word == marker) {
        if (marker_file_offset) {
          result.errors.emplace_back("ConSan barrier move fault marker is not unique");
          return;
        }
        marker_file_offset = section.file_offset + offset;
        marker_text_offset = offset;
      } else {
        if (second_marker_file_offset) {
          result.errors.emplace_back("ConSan barrier move fault second marker is not unique");
          return;
        }
        second_marker_file_offset = section.file_offset + offset;
        second_marker_text_offset = offset;
      }
    }
  }
  if (!marker_file_offset || !second_marker_file_offset) {
    result.warnings.emplace_back(
        "ConSan barrier move fault found no s_nop marker pair starting at " +
        std::to_string(marker_imm));
    return;
  }
  if (*marker_file_offset <= second_site.file_offset ||
      *second_marker_file_offset != *marker_file_offset + sizeof(uint32_t)) {
    result.errors.emplace_back("ConSan barrier move fault marker must follow the selected barrier");
    return;
  }

  if (result.replacement.empty())
    result.replacement.assign(original_bytes.begin(), original_bytes.end());
  uint32_t barrier_word = 0;
  uint32_t second_barrier_word = 0;
  std::memcpy(&barrier_word, original_bytes.data() + site->file_offset, sizeof(barrier_word));
  std::memcpy(&second_barrier_word, original_bytes.data() + second_site.file_offset,
              sizeof(second_barrier_word));
  const uint32_t nop = build_s_nop(0, arch);
  std::memcpy(result.replacement.data() + site->file_offset, &nop, sizeof(nop));
  std::memcpy(result.replacement.data() + second_site.file_offset, &nop, sizeof(nop));
  std::memcpy(result.replacement.data() + *marker_file_offset, &barrier_word, sizeof(barrier_word));
  std::memcpy(result.replacement.data() + *second_marker_file_offset, &second_barrier_word,
              sizeof(second_barrier_word));

  ConSanPatchInfo source_info;
  source_info.phase = ConSanPatchPhase::Mutation;
  source_info.kind = ConSanPatchKind::InlineBarrierMoveSourceRewrite;
  source_info.anchor_offset = site->text_offset;
  source_info.trampoline_offset = site->text_offset;
  source_info.original_size = site->size;
  result.patches.push_back(source_info);
  source_info.anchor_offset = second_site.text_offset;
  source_info.trampoline_offset = second_site.text_offset;
  source_info.original_size = second_site.size;
  result.patches.push_back(source_info);
  ConSanPatchInfo target_info;
  target_info.phase = ConSanPatchPhase::Mutation;
  target_info.kind = ConSanPatchKind::InlineBarrierMoveTargetRewrite;
  target_info.anchor_offset = *marker_text_offset;
  target_info.trampoline_offset = *marker_text_offset;
  target_info.original_size = sizeof(uint32_t);
  result.patches.push_back(target_info);
  target_info.anchor_offset = *second_marker_text_offset;
  target_info.trampoline_offset = *second_marker_text_offset;
  result.patches.push_back(target_info);
  result.mark_modified();
  result.warnings.emplace_back("ConSan barrier move fault moved " + site->mnemonic + " in " +
                               site->range_name);
}

static void try_apply_atomic_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                         const ConSanFaultMutationPlan *address_plan,
                                         const ConSanFaultMutationPlan *order_plan,
                                         const ConSanFaultMutationPlan *scope_plan,
                                         FaultApplicationState &result) {
  const ConSanFaultMutationPlan *primary_plan =
      address_plan != nullptr ? address_plan : (order_plan != nullptr ? order_plan : scope_plan);
  const ConSanFaultSite *selected =
      primary_plan != nullptr
          ? find_fault_site_by_identity(fault_selection_view(result),
                                        primary_plan->primary_identity, ConSanFaultSiteKind::Atomic)
          : nullptr;
  if (!selected) {
    result.errors.emplace_back("ConSan atomic fault plan no longer matches its pristine site");
    return;
  }

  const AtomicFaultEncoding encoding =
      classify_consan_atomic_fault_encoding(selected->mnemonic, selected->size, arch);
  const std::string_view unsupported_fault =
      scope_plan != nullptr && !consan_atomic_fault_supports_scope(encoding)       ? "scope"
      : order_plan != nullptr && !consan_atomic_fault_supports_order(encoding)     ? "order"
      : address_plan != nullptr && !consan_atomic_fault_supports_address(encoding) ? "address"
                                                                                   : "";
  if (!unsupported_fault.empty()) {
    result.errors.emplace_back("ConSan atomic " + std::string(unsupported_fault) +
                               " fault is unsupported for the selected atomic encoding");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  if (result.replacement.empty())
    result.replacement.assign(original_bytes.begin(), original_bytes.end());
  std::array<uint32_t, 3> words{};
  std::memcpy(words.data(), result.replacement.data() + selected->file_offset, selected->size);
  std::span<uint8_t> atomic_instruction(reinterpret_cast<uint8_t *>(words.data()), selected->size);
  if (address_plan != nullptr) {
    const uint32_t address_delta = address_plan->address_delta_bytes.value_or(0u);
    if (address_delta == 0 || address_delta % sizeof(uint32_t) != 0 || address_delta > 0x7fffffu) {
      result.errors.emplace_back(
          "ConSan atomic address delta must be a positive aligned signed-24-bit offset");
      return;
    }
    const ConSanAtomicFaultRewriteResult rewrite = rewrite_consan_atomic_fault_address(
        atomic_instruction, encoding, selected->width_bits, address_delta);
    if (!rewrite.rewritten()) {
      switch (rewrite.status) {
      case ConSanAtomicFaultRewriteStatus::OffsetOverflow:
        result.errors.emplace_back(
            encoding == AtomicFaultEncoding::Ds
                ? "ConSan DS atomic address delta overflows 8-bit offset0"
                : "ConSan atomic address delta overflows signed-24-bit ioffset");
        break;
      case ConSanAtomicFaultRewriteStatus::MisalignedOffset:
        result.errors.emplace_back("ConSan DS atomic replacement offset is not naturally aligned");
        break;
      case ConSanAtomicFaultRewriteStatus::InvalidEncoding:
      case ConSanAtomicFaultRewriteStatus::AlreadyWaveScope:
      case ConSanAtomicFaultRewriteStatus::Rewritten:
        result.errors.emplace_back("ConSan atomic address fault could not rewrite target encoding");
        break;
      }
      return;
    }
    ConSanPatchInfo info;
    info.phase = ConSanPatchPhase::Mutation;
    info.kind = ConSanPatchKind::InlineAtomicAddressRewrite;
    info.anchor_offset = selected->text_offset;
    info.trampoline_offset = selected->text_offset;
    info.original_size = selected->size;
    result.patches.push_back(info);
  }
  if (order_plan != nullptr) {
    const ConSanAtomicOrderEdge order_edge =
        order_plan->atomic_order_edge.value_or(ConSanAtomicOrderEdge::Any);
    AtomicOrderBoundary boundary =
        resolve_atomic_order_boundary(result.program_inventory, *selected, encoding, order_edge);
    const ConSanSyncSequence *sequence = boundary.sequence;
    const ConSanSyncEvent *fence = boundary.fence;
    std::optional<uint64_t> &release_wait_text_offset = boundary.release_wait_text_offset;
    // A cache-associated release sequence may not record the immediately
    // preceding store wait as its primary proof, but that wait becomes an
    // independently sufficient release proof once the cache operation is
    // removed. Discover it directly so the staged reinventory observes the
    // complete weakened boundary rather than re-admitting the same atomic.
    if (encoding != AtomicFaultEncoding::CdnaFlat && !release_wait_text_offset &&
        order_edge != ConSanAtomicOrderEdge::Acquire && selected->text_offset >= sizeof(uint32_t) &&
        selected->file_offset >= sizeof(uint32_t)) {
      uint32_t preceding_word = 0;
      std::memcpy(&preceding_word,
                  result.replacement.data() + selected->file_offset - sizeof(uint32_t),
                  sizeof(preceding_word));
      if (classify_consan_wait_instruction({}, preceding_word, arch).release_boundary)
        release_wait_text_offset = selected->text_offset - sizeof(uint32_t);
    }
    if (fence == nullptr && !release_wait_text_offset) {
      result.errors.emplace_back(
          "ConSan atomic weaken-order mutation found no qualified ordering boundary");
      return;
    }
    const uint32_t nop = build_s_nop(0, arch);
    ConSanPatchInfo info;
    info.phase = ConSanPatchPhase::Mutation;
    info.kind = ConSanPatchKind::InlineAtomicOrderRewrite;
    info.fault_primary_identity = selected->identity;
    info.fault_sequence_identity = sequence != nullptr ? sequence->identity : selected->identity;
    std::string removed_boundary;
    if (fence != nullptr) {
      if (fence->size == 0 || fence->size % sizeof(uint32_t) != 0) {
        result.errors.emplace_back(
            "ConSan atomic weaken-order mutation found malformed associated cache operation");
        return;
      }
      for (uint32_t offset = 0; offset < fence->size; offset += sizeof(uint32_t))
        std::memcpy(result.replacement.data() + fence->file_offset + offset, &nop, sizeof(nop));
      info.anchor_offset = fence->text_offset;
      info.trampoline_offset = fence->text_offset;
      info.original_size = fence->size;
      result.patches.push_back(info);
      removed_boundary = fence->mnemonic;
    }
    // RDNA4 release lowering commonly pairs a cache operation with an exact
    // store wait. Once either is selected, remove the complete independently
    // sufficient ordering boundary so staged reinventory cannot re-admit it.
    if (release_wait_text_offset) {
      const uint64_t wait_text_offset = *release_wait_text_offset;
      if (wait_text_offset >= selected->text_offset ||
          selected->file_offset < selected->text_offset - wait_text_offset) {
        result.errors.emplace_back(
            "ConSan atomic weaken-order mutation found malformed associated release wait");
        return;
      }
      const uint64_t wait_file_offset =
          selected->file_offset - (selected->text_offset - wait_text_offset);
      if (wait_file_offset > result.replacement.size() ||
          sizeof(uint32_t) > result.replacement.size() - wait_file_offset) {
        result.errors.emplace_back(
            "ConSan atomic weaken-order mutation release wait exceeds the code object");
        return;
      }
      uint32_t wait_word = 0;
      std::memcpy(&wait_word, result.replacement.data() + wait_file_offset, sizeof(wait_word));
      const ConSanWaitInstructionEncoding wait =
          classify_consan_wait_instruction({}, wait_word, arch);
      if (!wait.release_boundary) {
        result.errors.emplace_back(
            "ConSan atomic weaken-order release wait no longer has its admitted encoding");
        return;
      }
      std::memcpy(result.replacement.data() + wait_file_offset, &nop, sizeof(nop));
      ConSanPatchInfo wait_info = info;
      wait_info.anchor_offset = wait_text_offset;
      wait_info.trampoline_offset = wait_text_offset;
      wait_info.original_size = sizeof(uint32_t);
      result.patches.push_back(wait_info);
      if (fence != nullptr)
        removed_boundary += "/";
      removed_boundary += wait.drains_lds ? "s_wait_storecnt_dscnt" : "s_wait_storecnt";
    }
    result.warnings.emplace_back("ConSan atomic fault removed associated " + removed_boundary +
                                 " while preserving " + selected->mnemonic);
  }
  if (scope_plan != nullptr) {
    const ConSanAtomicFaultRewriteResult rewrite =
        rewrite_consan_atomic_fault_scope_to_wave(atomic_instruction, encoding);
    if (!rewrite.rewritten()) {
      if (rewrite.status == ConSanAtomicFaultRewriteStatus::AlreadyWaveScope) {
        result.errors.emplace_back("ConSan atomic scope fault cannot weaken wave scope further");
      } else {
        result.errors.emplace_back("ConSan atomic scope fault could not rewrite target encoding");
      }
      return;
    }
    ConSanPatchInfo info;
    info.phase = ConSanPatchPhase::Mutation;
    info.kind = ConSanPatchKind::InlineAtomicScopeRewrite;
    info.anchor_offset = selected->text_offset;
    info.trampoline_offset = selected->text_offset;
    info.original_size = selected->size;
    result.patches.push_back(info);
    result.warnings.emplace_back("ConSan atomic fault lowered " + selected->mnemonic +
                                 " scope from " + std::to_string(rewrite.previous_value) +
                                 " to wave");
  }
  std::memcpy(result.replacement.data() + selected->file_offset, words.data(), selected->size);
  result.mark_modified();
  if (address_plan != nullptr) {
    result.warnings.emplace_back("ConSan atomic fault rewrote " + selected->mnemonic +
                                 " with validated address delta " +
                                 std::to_string(*address_plan->address_delta_bytes));
  }
}

static void try_apply_lds_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                      const ConSanFaultMutationPlan &plan,
                                      FaultApplicationState &result) {
  if (!consan_lds_address_fault_arch_supported(arch)) {
    result.errors.emplace_back("ConSan LDS address fault is unsupported for the selected target");
    return;
  }
  const ConSanFaultSite *selected = find_fault_site_by_identity(
      fault_selection_view(result), plan.primary_identity, ConSanFaultSiteKind::LdsAccess);
  if (selected == nullptr || selected->address_vgpr != plan.original_address_vgpr) {
    result.errors.emplace_back(
        "ConSan LDS fault plan no longer matches its pristine address operand");
    return;
  }
  if (!selected->address_vgpr || selected->size != 2u * sizeof(uint32_t)) {
    result.errors.emplace_back(
        "ConSan LDS address fault selected an unsupported decoded instruction shape");
    return;
  }
  if (!plan.target_address_vgpr) {
    result.errors.emplace_back("ConSan LDS address fault requires an explicit replacement VGPR");
    return;
  }
  const uint16_t target_address_vgpr = *plan.target_address_vgpr;
  if (target_address_vgpr == *selected->address_vgpr) {
    result.errors.emplace_back(
        "ConSan LDS address fault requires a distinct 8-bit replacement VGPR");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  if (selected->execution_owners.empty()) {
    result.errors.emplace_back(
        "ConSan LDS address fault selected an access without an execution owner");
    return;
  }
  const uint32_t address_bank =
      consan_arch_has_selectable_vgpr_bank(arch)
          ? static_cast<uint32_t>(selected->selectable_vgpr_bank_mode.value_or(0u) & 0x3u)
          : 0u;
  const uint32_t physical_target_address_vgpr = address_bank * 256u + target_address_vgpr;
  for (const ConSanExecutionOwner &owner : selected->execution_owners) {
    const auto allocated =
        read_descriptor_vgpr_allocation_count(original_bytes, owner.descriptor_file_offset, arch);
    if (!allocated) {
      result.errors.emplace_back(
          "ConSan LDS address fault could not read an execution-owner descriptor");
      return;
    }
    if (physical_target_address_vgpr >= *allocated) {
      result.errors.emplace_back("ConSan LDS address fault replacement v" +
                                 std::to_string(physical_target_address_vgpr) +
                                 " is outside an execution owner's allocated VGPR window");
      return;
    }
  }
  if (selected->file_offset > original_bytes.size() ||
      selected->size > original_bytes.size() - selected->file_offset) {
    result.errors.emplace_back("ConSan LDS address fault selected an invalid instruction range");
    return;
  }
  if (result.replacement.empty())
    result.replacement.assign(original_bytes.begin(), original_bytes.end());
  std::array<uint32_t, 2> words{};
  std::memcpy(words.data(), result.replacement.data() + selected->file_offset, selected->size);
  // Every supported target encodes the DS address VGPR in the low byte of
  // word one. Cross-check the decoder before rewriting that operand alone.
  if ((words[1] & 0xffu) != *selected->address_vgpr) {
    result.errors.emplace_back("ConSan LDS address fault found a stale decoded address operand");
    return;
  }
  words[1] = (words[1] & ~0xffu) | target_address_vgpr;
  std::memcpy(result.replacement.data() + selected->file_offset, words.data(), selected->size);

  ConSanPatchInfo info;
  info.phase = ConSanPatchPhase::Mutation;
  info.kind = ConSanPatchKind::InlineLdsAddressRewrite;
  info.anchor_offset = selected->text_offset;
  info.trampoline_offset = selected->text_offset;
  info.original_size = selected->size;
  info.fault_primary_identity = selected->identity;
  info.fault_original_address_vgpr = selected->address_vgpr;
  info.fault_target_address_vgpr = target_address_vgpr;
  for (const ConSanExecutionOwner &owner : selected->execution_owners)
    info.owner_descriptor_file_offsets.push_back(owner.descriptor_file_offset);
  result.patches.push_back(std::move(info));
  result.mutation.applied_fault_logical_identity = selected->identity;
  result.mark_modified();
  result.warnings.emplace_back("ConSan LDS fault rewrote " + selected->mnemonic +
                               " address from v" + std::to_string(*selected->address_vgpr) +
                               " to v" + std::to_string(target_address_vgpr));
}

static void try_apply_ordinary_fault_patch(const AmdGpuCodeObject &code_object,
                                           const ConSanFaultMutationPlan *address_plan,
                                           const ConSanFaultMutationPlan *order_plan,
                                           const ConSanFaultMutationPlan *scope_plan,
                                           FaultApplicationState &result) {
  const ConSanFaultMutationPlan *primary_plan =
      address_plan != nullptr ? address_plan : (order_plan != nullptr ? order_plan : scope_plan);
  const auto target = primary_plan != nullptr
                          ? select_ordinary_acquire_mutation_target(
                                fault_selection_view(result),
                                ConSanFaultSelection{
                                    .primary_site_identity = primary_plan->primary_identity,
                                    .primary_sequence_identity = {},
                                    .companion_site_identity = {},
                                    .companion_sequence_identity = {},
                                    .kernel_name_filter = {},
                                    .ordinal = 0,
                                })
                          : std::nullopt;
  if (!target) {
    result.errors.emplace_back("ConSan ordinary fault plan no longer matches its pristine site");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  std::vector<uint8_t> candidate =
      result.replacement.empty()
          ? std::vector<uint8_t>(original_bytes.begin(), original_bytes.end())
          : result.replacement;
  if (target->site->file_offset > candidate.size() ||
      target->site->size > candidate.size() - target->site->file_offset ||
      target->cache->file_offset > candidate.size() ||
      target->cache->size > candidate.size() - target->cache->file_offset) {
    result.errors.emplace_back("ConSan ordinary fault found an invalid exact mutation range");
    return;
  }
  std::span<uint8_t> target_instruction(candidate.data() + target->site->file_offset,
                                        target->site->size);
  const auto original_encoding = decode_consan_ordinary_global_fault_encoding(target_instruction);
  if (!original_encoding) {
    result.errors.emplace_back("ConSan ordinary fault found an unsupported target encoding");
    return;
  }

  std::vector<ConSanPatchInfo> patches;
  const auto make_info = [&](ConSanPatchKind kind, uint64_t anchor, uint32_t size,
                             std::string companion) {
    ConSanPatchInfo info;
    info.phase = ConSanPatchPhase::Mutation;
    info.kind = kind;
    info.anchor_offset = anchor;
    info.trampoline_offset = anchor;
    info.original_size = size;
    info.fault_primary_identity = target->site->identity;
    info.fault_companion_identity = std::move(companion);
    info.fault_sequence_identity = target->sequence->identity;
    for (const ConSanExecutionOwner &owner : target->site->execution_owners)
      info.owner_descriptor_file_offsets.push_back(owner.descriptor_file_offset);
    return info;
  };

  if (address_plan != nullptr) {
    const uint32_t address_delta = address_plan->address_delta_bytes.value_or(0u);
    if (address_delta == 0u || address_delta % sizeof(uint32_t) != 0u ||
        address_delta > 0x7fffffu) {
      result.errors.emplace_back(
          "ConSan ordinary address delta must be a positive aligned signed-24-bit offset");
      return;
    }
    const int32_t original_offset = original_encoding->byte_offset;
    if (!target->load->static_byte_offset || *target->load->static_byte_offset != original_offset ||
        target->load->address_source != ConSanSyncAddressSource::GlobalScalarVector ||
        target->site->mnemonic != "global_load_b32" || target->site->width_bits != 32u) {
      result.errors.emplace_back(
          "ConSan ordinary address fault found stale or unsupported decoded load fields");
      return;
    }
    const int64_t replacement_offset = static_cast<int64_t>(original_offset) + address_delta;
    if (replacement_offset < -0x800000ll || replacement_offset > 0x7fffffll ||
        replacement_offset % static_cast<int64_t>(sizeof(uint32_t)) != 0) {
      result.errors.emplace_back(
          "ConSan ordinary address delta produces an invalid signed-24-bit aligned ioffset");
      return;
    }
    if (!rewrite_consan_ordinary_global_fault_offset(target_instruction,
                                                     static_cast<int32_t>(replacement_offset))) {
      result.errors.emplace_back("ConSan ordinary address fault could not encode replacement");
      return;
    }
    patches.push_back(make_info(ConSanPatchKind::InlineOrdinaryAddressRewrite,
                                target->site->text_offset, target->site->size, {}));
  }

  if (order_plan != nullptr) {
    const uint32_t nop = build_s_nop(0, consan_arch_for_target(code_object.target_id()));
    for (uint32_t offset = 0; offset < target->cache->size; offset += sizeof(uint32_t))
      std::memcpy(candidate.data() + target->cache->file_offset + offset, &nop, sizeof(nop));
    patches.push_back(make_info(ConSanPatchKind::InlineOrdinaryOrderRewrite,
                                target->cache->text_offset, target->cache->size,
                                target->cache->identity));
  }
  if (scope_plan != nullptr) {
    const auto current_encoding = decode_consan_ordinary_global_fault_encoding(target_instruction);
    if (!current_encoding || !target->load->raw_scope || !target->load->scope ||
        current_encoding->scope != *target->load->raw_scope ||
        !consan_memory_scope_is_agent_or_system(*target->load->scope)) {
      result.errors.emplace_back(
          "ConSan ordinary scope fault found stale or already-wave decoded scope bits");
      return;
    }
    if (!rewrite_consan_ordinary_global_fault_scope(target_instruction, 0u)) {
      result.errors.emplace_back("ConSan ordinary scope fault could not encode replacement");
      return;
    }
    patches.push_back(make_info(ConSanPatchKind::InlineOrdinaryScopeRewrite,
                                target->site->text_offset, target->site->size, {}));
  }

  result.replacement = std::move(candidate);
  result.patches.insert(result.patches.end(), std::make_move_iterator(patches.begin()),
                        std::make_move_iterator(patches.end()));
  result.mutation.applied_fault_logical_identity = target->sequence->identity;
  result.mark_modified();
  if (address_plan != nullptr) {
    result.warnings.emplace_back("ConSan ordinary fault shifted global_load_b32 by validated " +
                                 std::to_string(*address_plan->address_delta_bytes) + " byte(s)");
  }
  if (order_plan != nullptr) {
    result.warnings.emplace_back(
        "ConSan ordinary fault removed associated global_inv while preserving the acquire load "
        "and waits");
  }
  if (scope_plan != nullptr) {
    result.warnings.emplace_back("ConSan ordinary fault lowered global_load_b32 scope to wave");
  }
}

namespace {

[[nodiscard]] const ConSanFaultMutationPlan *
find_fault_plan(std::span<const ConSanFaultMutationPlan> plans, ConSanFaultMutationKind kind) {
  const auto plan = std::ranges::find(plans, kind, &ConSanFaultMutationPlan::kind);
  return plan == plans.end() ? nullptr : &*plan;
}

[[nodiscard]] bool validate_fault_application_plans(std::span<const ConSanFaultMutationPlan> plans,
                                                    const ProgramInventory &inventory,
                                                    std::vector<std::string> &errors) {
  for (size_t index = 0; index < plans.size(); ++index) {
    const ConSanFaultMutationPlan &plan = plans[index];
    if (!plan.well_formed()) {
      errors.emplace_back("ConSan fault application received malformed typed plan " +
                          std::to_string(index));
      return false;
    }
    if (plan.source_code_object != inventory.code_object_id()) {
      errors.emplace_back(
          "ConSan fault application rejected a plan from a different pristine code object");
      return false;
    }
    if (std::ranges::count(plans, plan.kind, &ConSanFaultMutationPlan::kind) != 1) {
      errors.emplace_back(
          "ConSan fault application received duplicate plans for one mutation kind");
      return false;
    }
  }
  return true;
}

} // namespace

void apply_consan_fault_mutations(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  const ConSanPatchedImageGrowthLimit &growth_limit,
                                  bool require_exactly_one,
                                  std::span<const ConSanFaultMutationPlan> plans,
                                  ConSanTransformArtifacts &result) {
  if (!validate_fault_application_plans(plans, result.program_inventory, result.errors))
    return;
  FaultApplicationState transaction(result);

  if (const ConSanFaultMutationPlan *plan =
          find_fault_plan(plans, ConSanFaultMutationKind::DropBarrier)) {
    try_apply_barrier_drop_fault_patch(code_object, arch, *plan, transaction);
  }
  if (const ConSanFaultMutationPlan *plan =
          find_fault_plan(plans, ConSanFaultMutationKind::MoveBarrierPair);
      plan != nullptr && transaction.errors.empty()) {
    try_apply_barrier_move_fault_patch(code_object, arch, *plan, growth_limit, transaction);
  }
  if (const ConSanFaultMutationPlan *plan =
          find_fault_plan(plans, ConSanFaultMutationKind::BarrierIdScope);
      plan != nullptr && transaction.errors.empty()) {
    try_apply_barrier_id_scope_fault_patch(code_object, *plan, transaction);
  }
  if (const ConSanFaultMutationPlan *plan =
          find_fault_plan(plans, ConSanFaultMutationKind::BarrierParticipantCount);
      plan != nullptr && transaction.errors.empty() &&
      transaction.outcome != ConSanTransformOutcome::Unsupported) {
    try_apply_barrier_participant_fault_patch(code_object, *plan, transaction);
  }

  const ConSanFaultMutationPlan *atomic_address =
      find_fault_plan(plans, ConSanFaultMutationKind::AtomicWrongAddress);
  const ConSanFaultMutationPlan *atomic_order =
      find_fault_plan(plans, ConSanFaultMutationKind::AtomicWeakenOrder);
  const ConSanFaultMutationPlan *atomic_scope =
      find_fault_plan(plans, ConSanFaultMutationKind::AtomicWeakenScope);
  if ((atomic_address || atomic_order || atomic_scope) && transaction.errors.empty()) {
    const ConSanFaultMutationPlan &primary =
        atomic_address ? *atomic_address : (atomic_order ? *atomic_order : *atomic_scope);
    if ((atomic_address && atomic_address->primary_identity != primary.primary_identity) ||
        (atomic_order && atomic_order->primary_identity != primary.primary_identity) ||
        (atomic_scope && atomic_scope->primary_identity != primary.primary_identity)) {
      transaction.errors.emplace_back(
          "ConSan atomic fault plans do not describe one composed semantic target");
    } else {
      try_apply_atomic_fault_patch(code_object, arch, atomic_address, atomic_order, atomic_scope,
                                   transaction);
    }
  }
  if (const ConSanFaultMutationPlan *plan =
          find_fault_plan(plans, ConSanFaultMutationKind::LdsWrongAddress);
      plan != nullptr && transaction.errors.empty()) {
    try_apply_lds_fault_patch(code_object, arch, *plan, transaction);
  }

  const ConSanFaultMutationPlan *ordinary_address =
      find_fault_plan(plans, ConSanFaultMutationKind::OrdinaryWrongAddress);
  const ConSanFaultMutationPlan *ordinary_order =
      find_fault_plan(plans, ConSanFaultMutationKind::OrdinaryWeakenOrder);
  const ConSanFaultMutationPlan *ordinary_scope =
      find_fault_plan(plans, ConSanFaultMutationKind::OrdinaryWeakenScope);
  if ((ordinary_address || ordinary_order || ordinary_scope) && transaction.errors.empty()) {
    const ConSanFaultMutationPlan &primary =
        ordinary_address ? *ordinary_address : (ordinary_order ? *ordinary_order : *ordinary_scope);
    const auto same_target = [&](const ConSanFaultMutationPlan *plan) {
      return plan == nullptr ||
             (plan->primary_identity == primary.primary_identity &&
              plan->selection_sequence_identity == primary.selection_sequence_identity &&
              plan->ordered_member_identities == primary.ordered_member_identities);
    };
    if (!same_target(ordinary_address) || !same_target(ordinary_order) ||
        !same_target(ordinary_scope)) {
      transaction.errors.emplace_back(
          "ConSan ordinary fault plans do not describe one composed semantic target");
    } else {
      try_apply_ordinary_fault_patch(code_object, ordinary_address, ordinary_order, ordinary_scope,
                                     transaction);
    }
  }
  transaction.mutation.fault.applied = applied_fault_mutation_count(transaction.patches);
  const bool mechanisms_succeeded = transaction.successful();
  if (require_exactly_one && transaction.outcome != ConSanTransformOutcome::Unsupported &&
      transaction.mutation.fault.applied != 1u) {
    transaction.errors.emplace_back(
        "ConSan fault injection required exactly one applied mutation, got " +
        std::to_string(transaction.mutation.fault.applied));
  }

  const bool accepted = transaction.successful();
  result.errors = std::move(transaction.errors);
  result.transform_failure_cause = transaction.transform_failure_cause;
  if (!accepted) {
    result.mutation.fault.applied = mechanisms_succeeded
                                        ? transaction.mutation.fault.applied
                                        : applied_fault_mutation_count(result.patches);
    return;
  }
  result.warnings = std::move(transaction.warnings);
  result.outcome = transaction.outcome;
  result.replacement = std::move(transaction.replacement);
  result.patches = std::move(transaction.patches);
  result.mutation = std::move(transaction.mutation);
}

void try_apply_proof_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  ConSanTransformArtifacts &result) {
  auto site = find_preferred_in_place_instruction_site(code_object, arch);
  if (!site) {
    result.errors.emplace_back(
        "ConSan proof endpgm patch found no preferred 4-byte vector ALU site");
    return;
  }

  const uint32_t replacement_endpgm = build_s_endpgm(arch);
  if (!rewrite_word_in_place(code_object, site->file_offset, replacement_endpgm, result))
    return;

  ConSanPatchInfo info;
  info.kind = ConSanPatchKind::InlineEndpgmRewrite;
  info.anchor_offset = site->text_offset;
  info.trampoline_offset = site->text_offset;
  info.original_size = site->size;
  result.patches.push_back(info);
}

[[nodiscard]] const ConSanAccessInventorySite *
find_first_supported_lds_read_b32_site(const ProgramInventory &inventory) {
  const ConSanAccessInventorySite *best = nullptr;
  for (const ConSanAccessInventorySite &site : inventory.access_sites()) {
    if (!site.container.is_kernel() || !site.container.kernel_descriptor_file_offset ||
        site.origin != ConSanAccessOrigin::NativeLds || site.kind != ConSanLdsAccessKind::Read ||
        !site.lowering.replay_guest_access.available() || site.decoded_width_bits != 32)
      continue;
    const ConSanKernelInfo *kernel =
        inventory.find_kernel_by_descriptor(*site.container.kernel_descriptor_file_offset);
    if (kernel == nullptr || kernel->preflight_action == ConSanPreflightAction::Blocked)
      continue;
    if (best == nullptr || site.file_offset < best->file_offset)
      best = &site;
  }
  return best;
}

void try_apply_lds_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                ConSanTransformArtifacts &result) {
  const ConSanAccessInventorySite *site =
      find_first_supported_lds_read_b32_site(result.program_inventory);
  if (site == nullptr) {
    result.warnings.emplace_back("ConSan LDS endpgm proof found no supported ds_read_b32 site");
    return;
  }

  const uint32_t replacement_endpgm = build_s_endpgm(arch);
  if (!rewrite_word_in_place(code_object, site->file_offset, replacement_endpgm, result))
    return;

  ConSanPatchInfo info;
  info.kind = ConSanPatchKind::InlineLdsEndpgmRewrite;
  info.anchor_offset = site->physical_id.original_text_offset;
  info.trampoline_offset = site->physical_id.original_text_offset;
  info.original_size = site->instruction_size;
  result.patches.push_back(info);
}

[[nodiscard]] BarrierSite barrier_site_from_inventory(const ConSanFaultSite &site) {
  return {.text_offset = site.text_offset,
          .file_offset = site.file_offset,
          .size = site.size,
          .range_name = std::string(site.in_kernel ? "kernel:" : "function:") + site.container_name,
          .mnemonic = site.mnemonic};
}

[[nodiscard]] bool owned_by_requested_kernel(std::span<const ConSanExecutionOwner> owners,
                                             const ConSanFaultSelectionView &inventory,
                                             const ConSanOptions &options) {
  return consan_execution_owners_include_requested_kernel(owners, inventory,
                                                          options.test_kernel_name_filter);
}

[[nodiscard]] const ConSanFaultSite *
select_fault_site_for_plan(const ConSanFaultSelectionView &inventory, const ConSanOptions &options,
                           ConSanFaultSiteKind kind, uint32_t legacy_index) {
  return select_fault_site_for_plan(inventory,
                                    ConSanFaultSelection{
                                        .primary_site_identity = options.fault_site_identity,
                                        .primary_sequence_identity = {},
                                        .companion_site_identity = {},
                                        .companion_sequence_identity = {},
                                        .kernel_name_filter = options.test_kernel_name_filter,
                                        .ordinal = legacy_index,
                                    },
                                    kind);
}
