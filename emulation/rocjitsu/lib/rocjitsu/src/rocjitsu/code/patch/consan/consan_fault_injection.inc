struct OrdinaryAcquireMutationTarget {
  const ConSanFaultSite *site = nullptr;
  const ConSanSyncEvent *load = nullptr;
  const ConSanSyncEvent *cache = nullptr;
  const ConSanSyncSequence *sequence = nullptr;
};

[[nodiscard]] std::optional<OrdinaryAcquireMutationTarget>
select_ordinary_acquire_mutation_target(const ConSanResult &result, const ConSanOptions &options) {
  const auto same_owners = [](std::span<const ConSanExecutionOwner> lhs,
                              std::span<const ConSanExecutionOwner> rhs) {
    return !lhs.empty() && lhs.size() == rhs.size() &&
           std::ranges::equal(lhs, rhs, [](const auto &left, const auto &right) {
             return left.descriptor_file_offset == right.descriptor_file_offset &&
                    left.proof == right.proof;
           });
  };
  const auto qualify =
      [&](const ConSanFaultSite &site) -> std::optional<OrdinaryAcquireMutationTarget> {
    if (site.kind != ConSanFaultSiteKind::OrdinaryMemory ||
        site.ordinary_memory_support_reason != ConSanOrdinaryMemorySupportReason::Supported ||
        site.mnemonic != "global_load_b32" || !site.sync_event_identity ||
        !site.sync_sequence_identity ||
        !owned_by_requested_kernel(site.execution_owners, result, options))
      return std::nullopt;
    const ConSanSyncEvent *load = find_sync_event(result, *site.sync_event_identity);
    const auto sequence_it = std::ranges::find(result.sync_sequences, *site.sync_sequence_identity,
                                               &ConSanSyncSequence::identity);
    if (load == nullptr || sequence_it == result.sync_sequences.end())
      return std::nullopt;
    const ConSanSyncSequence *sequence = &*sequence_it;
    if (load->operation != ConSanSyncOperation::OrdinaryLoad || !load->raw_scope ||
        (*load->raw_scope != 2u && *load->raw_scope != 3u) ||
        sequence->kind != ConSanSyncSequenceKind::OrdinaryMemory ||
        sequence->operation != ConSanSyncOperation::OrdinaryLoad ||
        sequence->memory_role != ConSanSyncMemoryRole::Acquire ||
        !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        !sequence->basic_block_index || sequence->member_event_identities.size() != 2u ||
        !sequence_has_exact_members(result, *sequence) ||
        sequence->member_event_identities.front() != load->identity ||
        !same_owners(sequence->execution_owners, site.execution_owners))
      return std::nullopt;
    const ConSanSyncEvent *cache =
        find_sync_event(result, sequence->member_event_identities.back());
    if (cache == nullptr || cache->kind != ConSanSyncEventKind::Fence ||
        cache->mnemonic != "global_inv" || cache->container_name != load->container_name ||
        cache->in_kernel != load->in_kernel ||
        !same_owners(cache->execution_owners, load->execution_owners) || cache->size == 0u ||
        cache->size % sizeof(uint32_t) != 0u)
      return std::nullopt;
    return OrdinaryAcquireMutationTarget{&site, load, cache, sequence};
  };

  if (!options.fault_site_identity.empty()) {
    const ConSanFaultSite *site = find_fault_site_by_identity(result, options.fault_site_identity,
                                                              ConSanFaultSiteKind::OrdinaryMemory);
    return site == nullptr ? std::nullopt : qualify(*site);
  }
  uint32_t index = 0;
  for (const ConSanFaultSite &site : result.fault_sites) {
    auto target = qualify(site);
    if (target && index++ == options.fault_ordinary_index)
      return target;
  }
  return std::nullopt;
}

[[nodiscard]] bool is_member_of_qualified_logical_barrier_pair(const ConSanResult &result,
                                                               const ConSanFaultSite &site) {
  if (!site.sync_event_identity)
    return false;
  const ConSanSyncSequence *sequence =
      find_consan_sync_sequence_for_event(result, *site.sync_event_identity);
  return sequence != nullptr && sequence->operation == ConSanSyncOperation::BarrierFull &&
         consan_sync_confidence_meets(sequence->confidence,
                                      ConSanSemanticConfidence::Conservative) &&
         sequence->member_event_identities.size() == 2u;
}

struct ExactBarrierDropPair {
  const ConSanSyncSequence *sequence = nullptr;
  const ConSanFaultSite *primary = nullptr;
  const ConSanFaultSite *companion = nullptr;
};

[[nodiscard]] std::optional<ExactBarrierDropPair>
resolve_exact_barrier_drop_pair(const ConSanResult &result, const ConSanOptions &options,
                                std::string *reason) {
  if (options.fault_site_identity.empty() || options.fault_barrier_sequence_identity.empty()) {
    *reason = "an exact site identity and logical sequence identity are both required";
    return std::nullopt;
  }
  const auto sequence =
      std::ranges::find(result.sync_sequences, options.fault_barrier_sequence_identity,
                        &ConSanSyncSequence::identity);
  if (sequence == result.sync_sequences.end()) {
    *reason = "the exact logical sequence identity was not found";
    return std::nullopt;
  }
  if (sequence->kind != ConSanSyncSequenceKind::Barrier ||
      sequence->operation != ConSanSyncOperation::BarrierFull ||
      !consan_sync_confidence_meets(sequence->confidence, ConSanSemanticConfidence::Conservative) ||
      sequence->member_event_identities.size() != 2u ||
      !sequence_has_exact_members(result, *sequence) ||
      !owned_by_requested_kernel(sequence->execution_owners, result, options)) {
    *reason = "the exact sequence is not an owned complete conservative two-member barrier";
    return std::nullopt;
  }

  const ConSanFaultSite *primary = find_fault_site_by_identity(result, options.fault_site_identity,
                                                               ConSanFaultSiteKind::Barrier);
  if (primary == nullptr || !primary->sync_event_identity || !primary->sync_sequence_identity ||
      *primary->sync_sequence_identity != sequence->identity ||
      std::ranges::find(sequence->member_event_identities, *primary->sync_event_identity) ==
          sequence->member_event_identities.end() ||
      !owned_by_requested_kernel(primary->execution_owners, result, options)) {
    *reason = "the exact site is not a member of the requested logical barrier";
    return std::nullopt;
  }

  const ConSanFaultSite *companion = nullptr;
  for (const std::string &member_identity : sequence->member_event_identities) {
    const auto matching_site =
        std::ranges::find_if(result.fault_sites, [&](const ConSanFaultSite &candidate) {
          return candidate.kind == ConSanFaultSiteKind::Barrier &&
                 candidate.sync_event_identity == member_identity &&
                 candidate.sync_sequence_identity == sequence->identity &&
                 candidate.container_name == sequence->container_name &&
                 candidate.in_kernel == sequence->in_kernel &&
                 owned_by_requested_kernel(candidate.execution_owners, result, options);
        });
    if (matching_site == result.fault_sites.end()) {
      *reason = "a logical barrier member has no exact owned patch site";
      return std::nullopt;
    }
    const auto duplicate = std::ranges::find_if(
        std::next(matching_site), result.fault_sites.end(), [&](const ConSanFaultSite &candidate) {
          return candidate.kind == ConSanFaultSiteKind::Barrier &&
                 candidate.sync_event_identity == member_identity;
        });
    if (duplicate != result.fault_sites.end()) {
      *reason = "a logical barrier member maps to multiple patch sites";
      return std::nullopt;
    }
    if (matching_site->identity != primary->identity)
      companion = &*matching_site;
  }
  if (companion == nullptr || primary->size != sizeof(uint32_t) ||
      companion->size != sizeof(uint32_t) || primary->file_offset == companion->file_offset) {
    *reason = "the exact logical barrier does not have two distinct one-word patch sites";
    return std::nullopt;
  }
  return ExactBarrierDropPair{&*sequence, primary, companion};
}

[[nodiscard]] std::optional<ExactBarrierDropPair>
resolve_exact_barrier_drop_pair(const ConSanResult &result, const ConSanOptions &options,
                                std::string_view site_identity, std::string_view sequence_identity,
                                std::string *reason) {
  ConSanOptions selection = options;
  selection.fault_site_identity = site_identity;
  selection.fault_barrier_sequence_identity = sequence_identity;
  selection.fault_barrier_companion_site_identity.clear();
  selection.fault_barrier_companion_sequence_identity.clear();
  return resolve_exact_barrier_drop_pair(result, selection, reason);
}

struct ExactBarrierDropGroup {
  ExactBarrierDropPair first;
  ExactBarrierDropPair second;
};

[[nodiscard]] bool same_execution_owners(std::span<const ConSanExecutionOwner> lhs,
                                         std::span<const ConSanExecutionOwner> rhs) {
  return !lhs.empty() && lhs.size() == rhs.size() &&
         std::ranges::equal(lhs, rhs, [](const auto &left, const auto &right) {
           return left.descriptor_file_offset == right.descriptor_file_offset &&
                  left.proof == right.proof;
         });
}

[[nodiscard]] std::pair<uint64_t, uint64_t>
exact_barrier_drop_pair_range(const ExactBarrierDropPair &pair) {
  const uint64_t begin = std::min(pair.primary->text_offset, pair.companion->text_offset);
  const uint64_t end = std::max(pair.primary->text_offset + pair.primary->size,
                                pair.companion->text_offset + pair.companion->size);
  return {begin, end};
}

[[nodiscard]] std::string exact_barrier_drop_group_identity(const ExactBarrierDropGroup &group) {
  return "barrier-group[" + group.first.sequence->identity + "][" +
         group.second.sequence->identity + "]";
}

[[nodiscard]] std::optional<ExactBarrierDropGroup>
resolve_exact_barrier_drop_group(const ConSanResult &result, const ConSanOptions &options,
                                 std::string *reason) {
  if (options.fault_barrier_companion_site_identity.empty() ||
      options.fault_barrier_companion_sequence_identity.empty()) {
    *reason = "a grouped drop requires an exact companion site and sequence identity";
    return std::nullopt;
  }
  std::string member_reason;
  const auto first =
      resolve_exact_barrier_drop_pair(result, options, options.fault_site_identity,
                                      options.fault_barrier_sequence_identity, &member_reason);
  if (!first) {
    *reason = "group member zero is invalid: " + member_reason;
    return std::nullopt;
  }
  const auto second = resolve_exact_barrier_drop_pair(
      result, options, options.fault_barrier_companion_site_identity,
      options.fault_barrier_companion_sequence_identity, &member_reason);
  if (!second) {
    *reason = "group member one is invalid: " + member_reason;
    return std::nullopt;
  }
  const auto first_range = exact_barrier_drop_pair_range(*first);
  const auto second_range = exact_barrier_drop_pair_range(*second);
  if (first->sequence->identity == second->sequence->identity ||
      first_range.second > second_range.first) {
    *reason = "the two exact pairs are duplicate, overlapping, or not strictly ordered";
    return std::nullopt;
  }
  if (first->sequence->container_name != second->sequence->container_name ||
      first->sequence->in_kernel != second->sequence->in_kernel ||
      !same_execution_owners(first->sequence->execution_owners,
                             second->sequence->execution_owners)) {
    *reason = "the two exact pairs do not have identical container and execution owners";
    return std::nullopt;
  }
  return ExactBarrierDropGroup{*first, *second};
}

[[nodiscard]] const ConSanBarrierLifecycleGroup *
find_lifecycle_group_for_sequence(const ConSanResult &result, const ConSanSyncSequence &sequence) {
  const auto group = std::ranges::find_if(result.barrier_lifecycle_groups, [&](const auto &item) {
    return std::ranges::all_of(sequence.member_event_identities, [&](const std::string &identity) {
      return std::ranges::find(item.member_event_identities, identity) !=
             item.member_event_identities.end();
    });
  });
  return group == result.barrier_lifecycle_groups.end() ? nullptr : &*group;
}

[[nodiscard]] const ConSanBarrierSite *
find_static_participant_init(const ConSanResult &result, const ConSanBarrierLifecycleGroup &group) {
  const auto init_event = std::ranges::find_if(result.sync_events, [&](const auto &event) {
    return event.operation == ConSanSyncOperation::BarrierInit &&
           event.container_name == group.container_name && event.in_kernel == group.in_kernel &&
           std::ranges::find(group.member_event_identities, event.identity) !=
               group.member_event_identities.end();
  });
  if (init_event == result.sync_events.end())
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
  return group.in_kernel ? find_in(result.kernels) : find_in(result.functions);
}

void mark_participant_mutation_unsupported(ConSanResult &result, std::string_view reason) {
  result.outcome = ConSanTransformOutcome::Unsupported;
  result.warnings.emplace_back("ConSan barrier participant mutation is unsupported: " +
                               std::string(reason));
}

[[nodiscard]] size_t requested_fault_mutation_count(const ConSanOptions &options) {
  return static_cast<size_t>(options.fault_drop_barrier) +
         static_cast<size_t>(options.fault_move_barrier) +
         static_cast<size_t>(options.fault_mutate_barrier_id_scope) +
         static_cast<size_t>(options.fault_mutate_barrier_participants) +
         static_cast<size_t>(options.fault_atomic_wrong_address) +
         static_cast<size_t>(options.fault_atomic_weaken_order) +
         static_cast<size_t>(options.fault_atomic_weaken_scope) +
         static_cast<size_t>(options.fault_ordinary_wrong_address) +
         static_cast<size_t>(options.fault_ordinary_weaken_order) +
         static_cast<size_t>(options.fault_ordinary_weaken_scope);
}

[[nodiscard]] size_t applied_fault_mutation_count(const ConSanResult &result) {
  size_t count = 0;
  size_t move_targets = 0;
  size_t barrier_id_scope_members = 0;
  size_t exact_barrier_drop_members = 0;
  std::vector<std::string> atomic_order_mutations;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.phase != ConSanPatchPhase::Mutation)
      continue;
    switch (patch.kind) {
    case ConSanPatchKind::InlineBarrierNopRewrite:
      if (!patch.fault_sequence_identity.empty()) {
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
      if (patch.fault_sequence_identity.empty()) {
        ++count;
      } else if (std::ranges::find(atomic_order_mutations, patch.fault_sequence_identity) ==
                 atomic_order_mutations.end()) {
        atomic_order_mutations.push_back(patch.fault_sequence_identity);
      }
      break;
    case ConSanPatchKind::InlineAtomicScopeRewrite:
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
  }
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

void build_fault_mutation_plan(const ConSanOptions &options, rj_code_arch_t arch,
                               ConSanResult &result) {
  auto append_primary = [&](ConSanFaultMutationKind kind, const ConSanFaultSite *site) {
    if (site == nullptr) {
      result.warnings.emplace_back("ConSan fault dry run found no site for the requested mutation");
      return;
    }
    result.fault_plans.push_back(
        {.kind = kind,
         .primary_identity = site->identity,
         .companion_identity = std::nullopt,
         .logical_sequence_identity = std::nullopt,
         .ordered_member_identities = {},
         .destination_identity = std::nullopt,
         .barrier_move_direction = ConSanBarrierMoveDirection::LegacyMarker,
         .structured_guard_block_index = std::nullopt,
         .structured_destination_block_index = std::nullopt,
         .structured_source_block_index = std::nullopt,
         .structured_guard_offset = std::nullopt,
         .structured_destination_offset = std::nullopt,
         .structured_source_offset = std::nullopt,
         .original_barrier_id = std::nullopt,
         .target_barrier_id = std::nullopt,
         .original_barrier_scope = ConSanBarrierSite::Scope::Unknown,
         .target_barrier_scope = ConSanBarrierSite::Scope::Unknown,
         .original_participant_count = std::nullopt,
         .target_participant_count = std::nullopt});
  };

  if (options.fault_drop_barrier) {
    const ConSanFaultSite *primary = select_fault_site_for_plan(
        result, options, ConSanFaultSiteKind::Barrier, options.fault_barrier_index);
    if (!options.fault_barrier_sequence_identity.empty() ||
        !options.fault_barrier_companion_site_identity.empty() ||
        !options.fault_barrier_companion_sequence_identity.empty()) {
      std::string reason;
      const bool grouped = !options.fault_barrier_companion_site_identity.empty() ||
                           !options.fault_barrier_companion_sequence_identity.empty();
      const auto group =
          grouped ? resolve_exact_barrier_drop_group(result, options, &reason) : std::nullopt;
      const auto pair = grouped ? std::optional<ExactBarrierDropPair>{}
                                : resolve_exact_barrier_drop_pair(result, options, &reason);
      if ((grouped && !group) || (!grouped && !pair)) {
        result.warnings.emplace_back("ConSan exact whole-barrier drop dry run rejected: " + reason);
      } else {
        ConSanFaultMutationPlan plan;
        plan.kind = ConSanFaultMutationKind::DropBarrier;
        if (group) {
          plan.primary_identity = group->first.primary->identity;
          plan.companion_identity = group->first.companion->identity;
          plan.logical_sequence_identity = exact_barrier_drop_group_identity(*group);
          plan.ordered_member_identities = group->first.sequence->member_event_identities;
          plan.ordered_member_identities.insert(
              plan.ordered_member_identities.end(),
              group->second.sequence->member_event_identities.begin(),
              group->second.sequence->member_event_identities.end());
        } else {
          plan.primary_identity = pair->primary->identity;
          plan.companion_identity = pair->companion->identity;
          plan.logical_sequence_identity = pair->sequence->identity;
          plan.ordered_member_identities = pair->sequence->member_event_identities;
        }
        result.fault_plans.push_back(std::move(plan));
      }
    } else if (primary != nullptr &&
               is_member_of_qualified_logical_barrier_pair(result, *primary) &&
               !options.fault_allow_destructive_incomplete_barrier_drop) {
      result.warnings.emplace_back(
          "ConSan fault dry run rejected dropping one half of a qualified logical barrier pair; "
          "explicit destructive incomplete-drop mode is required");
    } else {
      append_primary(ConSanFaultMutationKind::DropBarrier, primary);
    }
  }
  if (options.fault_move_barrier) {
    const ConSanFaultSite *primary = select_fault_site_for_plan(
        result, options, ConSanFaultSiteKind::Barrier, options.fault_barrier_index);
    const ConSanSyncSequence *sequence =
        primary != nullptr && primary->sync_event_identity
            ? find_consan_sync_sequence_for_event(result, *primary->sync_event_identity)
            : nullptr;
    if (primary == nullptr || sequence == nullptr ||
        !owned_by_requested_kernel(sequence->execution_owners, result, options) ||
        sequence->operation != ConSanSyncOperation::BarrierFull ||
        !consan_sync_confidence_meets(sequence->confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        sequence->member_event_identities.size() != 2u) {
      result.warnings.emplace_back("ConSan fault dry run found no barrier pair to move");
    } else {
      const auto companion = std::ranges::find_if(result.fault_sites, [&](const auto &candidate) {
        return candidate.kind == ConSanFaultSiteKind::Barrier && candidate.sync_event_identity &&
               owned_by_requested_kernel(candidate.execution_owners, result, options) &&
               candidate.identity != primary->identity &&
               std::ranges::find(sequence->member_event_identities,
                                 *candidate.sync_event_identity) !=
                   sequence->member_event_identities.end();
      });
      if (companion == result.fault_sites.end()) {
        result.warnings.emplace_back(
            "ConSan fault dry run found no completing barrier in the selected container");
      } else {
        ConSanFaultMutationPlan plan{
            .kind = ConSanFaultMutationKind::MoveBarrierPair,
            .primary_identity = primary->identity,
            .companion_identity = companion->identity,
            .logical_sequence_identity = sequence->identity,
            .ordered_member_identities = sequence->member_event_identities,
            .destination_identity = std::nullopt,
            .barrier_move_direction = options.fault_barrier_move_direction,
            .structured_guard_block_index = std::nullopt,
            .structured_destination_block_index = std::nullopt,
            .structured_source_block_index = std::nullopt,
            .structured_guard_offset = std::nullopt,
            .structured_destination_offset = std::nullopt,
            .structured_source_offset = std::nullopt,
            .original_barrier_id = std::nullopt,
            .target_barrier_id = std::nullopt,
            .original_barrier_scope = ConSanBarrierSite::Scope::Unknown,
            .target_barrier_scope = ConSanBarrierSite::Scope::Unknown,
            .original_participant_count = std::nullopt,
            .target_participant_count = std::nullopt,
        };
        if (options.fault_barrier_move_direction == ConSanBarrierMoveDirection::LegacyMarker) {
          result.fault_plans.push_back(std::move(plan));
        } else if (options.fault_barrier_destination_identity.empty()) {
          result.warnings.emplace_back(
              "ConSan fault dry run markerless barrier move requires an exact destination "
              "identity");
        } else {
          const auto destination = std::ranges::find(result.barrier_move_destinations,
                                                     options.fault_barrier_destination_identity,
                                                     &ConSanBarrierMoveDestination::identity);
          if (destination == result.barrier_move_destinations.end()) {
            result.warnings.emplace_back(
                "ConSan fault dry run found no barrier-move destination with exact identity '" +
                options.fault_barrier_destination_identity + "'");
          } else if (!owned_by_requested_kernel(destination->execution_owners, result, options)) {
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
                     !admits_cross_block_barrier_move(*destination, options)) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected control-flow or container boundary: cross-block "
                "barrier move lacks a matching proven completing or destructive structured "
                "diamond opt-in");
          } else if (destination->text_offset < sequence->end_text_offset &&
                     sequence->begin_text_offset < destination->text_offset + destination->size) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected barrier-move destination overlapping the "
                "selected logical barrier");
          } else if (!destination->suitable) {
            result.warnings.emplace_back(
                "ConSan fault dry run rejected unsuitable barrier-move destination: " +
                destination->rejection_reason);
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
      result.warnings.emplace_back(
          "ConSan barrier ID/scope dry run requires an exact logical sequence identity");
    } else {
      const auto selected =
          std::ranges::find(result.sync_sequences, options.fault_barrier_sequence_identity,
                            &ConSanSyncSequence::identity);
      if (selected == result.sync_sequences.end()) {
        result.warnings.emplace_back(
            "ConSan barrier ID/scope dry run found no logical sequence with exact identity '" +
            options.fault_barrier_sequence_identity + "'");
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
                   find_lifecycle_group_for_sequence(result, *sequence);
               group != nullptr && !group->admissible) {
      result.warnings.emplace_back(
          "ConSan barrier ID/scope dry run rejected a partial unresolved lifecycle mutation: " +
          group->rejection_reason);
    } else if (!options.fault_barrier_target_id) {
      result.warnings.emplace_back("ConSan barrier ID/scope dry run requires a target barrier ID");
    } else if (target_scope == ConSanBarrierSite::Scope::Unknown) {
      result.warnings.emplace_back(
          "ConSan barrier ID/scope dry run rejected a target ID with unknown scope");
    } else if (*options.fault_barrier_target_id == *sequence->barrier_id) {
      result.warnings.emplace_back(
          "ConSan barrier ID/scope dry run requires a target different from the original ID");
    } else if (const ConSanBarrierLifecycleGroup *group =
                   find_lifecycle_group_for_sequence(result, *sequence);
               group != nullptr &&
               (*options.fault_barrier_target_id < 1 || *options.fault_barrier_target_id > 16)) {
      result.warnings.emplace_back(
          "ConSan barrier lifecycle retarget requires a named-barrier target ID in [1, 16]");
    } else {
      std::array<const ConSanFaultSite *, 2> members{};
      for (size_t i = 0; i < members.size(); ++i) {
        const std::string &event_identity = sequence->member_event_identities[i];
        const auto member = std::ranges::find_if(result.fault_sites, [&](const auto &candidate) {
          return candidate.kind == ConSanFaultSiteKind::Barrier &&
                 candidate.sync_event_identity == event_identity;
        });
        if (member != result.fault_sites.end())
          members[i] = &*member;
      }
      if (members[0] == nullptr || members[1] == nullptr) {
        result.warnings.emplace_back(
            "ConSan barrier ID/scope dry run could not resolve both sequence members");
      } else {
        const ConSanBarrierLifecycleGroup *group =
            find_lifecycle_group_for_sequence(result, *sequence);
        result.fault_plans.push_back(
            {.kind = ConSanFaultMutationKind::BarrierIdScope,
             .primary_identity = members[0]->identity,
             .companion_identity = members[1]->identity,
             .logical_sequence_identity = group != nullptr ? group->identity : sequence->identity,
             .ordered_member_identities = group != nullptr ? group->member_event_identities
                                                           : sequence->member_event_identities,
             .destination_identity = std::nullopt,
             .barrier_move_direction = ConSanBarrierMoveDirection::LegacyMarker,
             .structured_guard_block_index = std::nullopt,
             .structured_destination_block_index = std::nullopt,
             .structured_source_block_index = std::nullopt,
             .structured_guard_offset = std::nullopt,
             .structured_destination_offset = std::nullopt,
             .structured_source_offset = std::nullopt,
             .original_barrier_id = sequence->barrier_id,
             .target_barrier_id = options.fault_barrier_target_id,
             .original_barrier_scope = sequence->barrier_scope,
             .target_barrier_scope = target_scope,
             .original_participant_count = std::nullopt,
             .target_participant_count = std::nullopt});
      }
    }
  }
  if (options.fault_mutate_barrier_participants) {
    const auto selected =
        std::ranges::find(result.sync_sequences, options.fault_barrier_sequence_identity,
                          &ConSanSyncSequence::identity);
    const ConSanBarrierLifecycleGroup *group =
        selected == result.sync_sequences.end()
            ? nullptr
            : find_lifecycle_group_for_sequence(result, *selected);
    const ConSanBarrierSite *init = group != nullptr && group->admissible
                                        ? find_static_participant_init(result, *group)
                                        : nullptr;
    if (options.fault_barrier_target_participant_mask) {
      mark_participant_mutation_unsupported(
          result, "the GFX12.5 lifecycle exposes a six-bit count in M0, not a participant mask");
    } else if (group == nullptr || !group->admissible || init == nullptr ||
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
      result.fault_plans.push_back(
          {.kind = ConSanFaultMutationKind::BarrierParticipantCount,
           .primary_identity = group->member_event_identities.front(),
           .companion_identity = std::nullopt,
           .logical_sequence_identity = group->identity,
           .ordered_member_identities = group->member_event_identities,
           .destination_identity = std::nullopt,
           .barrier_move_direction = ConSanBarrierMoveDirection::LegacyMarker,
           .structured_guard_block_index = std::nullopt,
           .structured_destination_block_index = std::nullopt,
           .structured_source_block_index = std::nullopt,
           .structured_guard_offset = std::nullopt,
           .structured_destination_offset = std::nullopt,
           .structured_source_offset = std::nullopt,
           .original_barrier_id = group->barrier_id,
           .target_barrier_id = group->barrier_id,
           .original_barrier_scope = group->barrier_scope,
           .target_barrier_scope = group->barrier_scope,
           .original_participant_count = init->participant_count,
           .target_participant_count = options.fault_barrier_target_participant_count});
    }
  }
  const ConSanFaultSite *atomic = nullptr;
  if (options.fault_atomic_wrong_address || options.fault_atomic_weaken_order ||
      options.fault_atomic_weaken_scope) {
    atomic = select_fault_site_for_plan(result, options, ConSanFaultSiteKind::Atomic,
                                        options.fault_atomic_index);
  }
  if (options.fault_atomic_wrong_address) {
    const AtomicFaultEncoding encoding =
        atomic == nullptr ? AtomicFaultEncoding::Unsupported
                          : atomic_fault_encoding(atomic->mnemonic, atomic->size, arch);
    if (!atomic_fault_supports_address(encoding)) {
      result.warnings.emplace_back(
          "ConSan fault dry run cannot rewrite the address for the selected atomic encoding");
    } else {
      append_primary(ConSanFaultMutationKind::AtomicWrongAddress, atomic);
    }
  }
  if (options.fault_atomic_weaken_order) {
    const AtomicFaultEncoding encoding =
        atomic == nullptr ? AtomicFaultEncoding::Unsupported
                          : atomic_fault_encoding(atomic->mnemonic, atomic->size, arch);
    const ConSanSyncSequence *sequence =
        atomic_fault_supports_order(encoding) && atomic->sync_event_identity
            ? find_consan_sync_sequence_for_event(result, *atomic->sync_event_identity)
            : nullptr;
    const ConSanSyncEvent *fence = nullptr;
    std::optional<uint64_t> release_wait_text_offset;
    if (sequence != nullptr &&
        consan_sync_confidence_meets(sequence->memory_role_confidence,
                                     ConSanSemanticConfidence::Conservative)) {
      if (options.fault_atomic_order_edge != ConSanAtomicOrderEdge::Acquire &&
          encoding != AtomicFaultEncoding::CdnaFlat)
        release_wait_text_offset = sequence->release_wait_text_offset;
      for (const std::string &identity : sequence->member_event_identities) {
        const auto event = std::ranges::find_if(result.sync_events, [&](const auto &candidate) {
          return candidate.identity == identity && candidate.kind == ConSanSyncEventKind::Fence;
        });
        const bool edge_matches =
            event != result.sync_events.end() &&
            (options.fault_atomic_order_edge == ConSanAtomicOrderEdge::Any ||
             (options.fault_atomic_order_edge == ConSanAtomicOrderEdge::Release &&
              is_release_cache_event(&*event)) ||
             (options.fault_atomic_order_edge == ConSanAtomicOrderEdge::Acquire &&
              is_acquire_cache_event(&*event)));
        if (edge_matches) {
          fence = &*event;
          break;
        }
      }
    }
    if (atomic == nullptr || (fence == nullptr && !release_wait_text_offset)) {
      result.warnings.emplace_back(
          "ConSan fault dry run found no qualified atomic ordering boundary to weaken");
    } else {
      result.fault_plans.push_back(
          {.kind = ConSanFaultMutationKind::AtomicWeakenOrder,
           .primary_identity = atomic->identity,
           .companion_identity =
               fence != nullptr
                   ? std::optional<std::string>(fence->identity)
                   : std::optional<std::string>(sequence->identity + "|release-wait-boundary"),
           .logical_sequence_identity = std::nullopt,
           .ordered_member_identities = {},
           .destination_identity = std::nullopt,
           .barrier_move_direction = ConSanBarrierMoveDirection::LegacyMarker,
           .structured_guard_block_index = std::nullopt,
           .structured_destination_block_index = std::nullopt,
           .structured_source_block_index = std::nullopt,
           .structured_guard_offset = std::nullopt,
           .structured_destination_offset = std::nullopt,
           .structured_source_offset = std::nullopt,
           .original_barrier_id = std::nullopt,
           .target_barrier_id = std::nullopt,
           .original_barrier_scope = ConSanBarrierSite::Scope::Unknown,
           .target_barrier_scope = ConSanBarrierSite::Scope::Unknown,
           .original_participant_count = std::nullopt,
           .target_participant_count = std::nullopt});
    }
  }
  if (options.fault_atomic_weaken_scope) {
    const AtomicFaultEncoding encoding =
        atomic == nullptr ? AtomicFaultEncoding::Unsupported
                          : atomic_fault_encoding(atomic->mnemonic, atomic->size, arch);
    if (!atomic_fault_supports_scope(encoding)) {
      result.warnings.emplace_back(
          "ConSan fault dry run cannot weaken scope for the selected atomic encoding");
    } else {
      append_primary(ConSanFaultMutationKind::AtomicWeakenScope, atomic);
    }
  }
  if (options.fault_ordinary_wrong_address || options.fault_ordinary_weaken_order ||
      options.fault_ordinary_weaken_scope) {
    const auto target = select_ordinary_acquire_mutation_target(result, options);
    if (!target) {
      result.warnings.emplace_back(
          "ConSan fault dry run found no exact supported ordinary acquire-load sequence");
    } else {
      const auto append_ordinary = [&](ConSanFaultMutationKind kind,
                                       std::optional<std::string> companion) {
        ConSanFaultMutationPlan plan;
        plan.kind = kind;
        plan.primary_identity = target->site->identity;
        plan.companion_identity = std::move(companion);
        plan.logical_sequence_identity = target->sequence->identity;
        plan.ordered_member_identities = target->sequence->member_event_identities;
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
  result.planned_fault_mutations = result.fault_plans.size();
  if (options.fault_require_exactly_one && result.outcome != ConSanTransformOutcome::Unsupported &&
      result.planned_fault_mutations != 1u) {
    result.errors.emplace_back("ConSan fault dry run required exactly one planned mutation, got " +
                               std::to_string(result.planned_fault_mutations));
  }
}

[[nodiscard]] bool has_proof_nop_candidate(const ConSanResult &result) {
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == ConSanPreflightAction::Candidate)
      return true;
  }
  return false;
}

void try_apply_proof_nop_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                               bool force_trampoline, ConSanResult &result) {
  if (has_only_rocclr_runtime_kernels(result)) {
    result.warnings.emplace_back(
        "ConSan proof NOP patch skipped ROCclr runtime helper code object");
    return;
  }
  if (!has_proof_nop_candidate(result)) {
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
      result.modified = true;
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

  result.elf_bytes = std::move(patched.elf_bytes);
  result.modified = true;
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

void try_apply_barrier_drop_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                        const ConSanOptions &options, ConSanResult &result) {
  if (has_only_rocclr_runtime_kernels(result)) {
    result.warnings.emplace_back("ConSan barrier fault skipped ROCclr runtime helper code object");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());

  if (!options.fault_barrier_sequence_identity.empty() ||
      !options.fault_barrier_companion_site_identity.empty() ||
      !options.fault_barrier_companion_sequence_identity.empty()) {
    std::string reason;
    const bool grouped = !options.fault_barrier_companion_site_identity.empty() ||
                         !options.fault_barrier_companion_sequence_identity.empty();
    const auto group =
        grouped ? resolve_exact_barrier_drop_group(result, options, &reason) : std::nullopt;
    const auto pair = grouped ? std::optional<ExactBarrierDropPair>{}
                              : resolve_exact_barrier_drop_pair(result, options, &reason);
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

    if (result.elf_bytes.empty())
      result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());
    const uint32_t replacement_nop = build_s_nop(0, arch);
    for (const ConSanFaultSite *member : members) {
      const ExactBarrierDropPair &owning_pair =
          group && (member == group->second.primary || member == group->second.companion)
              ? group->second
              : (group ? group->first : *pair);
      std::memcpy(result.elf_bytes.data() + member->file_offset, &replacement_nop,
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
    result.modified = true;
    result.applied_fault_logical_identity =
        group ? exact_barrier_drop_group_identity(*group) : pair->sequence->identity;
    result.warnings.emplace_back(group ? "ConSan exact grouped-barrier fault transactionally "
                                         "rewrote four physical members"
                                       : "ConSan exact whole-barrier fault transactionally rewrote "
                                         "both logical members");
    return;
  }

  std::optional<BarrierSite> site;
  const ConSanFaultSite *selected_inventory_site = select_fault_site_for_plan(
      result, options, ConSanFaultSiteKind::Barrier, options.fault_barrier_index);
  if (selected_inventory_site != nullptr)
    site = barrier_site_from_inventory(*selected_inventory_site);
  if (!site) {
    if (!options.fault_site_identity.empty()) {
      result.warnings.emplace_back("ConSan barrier fault found no site with exact identity '" +
                                   options.fault_site_identity + "'");
    } else {
      result.warnings.emplace_back("ConSan barrier fault found no patchable barrier at index " +
                                   std::to_string(options.fault_barrier_index));
    }
    return;
  }
  if (selected_inventory_site != nullptr &&
      is_member_of_qualified_logical_barrier_pair(result, *selected_inventory_site) &&
      !options.fault_allow_destructive_incomplete_barrier_drop) {
    result.warnings.emplace_back(
        "ConSan barrier fault rejected dropping one half of a qualified logical barrier pair; "
        "set explicit destructive incomplete-drop mode only in a contained run");
    return;
  }

  if (result.elf_bytes.empty())
    result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());

  const uint32_t replacement_nop = build_s_nop(0, arch);
  if (site->file_offset > result.elf_bytes.size() ||
      sizeof(replacement_nop) > result.elf_bytes.size() - site->file_offset) {
    result.errors.emplace_back("ConSan barrier fault offset is outside ELF bytes");
    if (!result.modified)
      result.elf_bytes.clear();
    return;
  }

  std::memcpy(result.elf_bytes.data() + site->file_offset, &replacement_nop,
              sizeof(replacement_nop));

  ConSanPatchInfo info;
  info.phase = ConSanPatchPhase::Mutation;
  info.kind = ConSanPatchKind::InlineBarrierNopRewrite;
  info.anchor_offset = site->text_offset;
  info.trampoline_offset = site->text_offset;
  info.original_size = site->size;
  result.patches.push_back(info);
  result.modified = true;
  result.warnings.emplace_back("ConSan barrier fault rewrote " + site->mnemonic + " in " +
                               site->range_name);
}

[[nodiscard]] std::optional<uint8_t> inline_barrier_selector_for_id(int32_t barrier_id) {
  if (barrier_id >= 0 && barrier_id <= 64)
    return static_cast<uint8_t>(128 + barrier_id);
  if (barrier_id >= -16 && barrier_id < 0)
    return static_cast<uint8_t>(192 - barrier_id);
  return std::nullopt;
}

void try_apply_barrier_id_scope_fault_patch(const AmdGpuCodeObject &code_object,
                                            const ConSanOptions &options, ConSanResult &result) {
  if (options.fault_barrier_sequence_identity.empty() || !options.fault_barrier_target_id) {
    result.errors.emplace_back(
        "ConSan barrier ID/scope mutation requires an exact sequence identity and target ID");
    return;
  }
  const auto selected =
      std::ranges::find(result.sync_sequences, options.fault_barrier_sequence_identity,
                        &ConSanSyncSequence::identity);
  if (selected == result.sync_sequences.end() ||
      selected->kind != ConSanSyncSequenceKind::Barrier ||
      selected->operation != ConSanSyncOperation::BarrierFull ||
      !consan_sync_confidence_meets(selected->confidence, ConSanSemanticConfidence::Conservative) ||
      selected->member_event_identities.size() != 2u || !selected->barrier_id ||
      selected->barrier_scope == ConSanBarrierSite::Scope::Unknown) {
    result.errors.emplace_back(
        "ConSan barrier ID/scope mutation requires an exact qualified two-member sequence");
    return;
  }
  const ConSanBarrierLifecycleGroup *lifecycle_group =
      find_lifecycle_group_for_sequence(result, *selected);
  if (lifecycle_group != nullptr && !lifecycle_group->admissible) {
    result.errors.emplace_back(
        "ConSan barrier ID/scope mutation refused a partial unresolved lifecycle mutation: " +
        lifecycle_group->rejection_reason);
    return;
  }
  const int32_t target_id = *options.fault_barrier_target_id;
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
    const auto event = std::ranges::find(result.sync_events, identity, &ConSanSyncEvent::identity);
    if (event == result.sync_events.end()) {
      result.errors.emplace_back(
          "ConSan barrier ID/scope mutation could not resolve every logical member");
      return;
    }
    members.push_back(&*event);
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

  result.elf_bytes = std::move(replacement);
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
  result.modified = true;
  result.applied_fault_logical_identity =
      lifecycle_group != nullptr ? lifecycle_group->identity : selected->identity;
  result.warnings.emplace_back(
      std::string("ConSan barrier ID/scope mutation rewrote one logical ") +
      (lifecycle_group != nullptr ? "lifecycle" : "pair") + " from " +
      std::to_string(*selected->barrier_id) + " to " + std::to_string(target_id));
}

void try_apply_barrier_participant_fault_patch(const AmdGpuCodeObject &code_object,
                                               const ConSanOptions &options, ConSanResult &result) {
  const auto selected =
      std::ranges::find(result.sync_sequences, options.fault_barrier_sequence_identity,
                        &ConSanSyncSequence::identity);
  const ConSanBarrierLifecycleGroup *group =
      selected == result.sync_sequences.end()
          ? nullptr
          : find_lifecycle_group_for_sequence(result, *selected);
  const ConSanBarrierSite *init = group != nullptr && group->admissible
                                      ? find_static_participant_init(result, *group)
                                      : nullptr;
  if (options.fault_barrier_target_participant_mask) {
    mark_participant_mutation_unsupported(
        result, "the GFX12.5 lifecycle exposes a six-bit count in M0, not a participant mask");
    return;
  }
  if (group == nullptr || !group->admissible || init == nullptr ||
      init->participant_encoding != ConSanBarrierSite::ParticipantEncoding::M0Literal32 ||
      !init->participant_count || !init->participant_setup_text_offset ||
      !init->participant_setup_file_offset || !init->participant_setup_literal) {
    mark_participant_mutation_unsupported(
        result, "no complete lifecycle has an immediately preceding literal M0 count setup");
    return;
  }
  if (!options.fault_barrier_target_participant_count ||
      *options.fault_barrier_target_participant_count == 0u ||
      *options.fault_barrier_target_participant_count > 63u ||
      *options.fault_barrier_target_participant_count == *init->participant_count) {
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
  replacement_literal = (replacement_literal & ~kParticipantCountMask) |
                        (*options.fault_barrier_target_participant_count << 16u);
  result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());
  std::memcpy(result.elf_bytes.data() + literal_file_offset, &replacement_literal,
              sizeof(replacement_literal));

  ConSanPatchInfo patch;
  patch.phase = ConSanPatchPhase::Mutation;
  patch.kind = ConSanPatchKind::InlineBarrierParticipantCountRewrite;
  patch.anchor_offset = *init->participant_setup_text_offset;
  patch.trampoline_offset = patch.anchor_offset;
  patch.original_size = 2u * sizeof(uint32_t);
  patch.original_participant_count = init->participant_count;
  patch.target_participant_count = options.fault_barrier_target_participant_count;
  result.patches.push_back(std::move(patch));
  result.modified = true;
  result.applied_fault_logical_identity = group->identity;
  result.warnings.emplace_back("ConSan barrier participant mutation rewrote one proven static "
                               "lifecycle count from " +
                               std::to_string(*init->participant_count) + " to " +
                               std::to_string(*options.fault_barrier_target_participant_count));
}

void try_apply_barrier_move_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                        const ConSanOptions &options, ConSanResult &result) {
  if (has_only_rocclr_runtime_kernels(result)) {
    result.warnings.emplace_back(
        "ConSan barrier move fault skipped ROCclr runtime helper code object");
    return;
  }
  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  std::optional<BarrierSite> site;
  const ConSanFaultSite *selected_inventory_site = select_fault_site_for_plan(
      result, options, ConSanFaultSiteKind::Barrier, options.fault_barrier_index);
  if (selected_inventory_site != nullptr)
    site = barrier_site_from_inventory(*selected_inventory_site);
  if (!site) {
    if (!options.fault_site_identity.empty()) {
      result.warnings.emplace_back("ConSan barrier move fault found no site with exact identity '" +
                                   options.fault_site_identity + "'");
    } else {
      result.warnings.emplace_back(
          "ConSan barrier move fault found no patchable barrier at index " +
          std::to_string(options.fault_barrier_index));
    }
    return;
  }
  std::optional<BarrierSite> second_site;
  const ConSanSyncSequence *selected_sequence = nullptr;
  if (selected_inventory_site != nullptr && selected_inventory_site->sync_event_identity) {
    selected_sequence =
        find_consan_sync_sequence_for_event(result, *selected_inventory_site->sync_event_identity);
    if (selected_sequence == nullptr ||
        !owned_by_requested_kernel(selected_sequence->execution_owners, result, options) ||
        selected_sequence->operation != ConSanSyncOperation::BarrierFull ||
        !consan_sync_confidence_meets(selected_sequence->confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        selected_sequence->member_event_identities.size() != 2u) {
      result.warnings.emplace_back(
          "ConSan barrier move fault selected no conservative logical barrier pair");
      return;
    }
    const auto next =
        std::ranges::find_if(result.fault_sites, [&](const ConSanFaultSite &candidate) {
          return candidate.kind == ConSanFaultSiteKind::Barrier && candidate.sync_event_identity &&
                 owned_by_requested_kernel(candidate.execution_owners, result, options) &&
                 candidate.identity != selected_inventory_site->identity &&
                 std::ranges::find(selected_sequence->member_event_identities,
                                   *candidate.sync_event_identity) !=
                     selected_sequence->member_event_identities.end();
        });
    if (next != result.fault_sites.end())
      second_site = barrier_site_from_inventory(*next);
  }
  if (!second_site) {
    result.warnings.emplace_back("ConSan barrier move fault found no second barrier instruction");
    return;
  }

  if (options.fault_barrier_move_direction != ConSanBarrierMoveDirection::LegacyMarker) {
    if (selected_sequence == nullptr || options.fault_barrier_destination_identity.empty()) {
      result.warnings.emplace_back(
          "ConSan markerless barrier move requires an exact destination identity");
      return;
    }
    const auto destination = std::ranges::find(result.barrier_move_destinations,
                                               options.fault_barrier_destination_identity,
                                               &ConSanBarrierMoveDestination::identity);
    if (destination == result.barrier_move_destinations.end()) {
      result.warnings.emplace_back(
          "ConSan barrier move fault found no destination with exact identity '" +
          options.fault_barrier_destination_identity + "'");
      return;
    }
    if (!owned_by_requested_kernel(destination->execution_owners, result, options)) {
      result.warnings.emplace_back(
          "ConSan barrier move fault rejected destination outside the requested dispatch owner");
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
    if (cross_block && !admits_cross_block_barrier_move(*destination, options)) {
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
    if (!destination->suitable) {
      result.warnings.emplace_back("ConSan barrier move fault rejected unsuitable destination: " +
                                   destination->rejection_reason);
      return;
    }
    if (options.fault_barrier_move_direction == ConSanBarrierMoveDirection::Earlier &&
        destination->text_offset + destination->size > selected_sequence->begin_text_offset) {
      result.warnings.emplace_back(
          "ConSan barrier move fault rejected a destination that is not earlier than the "
          "selected logical barrier");
      return;
    }
    if (options.fault_barrier_move_direction == ConSanBarrierMoveDirection::Later &&
        destination->text_offset < selected_sequence->end_text_offset) {
      result.warnings.emplace_back(
          "ConSan barrier move fault rejected a destination that is not later than the selected "
          "logical barrier");
      return;
    }

    std::array<BarrierSite, 2> ordered_sources = {*site, *second_site};
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
    for (const LocalNopCave &candidate : find_uncovered_nop_caves(code_object, result, arch)) {
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
    if (options.fault_barrier_move_direction == ConSanBarrierMoveDirection::Earlier) {
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
      append_words(new_text, cave_words);
    } else {
      std::memcpy(new_text.data() + placement->body_offset, cave_words.data(),
                  cave_words.size() * sizeof(uint32_t));
    }
    if (!replace_consan_text(patcher, new_text, options, "markerless barrier move", result)) {
      return;
    }
    result.elf_bytes = std::move(patcher).emit();

    for (const BarrierSite &source : ordered_sources) {
      ConSanPatchInfo source_info;
      source_info.phase = ConSanPatchPhase::Mutation;
      source_info.kind = ConSanPatchKind::InlineBarrierMoveSourceRewrite;
      source_info.anchor_offset = source.text_offset;
      source_info.trampoline_offset = source.text_offset;
      source_info.original_size = source.size;
      source_info.barrier_move_direction = options.fault_barrier_move_direction;
      result.patches.push_back(source_info);
    }
    ConSanPatchInfo target_info;
    target_info.phase = ConSanPatchPhase::Mutation;
    target_info.kind = ConSanPatchKind::InlineBarrierMoveTargetRewrite;
    target_info.anchor_offset = destination->text_offset;
    target_info.trampoline_offset = placement->body_offset;
    target_info.original_size = destination->size;
    target_info.trampoline_size = static_cast<uint32_t>(cave_words.size() * sizeof(uint32_t));
    target_info.barrier_move_direction = options.fault_barrier_move_direction;
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
    result.modified = true;
    result.warnings.emplace_back(
        "ConSan barrier move fault relocated an adjacent logical pair " +
        std::string(options.fault_barrier_move_direction == ConSanBarrierMoveDirection::Earlier
                        ? "earlier"
                        : "later") +
        " at " + destination->mnemonic + " in " + destination->container_name);
    return;
  }

  const uint32_t marker = build_s_nop(options.fault_barrier_marker_imm, arch);
  const uint32_t second_marker = build_s_nop(options.fault_barrier_marker_imm + 1u, arch);
  std::optional<uint64_t> marker_file_offset;
  std::optional<uint64_t> marker_text_offset;
  std::optional<uint64_t> second_marker_file_offset;
  std::optional<uint64_t> second_marker_text_offset;
  for (const ConSanTextSection &section : result.text_sections) {
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
        std::to_string(options.fault_barrier_marker_imm));
    return;
  }
  if (*marker_file_offset <= second_site->file_offset ||
      *second_marker_file_offset != *marker_file_offset + sizeof(uint32_t)) {
    result.errors.emplace_back("ConSan barrier move fault marker must follow the selected barrier");
    return;
  }

  if (result.elf_bytes.empty())
    result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());
  uint32_t barrier_word = 0;
  uint32_t second_barrier_word = 0;
  std::memcpy(&barrier_word, original_bytes.data() + site->file_offset, sizeof(barrier_word));
  std::memcpy(&second_barrier_word, original_bytes.data() + second_site->file_offset,
              sizeof(second_barrier_word));
  const uint32_t nop = build_s_nop(0, arch);
  std::memcpy(result.elf_bytes.data() + site->file_offset, &nop, sizeof(nop));
  std::memcpy(result.elf_bytes.data() + second_site->file_offset, &nop, sizeof(nop));
  std::memcpy(result.elf_bytes.data() + *marker_file_offset, &barrier_word, sizeof(barrier_word));
  std::memcpy(result.elf_bytes.data() + *second_marker_file_offset, &second_barrier_word,
              sizeof(second_barrier_word));

  ConSanPatchInfo source_info;
  source_info.phase = ConSanPatchPhase::Mutation;
  source_info.kind = ConSanPatchKind::InlineBarrierMoveSourceRewrite;
  source_info.anchor_offset = site->text_offset;
  source_info.trampoline_offset = site->text_offset;
  source_info.original_size = site->size;
  result.patches.push_back(source_info);
  source_info.anchor_offset = second_site->text_offset;
  source_info.trampoline_offset = second_site->text_offset;
  source_info.original_size = second_site->size;
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
  result.modified = true;
  result.warnings.emplace_back("ConSan barrier move fault moved " + site->mnemonic + " in " +
                               site->range_name);
}

void try_apply_atomic_fault_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  const ConSanOptions &options, ConSanResult &result) {
  const ConSanFaultSite *selected = select_fault_site_for_plan(
      result, options, ConSanFaultSiteKind::Atomic, options.fault_atomic_index);
  if (!selected) {
    if (!options.fault_site_identity.empty()) {
      result.warnings.emplace_back("ConSan atomic fault found no site with exact identity '" +
                                   options.fault_site_identity + "'");
    } else {
      result.warnings.emplace_back(
          "ConSan atomic fault found no patchable flat/global atomic at index " +
          std::to_string(options.fault_atomic_index));
    }
    return;
  }

  const AtomicFaultEncoding encoding =
      atomic_fault_encoding(selected->mnemonic, selected->size, arch);
  if (options.fault_atomic_weaken_scope && !atomic_fault_supports_scope(encoding)) {
    result.errors.emplace_back(
        "ConSan atomic scope fault is unsupported for the selected atomic encoding");
    return;
  }
  if (options.fault_atomic_weaken_order && !atomic_fault_supports_order(encoding)) {
    result.errors.emplace_back(
        "ConSan atomic order fault is unsupported for the selected atomic encoding");
    return;
  }
  if (options.fault_atomic_wrong_address && !atomic_fault_supports_address(encoding)) {
    result.errors.emplace_back(
        "ConSan atomic address fault is unsupported for the selected atomic encoding");
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  if (result.elf_bytes.empty())
    result.elf_bytes.assign(original_bytes.begin(), original_bytes.end());
  std::array<uint32_t, 3> words{};
  std::memcpy(words.data(), result.elf_bytes.data() + selected->file_offset, selected->size);
  if (options.fault_atomic_wrong_address) {
    if (options.fault_atomic_address_delta == 0 ||
        options.fault_atomic_address_delta % sizeof(uint32_t) != 0 ||
        options.fault_atomic_address_delta > 0x7fffffu) {
      result.errors.emplace_back(
          "ConSan atomic address delta must be a positive aligned signed-24-bit offset");
      return;
    }
    if (encoding == AtomicFaultEncoding::Ds) {
      // RDNA4 VDS offset0 is the low byte of word 0. DS atomic addresses are
      // byte addressed, so retain the requested natural u32 alignment.
      const uint32_t replacement_offset = (words[0] & 0xffu) + options.fault_atomic_address_delta;
      if (replacement_offset > 0xffu) {
        result.errors.emplace_back("ConSan DS atomic address delta overflows 8-bit offset0");
        return;
      }
      const uint32_t natural_alignment =
          std::max<uint32_t>(sizeof(uint32_t), selected->width_bits / 8u);
      if (replacement_offset % natural_alignment != 0) {
        result.errors.emplace_back("ConSan DS atomic replacement offset is not naturally aligned");
        return;
      }
      words[0] = (words[0] & ~0xffu) | replacement_offset;
    } else {
      // The high 24 bits of word 2 are the signed byte ioffset on RDNA4 flat,
      // global, and buffer atomics.
      const int32_t original_offset = sign_extend_24(words[2] >> 8u);
      const int64_t replacement_offset =
          static_cast<int64_t>(original_offset) + options.fault_atomic_address_delta;
      if (replacement_offset > 0x7fffff) {
        result.errors.emplace_back("ConSan atomic address delta overflows signed-24-bit ioffset");
        return;
      }
      words[2] =
          (words[2] & 0xffu) | ((static_cast<uint32_t>(replacement_offset) & 0xffffffu) << 8u);
    }
    ConSanPatchInfo info;
    info.phase = ConSanPatchPhase::Mutation;
    info.kind = ConSanPatchKind::InlineAtomicAddressRewrite;
    info.anchor_offset = selected->text_offset;
    info.trampoline_offset = selected->text_offset;
    info.original_size = selected->size;
    result.patches.push_back(info);
  }
  if (options.fault_atomic_weaken_order) {
    const ConSanSyncSequence *sequence =
        selected->sync_event_identity
            ? find_consan_sync_sequence_for_event(result, *selected->sync_event_identity)
            : nullptr;
    const ConSanSyncEvent *fence = nullptr;
    std::optional<uint64_t> release_wait_text_offset;
    if (sequence != nullptr &&
        consan_sync_confidence_meets(sequence->memory_role_confidence,
                                     ConSanSemanticConfidence::Conservative)) {
      if (options.fault_atomic_order_edge != ConSanAtomicOrderEdge::Acquire)
        release_wait_text_offset = sequence->release_wait_text_offset;
      for (const std::string &identity : sequence->member_event_identities) {
        const auto event = std::ranges::find_if(result.sync_events, [&](const auto &candidate) {
          return candidate.identity == identity && candidate.kind == ConSanSyncEventKind::Fence;
        });
        const bool edge_matches =
            event != result.sync_events.end() &&
            (options.fault_atomic_order_edge == ConSanAtomicOrderEdge::Any ||
             (options.fault_atomic_order_edge == ConSanAtomicOrderEdge::Release &&
              is_release_cache_event(&*event)) ||
             (options.fault_atomic_order_edge == ConSanAtomicOrderEdge::Acquire &&
              is_acquire_cache_event(&*event)));
        if (edge_matches) {
          fence = &*event;
          break;
        }
      }
    }
    // A cache-associated release sequence may not record the immediately
    // preceding store wait as its primary proof, but that wait becomes an
    // independently sufficient release proof once the cache operation is
    // removed. Discover it directly so the staged reinventory observes the
    // complete weakened boundary rather than re-admitting the same atomic.
    if (encoding != AtomicFaultEncoding::CdnaFlat && !release_wait_text_offset &&
        options.fault_atomic_order_edge != ConSanAtomicOrderEdge::Acquire &&
        selected->text_offset >= sizeof(uint32_t) && selected->file_offset >= sizeof(uint32_t)) {
      const auto expected_combined = build_s_wait_storecnt_dscnt0(arch);
      const auto expected_store = build_s_wait_storecnt0(arch);
      uint32_t preceding_word = 0;
      std::memcpy(&preceding_word,
                  result.elf_bytes.data() + selected->file_offset - sizeof(uint32_t),
                  sizeof(preceding_word));
      if ((expected_combined && preceding_word == *expected_combined) ||
          (expected_store && preceding_word == *expected_store)) {
        release_wait_text_offset = selected->text_offset - sizeof(uint32_t);
      }
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
    std::optional<ConSanPatchInfo> release_wait_info;
    if (fence != nullptr) {
      if (fence->size == 0 || fence->size % sizeof(uint32_t) != 0) {
        result.errors.emplace_back(
            "ConSan atomic weaken-order mutation found malformed associated cache operation");
        return;
      }
      for (uint32_t offset = 0; offset < fence->size; offset += sizeof(uint32_t))
        std::memcpy(result.elf_bytes.data() + fence->file_offset + offset, &nop, sizeof(nop));
      info.anchor_offset = fence->text_offset;
      info.trampoline_offset = fence->text_offset;
      info.original_size = fence->size;
      std::string removed_boundary = fence->mnemonic;
      // RDNA4 release lowering commonly uses global_wb followed immediately
      // by s_wait_storecnt before the atomic. Removing only global_wb leaves
      // a boundary which reinventory independently and correctly admits as a
      // release, defeating the requested weaken-order mutation. Treat that
      // adjacent cache/wait pair as one logical ordering boundary.
      if (release_wait_text_offset) {
        const auto expected_combined = build_s_wait_storecnt_dscnt0(arch);
        const auto expected_store = build_s_wait_storecnt0(arch);
        if (!expected_combined || !expected_store ||
            *release_wait_text_offset >= selected->text_offset ||
            selected->file_offset < selected->text_offset - *release_wait_text_offset) {
          result.errors.emplace_back(
              "ConSan atomic weaken-order mutation found malformed associated release wait");
          return;
        }
        const uint64_t wait_file_offset =
            selected->file_offset - (selected->text_offset - *release_wait_text_offset);
        if (wait_file_offset > result.elf_bytes.size() ||
            sizeof(uint32_t) > result.elf_bytes.size() - wait_file_offset) {
          result.errors.emplace_back(
              "ConSan atomic weaken-order mutation release wait exceeds the code object");
          return;
        }
        uint32_t wait_word = 0;
        std::memcpy(&wait_word, result.elf_bytes.data() + wait_file_offset, sizeof(wait_word));
        if (wait_word != *expected_combined && wait_word != *expected_store) {
          result.errors.emplace_back(
              "ConSan atomic weaken-order release wait no longer has its admitted encoding");
          return;
        }
        std::memcpy(result.elf_bytes.data() + wait_file_offset, &nop, sizeof(nop));
        release_wait_info = info;
        release_wait_info->anchor_offset = *release_wait_text_offset;
        release_wait_info->trampoline_offset = *release_wait_text_offset;
        release_wait_info->original_size = sizeof(uint32_t);
        removed_boundary +=
            wait_word == *expected_combined ? "/s_wait_storecnt_dscnt" : "/s_wait_storecnt";
      }
      result.warnings.emplace_back("ConSan atomic fault removed associated " + removed_boundary +
                                   " while preserving " + selected->mnemonic);
    } else {
      const auto expected_combined = build_s_wait_storecnt_dscnt0(arch);
      const auto expected_store = build_s_wait_storecnt0(arch);
      if (!expected_combined || !expected_store ||
          *release_wait_text_offset >= selected->text_offset ||
          selected->file_offset < selected->text_offset - *release_wait_text_offset) {
        result.errors.emplace_back(
            "ConSan atomic weaken-order mutation found malformed associated release wait");
        return;
      }
      const uint64_t wait_file_offset =
          selected->file_offset - (selected->text_offset - *release_wait_text_offset);
      if (wait_file_offset > result.elf_bytes.size() ||
          sizeof(uint32_t) > result.elf_bytes.size() - wait_file_offset) {
        result.errors.emplace_back(
            "ConSan atomic weaken-order mutation release wait exceeds the code object");
        return;
      }
      uint32_t wait_word = 0;
      std::memcpy(&wait_word, result.elf_bytes.data() + wait_file_offset, sizeof(wait_word));
      if (wait_word != *expected_combined && wait_word != *expected_store) {
        result.errors.emplace_back(
            "ConSan atomic weaken-order release wait no longer has its admitted encoding");
        return;
      }
      std::memcpy(result.elf_bytes.data() + wait_file_offset, &nop, sizeof(nop));
      info.anchor_offset = *release_wait_text_offset;
      info.trampoline_offset = *release_wait_text_offset;
      info.original_size = sizeof(uint32_t);
      const std::string_view wait_name =
          wait_word == *expected_combined ? "s_wait_storecnt_dscnt" : "s_wait_storecnt";
      result.warnings.emplace_back("ConSan atomic fault removed associated " +
                                   std::string(wait_name) + " while preserving " +
                                   selected->mnemonic);
    }
    result.patches.push_back(info);
    if (release_wait_info)
      result.patches.push_back(*release_wait_info);
  }
  if (options.fault_atomic_weaken_scope) {
    const uint32_t original_scope = (words[1] >> 18u) & 0x3u;
    if (original_scope == 0) {
      result.errors.emplace_back("ConSan atomic scope fault cannot weaken wave scope further");
      return;
    }
    words[1] &= ~(0x3u << 18u);
    ConSanPatchInfo info;
    info.phase = ConSanPatchPhase::Mutation;
    info.kind = ConSanPatchKind::InlineAtomicScopeRewrite;
    info.anchor_offset = selected->text_offset;
    info.trampoline_offset = selected->text_offset;
    info.original_size = selected->size;
    result.patches.push_back(info);
    result.warnings.emplace_back("ConSan atomic fault lowered " + selected->mnemonic +
                                 " scope from " + std::to_string(original_scope) + " to wave");
  }
  std::memcpy(result.elf_bytes.data() + selected->file_offset, words.data(), selected->size);
  result.modified = true;
  if (options.fault_atomic_wrong_address) {
    result.warnings.emplace_back("ConSan atomic fault rewrote " + selected->mnemonic +
                                 " with validated address delta " +
                                 std::to_string(options.fault_atomic_address_delta));
  }
}

void try_apply_ordinary_fault_patch(const AmdGpuCodeObject &code_object,
                                    const ConSanOptions &options, ConSanResult &result) {
  const auto target = select_ordinary_acquire_mutation_target(result, options);
  if (!target) {
    if (!options.fault_site_identity.empty()) {
      result.warnings.emplace_back(
          "ConSan ordinary fault found no exact supported acquire-load site with identity '" +
          options.fault_site_identity + "'");
    } else {
      result.warnings.emplace_back(
          "ConSan ordinary fault found no exact supported acquire-load at index " +
          std::to_string(options.fault_ordinary_index));
    }
    return;
  }

  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());
  std::vector<uint8_t> candidate =
      result.elf_bytes.empty() ? std::vector<uint8_t>(original_bytes.begin(), original_bytes.end())
                               : result.elf_bytes;
  if (target->site->size != sizeof(rdna4::VglobalMachineInst) ||
      target->site->file_offset > candidate.size() ||
      target->site->size > candidate.size() - target->site->file_offset ||
      target->cache->file_offset > candidate.size() ||
      target->cache->size > candidate.size() - target->cache->file_offset) {
    result.errors.emplace_back("ConSan ordinary fault found an invalid exact mutation range");
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

  if (options.fault_ordinary_wrong_address) {
    if (options.fault_ordinary_address_delta == 0u ||
        options.fault_ordinary_address_delta % sizeof(uint32_t) != 0u ||
        options.fault_ordinary_address_delta > 0x7fffffu) {
      result.errors.emplace_back(
          "ConSan ordinary address delta must be a positive aligned signed-24-bit offset");
      return;
    }
    rdna4::VglobalMachineInst raw{};
    std::memcpy(&raw, candidate.data() + target->site->file_offset, sizeof(raw));
    const int32_t original_offset = sign_extend_24(static_cast<uint32_t>(raw.ioffset));
    if (!target->load->static_byte_offset || *target->load->static_byte_offset != original_offset ||
        target->load->address_source != ConSanSyncAddressSource::GlobalScalarVector ||
        target->site->mnemonic != "global_load_b32" || target->site->width_bits != 32u) {
      result.errors.emplace_back(
          "ConSan ordinary address fault found stale or unsupported decoded load fields");
      return;
    }
    const int64_t replacement_offset =
        static_cast<int64_t>(original_offset) + options.fault_ordinary_address_delta;
    if (replacement_offset < -0x800000ll || replacement_offset > 0x7fffffll ||
        replacement_offset % static_cast<int64_t>(sizeof(uint32_t)) != 0) {
      result.errors.emplace_back(
          "ConSan ordinary address delta produces an invalid signed-24-bit aligned ioffset");
      return;
    }
    raw.ioffset = static_cast<uint32_t>(replacement_offset) & 0xffffffu;
    std::memcpy(candidate.data() + target->site->file_offset, &raw, sizeof(raw));
    patches.push_back(make_info(ConSanPatchKind::InlineOrdinaryAddressRewrite,
                                target->site->text_offset, target->site->size, {}));
  }

  if (options.fault_ordinary_weaken_order) {
    const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_RDNA4);
    for (uint32_t offset = 0; offset < target->cache->size; offset += sizeof(uint32_t))
      std::memcpy(candidate.data() + target->cache->file_offset + offset, &nop, sizeof(nop));
    patches.push_back(make_info(ConSanPatchKind::InlineOrdinaryOrderRewrite,
                                target->cache->text_offset, target->cache->size,
                                target->cache->identity));
  }
  if (options.fault_ordinary_weaken_scope) {
    rdna4::VglobalMachineInst raw{};
    std::memcpy(&raw, candidate.data() + target->site->file_offset, sizeof(raw));
    if (!target->load->raw_scope || raw.scope != *target->load->raw_scope ||
        (raw.scope != 2u && raw.scope != 3u)) {
      result.errors.emplace_back(
          "ConSan ordinary scope fault found stale or already-wave decoded scope bits");
      return;
    }
    raw.scope = 0u;
    std::memcpy(candidate.data() + target->site->file_offset, &raw, sizeof(raw));
    patches.push_back(make_info(ConSanPatchKind::InlineOrdinaryScopeRewrite,
                                target->site->text_offset, target->site->size, {}));
  }

  result.elf_bytes = std::move(candidate);
  result.patches.insert(result.patches.end(), std::make_move_iterator(patches.begin()),
                        std::make_move_iterator(patches.end()));
  result.applied_fault_logical_identity = target->sequence->identity;
  result.modified = true;
  if (options.fault_ordinary_wrong_address) {
    result.warnings.emplace_back("ConSan ordinary fault shifted global_load_b32 by validated " +
                                 std::to_string(options.fault_ordinary_address_delta) + " byte(s)");
  }
  if (options.fault_ordinary_weaken_order) {
    result.warnings.emplace_back(
        "ConSan ordinary fault removed associated global_inv while preserving the acquire load "
        "and waits");
  }
  if (options.fault_ordinary_weaken_scope) {
    result.warnings.emplace_back("ConSan ordinary fault lowered global_load_b32 scope to wave");
  }
}

void try_apply_proof_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                  ConSanResult &result) {
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

[[nodiscard]] const ConSanLdsSite *
find_first_supported_lds_read_b32_site(const ConSanResult &result) {
  const ConSanLdsSite *best = nullptr;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.preflight_action == ConSanPreflightAction::Reject)
      continue;
    for (const ConSanLdsSite &site : kernel.lds_sites) {
      if (site.kind == ConSanLdsAccessKind::Read && site.supported_mvp && site.width_bits == 32 &&
          (best == nullptr || site.file_offset < best->file_offset))
        best = &site;
    }
  }
  return best;
}

void try_apply_lds_endpgm_patch(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                ConSanResult &result) {
  const ConSanLdsSite *site = find_first_supported_lds_read_b32_site(result);
  if (site == nullptr) {
    result.warnings.emplace_back("ConSan LDS endpgm proof found no supported ds_read_b32 site");
    return;
  }

  const uint32_t replacement_endpgm = build_s_endpgm(arch);
  if (!rewrite_word_in_place(code_object, site->file_offset, replacement_endpgm, result))
    return;

  ConSanPatchInfo info;
  info.kind = ConSanPatchKind::InlineLdsEndpgmRewrite;
  info.anchor_offset = site->text_offset;
  info.trampoline_offset = site->text_offset;
  info.original_size = site->size;
  result.patches.push_back(info);
}

[[nodiscard]] bool is_supported_check_trap_site(const ConSanLdsSite &site, rj_code_arch_t arch) {
  if (site.size != 2u * sizeof(uint32_t) || !site.addr_vgpr)
    return false;
  auto dword_count = lds_dword_count(site);
  if (!dword_count)
    return false;
  if (site.kind == ConSanLdsAccessKind::Read) {
    if (site.mnemonic != "ds_load_i8" && site.mnemonic != "ds_load_u8" &&
        site.mnemonic != "ds_load_i16" && site.mnemonic != "ds_load_b32" &&
        site.mnemonic != "ds_load_b64" && site.mnemonic != "ds_load_b96" &&
        site.mnemonic != "ds_load_b128" && site.mnemonic != "ds_load_tr8_b64" &&
        site.mnemonic != "ds_load_tr16_b128" && site.mnemonic != "ds_load_2addr_b32" &&
        site.mnemonic != "ds_load_2addr_b64" && site.mnemonic != "ds_load_2addr_stride64_b32" &&
        site.mnemonic != "ds_load_2addr_stride64_b64" && site.mnemonic != "ds_load_u16" &&
        site.mnemonic != "ds_load_u8_d16" && site.mnemonic != "ds_load_u8_d16_hi" &&
        site.mnemonic != "ds_load_i8_d16" && site.mnemonic != "ds_load_i8_d16_hi" &&
        site.mnemonic != "ds_load_u16_d16" && site.mnemonic != "ds_load_u16_d16_hi" &&
        site.mnemonic != "ds_read_u8_d16" && site.mnemonic != "ds_read_u8_d16_hi" &&
        site.mnemonic != "ds_read_i8_d16" && site.mnemonic != "ds_read_i8_d16_hi" &&
        site.mnemonic != "ds_read_u16_d16" && site.mnemonic != "ds_read_u16_d16_hi" &&
        site.mnemonic != "ds_read_b32" && site.mnemonic != "ds_read_b64" &&
        site.mnemonic != "ds_read_b128" && site.mnemonic != "ds_read_i16" &&
        site.mnemonic != "ds_read_u16" && site.mnemonic != "ds_read_b64_tr_b16" &&
        site.mnemonic != "ds_read2_b32" && site.mnemonic != "ds_read2st64_b32" &&
        site.mnemonic != "ds_read2_b64" && site.mnemonic != "ds_read2st64_b64")
      return false;
    if (site.dst_accvgpr) {
      return (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
             static_cast<uint32_t>(*site.dst_accvgpr) + *dword_count <= 256u;
    }
    const uint32_t register_limit = arch == ROCJITSU_CODE_ARCH_GFX1250 ? 1024u : 256u;
    return site.dst_vgpr.has_value() &&
           static_cast<uint32_t>(*site.dst_vgpr) + *dword_count <= register_limit;
  }
  if (site.kind == ConSanLdsAccessKind::Write) {
    if (site.mnemonic != "ds_store_b8" && site.mnemonic != "ds_store_b8_d16_hi" &&
        site.mnemonic != "ds_store_b16" && site.mnemonic != "ds_store_b16_d16_hi" &&
        site.mnemonic != "ds_store_b32" && site.mnemonic != "ds_store_b64" &&
        site.mnemonic != "ds_store_b96" && site.mnemonic != "ds_store_b128" &&
        site.mnemonic != "ds_store_2addr_b32" && site.mnemonic != "ds_store_2addr_b64" &&
        site.mnemonic != "ds_store_2addr_stride64_b32" &&
        site.mnemonic != "ds_store_2addr_stride64_b64" && site.mnemonic != "ds_write_b32" &&
        site.mnemonic != "ds_write_b64" && site.mnemonic != "ds_write_b128" &&
        site.mnemonic != "ds_write2_b32" && site.mnemonic != "ds_write2st64_b32" &&
        site.mnemonic != "ds_write2_b64" && site.mnemonic != "ds_write2st64_b64")
      return false;
    const bool two_address = is_two_address_lds_store(site);
    if (two_address) {
      const uint32_t dwords_per_address =
          two_address_lds_element_width_bits(site).value_or(0u) / 32u;
      return site.data_vgpr.has_value() && site.second_data_vgpr.has_value() &&
             static_cast<uint32_t>(*site.data_vgpr) + dwords_per_address <= 256 &&
             static_cast<uint32_t>(*site.second_data_vgpr) + dwords_per_address <= 256;
    }
    return site.data_vgpr.has_value() &&
           static_cast<uint32_t>(*site.data_vgpr) + *dword_count <= 256;
  }
  return false;
}

[[nodiscard]] bool is_instrumentable_group_flat_hint(ConSanFlatAddressSpaceHint hint,
                                                     ConSanFlatProvenanceMode mode) {
  return hint == ConSanFlatAddressSpaceHint::Group ||
         (mode == ConSanFlatProvenanceMode::Likely &&
          hint == ConSanFlatAddressSpaceHint::MaybeGroup);
}

[[nodiscard]] bool is_supported_flat_trap_site(const ConSanFlatSite &site,
                                               ConSanFlatProvenanceMode mode) {
  if (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write)
    return false;
  if (site.size != 3u * sizeof(uint32_t))
    return false;
  return is_instrumentable_group_flat_hint(site.address_space_hint, mode);
}

[[nodiscard]] std::optional<uint16_t> flat_dword_count(const ConSanFlatSite &site) {
  if (consan_flat_load_subword_semantics(site.mnemonic) ||
      consan_flat_store_subword_semantics(site.mnemonic))
    return 1u;
  if (site.width_bits == 16)
    return 1u;
  if (site.width_bits == 32 || site.width_bits == 64 || site.width_bits == 128)
    return static_cast<uint16_t>(site.width_bits / 32u);
  return std::nullopt;
}

[[nodiscard]] bool flat_scratch_tuple_base_is_valid(const ConSanFlatSite &site,
                                                    rj_code_arch_t arch, uint16_t candidate) {
  const auto dword_count = flat_dword_count(site);
  return !dword_count || *dword_count <= 1u ||
         !consan_arch_requires_even_vgpr_tuples(arch) || candidate % 2u == 0u;
}

[[nodiscard]] bool is_supported_flat_check_trap_site(const ConSanFlatSite &site,
                                                     ConSanFlatProvenanceMode mode) {
  if (site.kind != ConSanLdsAccessKind::Read && site.kind != ConSanLdsAccessKind::Write)
    return false;
  if ((site.size != 2u * sizeof(uint32_t) && site.size != 3u * sizeof(uint32_t)) || !site.addr_vgpr)
    return false;
  auto dword_count = flat_dword_count(site);
  if (!dword_count)
    return false;
  if (site.kind == ConSanLdsAccessKind::Read) {
    if (!consan_flat_load_subword_semantics(site.mnemonic) && site.mnemonic != "flat_load_b32" &&
        site.mnemonic != "flat_load_b64" && site.mnemonic != "flat_load_b128" &&
        site.mnemonic != "flat_load_dword" && site.mnemonic != "flat_load_dwordx2" &&
        site.mnemonic != "flat_load_dwordx4" && site.mnemonic != "flat_load_ushort" &&
        site.mnemonic != "flat_load_u16")
      return false;
    if (!site.dst_vgpr || static_cast<uint32_t>(*site.dst_vgpr) + *dword_count > 256)
      return false;
  } else {
    if (!consan_flat_store_subword_semantics(site.mnemonic) && site.mnemonic != "flat_store_b32" &&
        site.mnemonic != "flat_store_b64" && site.mnemonic != "flat_store_b128" &&
        site.mnemonic != "flat_store_dword" && site.mnemonic != "flat_store_dwordx2" &&
        site.mnemonic != "flat_store_dwordx4" && site.mnemonic != "flat_store_short" &&
        site.mnemonic != "flat_store_b16")
      return false;
    if (!site.data_vgpr || static_cast<uint32_t>(*site.data_vgpr) + *dword_count > 256)
      return false;
  }
  return is_instrumentable_group_flat_hint(site.address_space_hint, mode);
}

[[nodiscard]] bool is_forbidden_flat_scratch_vgpr_run(const ConSanFlatSite &site,
                                                      uint16_t candidate, uint16_t required_vgprs) {
  if (site.addr_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.addr_vgpr, 2))
    return true;

  const uint16_t data_vgprs = flat_dword_count(site).value_or(1);
  if (site.dst_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.dst_vgpr, data_vgprs))
    return true;
  if (site.data_vgpr && vgpr_ranges_overlap(candidate, required_vgprs, *site.data_vgpr, data_vgprs))
    return true;
  return false;
}

[[nodiscard]] uint16_t flat_scratch_search_start(const ConSanFlatSite &site) {
  std::optional<uint32_t> first_after_operands;
  auto note_range = [&first_after_operands](std::optional<uint16_t> base, uint16_t count) {
    if (!base)
      return;
    const uint32_t end = static_cast<uint32_t>(*base) + count;
    if (!first_after_operands || end > *first_after_operands)
      first_after_operands = end;
  };

  const uint16_t data_vgprs = flat_dword_count(site).value_or(1);
  note_range(site.addr_vgpr, 2);
  note_range(site.dst_vgpr, data_vgprs);
  note_range(site.data_vgpr, data_vgprs);
  if (!first_after_operands)
    return 0;
  return *first_after_operands <= 255 ? static_cast<uint16_t>(*first_after_operands) : 256;
}

[[nodiscard]] std::optional<uint16_t>
choose_flat_scratch_vgpr(const ConSanFlatSite &site, const ConSanOptions &options,
                         const Instruction *inst, const LivenessAnalysis *liveness,
                         std::optional<uint16_t> min_auto_scratch_vgpr,
                         std::optional<uint16_t> max_auto_scratch_vgpr, uint16_t required_vgprs,
                         rj_code_arch_t arch) {
  if (options.scratch_vgpr) {
    if (static_cast<uint32_t>(*options.scratch_vgpr) + required_vgprs <= 256 &&
        flat_scratch_tuple_base_is_valid(site, arch, *options.scratch_vgpr) &&
        !is_forbidden_flat_scratch_vgpr_run(site, *options.scratch_vgpr, required_vgprs))
      return *options.scratch_vgpr;
    return std::nullopt;
  }

  if (inst == nullptr || liveness == nullptr)
    return std::nullopt;

  uint16_t search_start =
      std::max(flat_scratch_search_start(site), min_auto_scratch_vgpr.value_or(0));
  while (search_start < REGISTER_SET_MAX_VGPRS) {
    auto candidate = liveness->find_free_run(inst, required_vgprs, search_start);
    if (!candidate)
      return std::nullopt;
    const uint32_t candidate_end = static_cast<uint32_t>(*candidate) + required_vgprs;
    if (max_auto_scratch_vgpr && candidate_end > *max_auto_scratch_vgpr)
      return std::nullopt;
    if (candidate_end <= REGISTER_SET_MAX_VGPRS &&
        flat_scratch_tuple_base_is_valid(site, arch, *candidate) &&
        !is_forbidden_flat_scratch_vgpr_run(site, *candidate, required_vgprs))
      return candidate;
    if (*candidate == UINT16_MAX)
      return std::nullopt;
    search_start = static_cast<uint16_t>(*candidate + 1u);
  }
  return std::nullopt;
}

/// Select the lowest allocated flat scratch window that does not overlap the
/// guest access operands. Liveness is intentionally irrelevant here: the
/// caller saves and restores the complete selected window through private
/// memory before using it.
[[nodiscard]] std::optional<uint16_t>
choose_flat_spill_scratch_vgpr(const ConSanFlatSite &site, uint16_t allocation_count,
                               uint16_t required_vgprs, rj_code_arch_t arch) {
  for (uint32_t candidate = 0;
       candidate + required_vgprs <= allocation_count &&
       candidate + required_vgprs <= REGISTER_SET_MAX_VGPRS;
       ++candidate) {
    if (flat_scratch_tuple_base_is_valid(site, arch, static_cast<uint16_t>(candidate)) &&
        !is_forbidden_flat_scratch_vgpr_run(site, static_cast<uint16_t>(candidate),
                                            required_vgprs)) {
      return static_cast<uint16_t>(candidate);
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
retarget_flat_load_vdst(std::array<uint32_t, 3> words, uint16_t vdst, rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
    words[1] = (words[1] & 0x00FFFFFFu) | (static_cast<uint32_t>(vdst) << 24u);
    return words;
  }
  if (!is_rdna4_family_arch(arch))
    return std::nullopt;
  if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
    words[1] = (words[1] & ~0xffu) | vdst;
    return words;
  }
  rdna4::VflatMachineInst inst{};
  std::memcpy(&inst, words.data(), sizeof(inst));
  inst.vdst = vdst;
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

[[nodiscard]] std::optional<uint8_t> flat_load_op_for_width(uint32_t width_bits) {
  switch (width_bits) {
  case 8:
    return rdna4::kFlatLoadU8Vflat;
  case 16:
    return rdna4::kFlatLoadU16Vflat;
  case 32:
    return rdna4::kFlatLoadB32Vflat;
  case 64:
    return rdna4::kFlatLoadB64Vflat;
  case 128:
    return rdna4::kFlatLoadB128Vflat;
  default:
    return std::nullopt;
  }
}

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_flat_load_from_flat_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst) {
  auto load_op = flat_load_op_for_width(width_bits);
  if (!load_op)
    return std::nullopt;
  rdna4::VflatMachineInst inst{};
  std::memcpy(&inst, words.data(), sizeof(inst));
  inst.op = *load_op;
  inst.vdst = vdst;
  inst.vsrc = 0;
  std::memcpy(words.data(), &inst, sizeof(inst));
  return words;
}

[[nodiscard]] std::optional<std::array<uint32_t, 3>>
build_flat_load_from_flat_store(std::array<uint32_t, 3> words, uint32_t width_bits, uint16_t vdst,
                                rj_code_arch_t arch) {
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    return build_flat_load_from_flat_store(words, width_bits, vdst);
  if (arch == ROCJITSU_CODE_ARCH_GFX1250) {
    uint16_t load_op = 0;
    switch (width_bits) {
    case 8:
      load_op = gfx1250::kFlatLoadU8Vflat;
      break;
    case 16:
      load_op = gfx1250::kFlatLoadU16Vflat;
      break;
    case 32:
      load_op = gfx1250::kFlatLoadB32Vflat;
      break;
    case 64:
      load_op = gfx1250::kFlatLoadB64Vflat;
      break;
    case 128:
      load_op = gfx1250::kFlatLoadB128Vflat;
      break;
    default:
      return std::nullopt;
    }
    constexpr uint32_t kGfx1250FlatOpMask = 0xffu << 14u;
    constexpr uint32_t kGfx1250FlatVsrcMask = 0xffu << 23u;
    words[0] = (words[0] & ~kGfx1250FlatOpMask) | (static_cast<uint32_t>(load_op) << 14u);
    words[1] = (words[1] & ~0xffu & ~kGfx1250FlatVsrcMask) | vdst;
    return words;
  }
  if (arch != ROCJITSU_CODE_ARCH_CDNA3 && arch != ROCJITSU_CODE_ARCH_CDNA4)
    return std::nullopt;
  uint32_t load_op = 0;
  switch (width_bits) {
  case 8:
    load_op = cdna4::kFlatLoadUbyteFlat;
    break;
  case 16:
    load_op = cdna4::kFlatLoadUshortFlat;
    break;
  case 32:
    load_op = cdna4::kFlatLoadDwordFlat;
    break;
  case 64:
    load_op = cdna4::kFlatLoadDwordx2Flat;
    break;
  case 128:
    load_op = cdna4::kFlatLoadDwordx4Flat;
    break;
  default:
    return std::nullopt;
  }
  constexpr uint32_t kCdnaFlatOpMask = 0x7fu << 18u;
  words[0] = (words[0] & ~kCdnaFlatOpMask) | (load_op << 18u);
  constexpr uint32_t kCdnaFlatDataMask = 0xffu << 8u;
  words[1] = (words[1] & ~kCdnaFlatDataMask & 0x00FFFFFFu) | (static_cast<uint32_t>(vdst) << 24u);
  return words;
}

[[nodiscard]] std::optional<uint16_t> flat_check_trap_compare_vgpr(const ConSanFlatSite &site) {
  if (site.kind == ConSanLdsAccessKind::Read)
    return site.dst_vgpr;
  if (site.kind == ConSanLdsAccessKind::Write)
    return site.data_vgpr;
  return std::nullopt;
}

[[nodiscard]] std::optional<uint16_t> check_trap_compare_vgpr(const ConSanLdsSite &site,
                                                              uint16_t chunk_index,
                                                              rj_code_arch_t arch,
                                                              uint16_t gfx1250_vgpr_msb_mode) {
  if (site.kind == ConSanLdsAccessKind::Read) {
    const uint32_t dst_bank = arch == ROCJITSU_CODE_ARCH_GFX1250
                                  ? static_cast<uint32_t>((gfx1250_vgpr_msb_mode >> 6u) & 0x3u)
                                  : 0u;
    const uint32_t physical_vgpr = dst_bank * 256u + site.dst_vgpr.value_or(0u) + chunk_index;
    const uint32_t register_limit = arch == ROCJITSU_CODE_ARCH_GFX1250 ? 1024u : 256u;
    return site.dst_vgpr && physical_vgpr < register_limit
               ? std::optional<uint16_t>(static_cast<uint16_t>(physical_vgpr))
               : std::nullopt;
  }
  if (site.kind == ConSanLdsAccessKind::Write) {
    const bool two_address = is_two_address_lds_store(site);
    const uint16_t dwords_per_address = static_cast<uint16_t>(
        two_address ? two_address_lds_element_width_bits(site).value_or(0u) / 32u
                    : site.width_bits / 32u);
    const std::optional<uint16_t> base =
        two_address && chunk_index >= dwords_per_address ? site.second_data_vgpr : site.data_vgpr;
    const uint16_t relative_index = two_address && chunk_index >= dwords_per_address
                                        ? static_cast<uint16_t>(chunk_index - dwords_per_address)
                                        : chunk_index;
    return base && static_cast<uint32_t>(*base) + relative_index < 256
               ? std::optional<uint16_t>(static_cast<uint16_t>(*base + relative_index))
               : std::nullopt;
  }
  return std::nullopt;
}
