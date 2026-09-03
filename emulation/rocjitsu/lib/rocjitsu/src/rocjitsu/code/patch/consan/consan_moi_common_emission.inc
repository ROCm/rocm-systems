// Retained intentionally for the structural dense-routing template; compiled
// shared lowering lives in consan_moi_shared_lowering.cpp.

std::optional<std::vector<uint32_t>> build_moi_dense_relay_dispatcher(
    std::span<const MoiDenseRelayRoute> routes, uint16_t jump_pc_sgpr,
    uint16_t saved_scc_sgpr, uint16_t key_sgpr, uint16_t call_return_sgpr,
    bool derive_key_at_entry, bool key_encodes_scc, bool match_call_return,
    bool clone_local_routes, bool explicit_key, bool normalize_encoded_scc,
    uint64_t dispatcher_offset, uint64_t island_offset, const ConSanTargetProfile &target,
    rj_code_arch_t arch) {
  std::vector<uint32_t> words;
  InstructionSequence sequence(words);
  if (key_encodes_scc) {
    words.push_back(build_s_lshr_b32(jump_pc_sgpr, key_sgpr,
                                     scalar_positive_inline_u32(1), arch));
  }
  if (!derive_key_at_entry && !match_call_return && !clone_local_routes && !explicit_key) {
    if (!sequence.emit(
            instrumentation::build_s_sub_u32(key_sgpr, call_return_sgpr, jump_pc_sgpr, arch)))
      return std::nullopt;
    words.push_back(build_s_add_u32(key_sgpr, key_sgpr, /*literal=*/255u, arch));
    words.push_back(static_cast<uint32_t>(dispatcher_offset - island_offset));
  }
  for (const MoiDenseRelayRoute &route : routes) {
    if (explicit_key || clone_local_routes) {
      if (!sequence.emit(instrumentation::build_s_cmp_eq_u32(
              key_encodes_scc ? jump_pc_sgpr : key_sgpr,
              scalar_positive_inline_u32(static_cast<uint16_t>(route.identity)), arch)))
        return std::nullopt;
    } else if (match_call_return) {
      if (!append_moi_call_return_match(words, dispatcher_offset, route.caller_return,
                                        jump_pc_sgpr, call_return_sgpr, arch))
        return std::nullopt;
    } else {
      if (!sequence.emit(instrumentation::build_s_cmp_eq_u32(
              key_encodes_scc ? jump_pc_sgpr : key_sgpr, /*literal=*/255u, arch)))
        return std::nullopt;
      words.push_back(static_cast<uint32_t>(route.caller_return - island_offset) /
                      (key_encodes_scc ? 2u : 1u));
    }
    std::vector<uint32_t> matched_words;
    if (key_encodes_scc &&
        !consan_detail::append_restore_moi_scc_from_route_key(
            matched_words,
            {.encoded_sgpr = key_sgpr,
             .normalize_encoded_sgpr = normalize_encoded_scc},
            target))
      return std::nullopt;
    const uint64_t matched_offset =
        dispatcher_offset + (words.size() + 1u) * sizeof(uint32_t);
    const uint64_t direct_branch_offset = matched_offset + matched_words.size() * sizeof(uint32_t);
    const auto direct = explicit_key
                            ? compute_sopp_branch_simm16(direct_branch_offset, route.target)
                            : std::nullopt;
    if (direct) {
      matched_words.push_back(build_s_branch(*direct, arch));
    } else if (!append_moi_scc_preserving_indirect_jump(
                   matched_words, matched_offset, route.target, jump_pc_sgpr, saved_scc_sgpr,
                   /*capture_scc=*/false, arch)) {
      return std::nullopt;
    } else if (explicit_key && key_encodes_scc) {
      matched_words[matched_words.size() - 2u] = matched_words.front();
    }
    if (!sequence.emit(instrumentation::build_s_cbranch_scc0(
            static_cast<int16_t>(matched_words.size()), arch)) ||
        !sequence.emit(matched_words))
      return std::nullopt;
  }
  words.push_back(build_s_endpgm(arch));
  return words;
}

uint64_t moi_dense_call_relay_island_words(size_t route_count, bool derive_key_at_entry,
                                           bool key_encodes_scc, bool clone_local_routes) {
  return clone_local_routes
             ? 2u * route_count + kMoiCompactIndirectIslandWords
             : moi_dense_entry_island_words(derive_key_at_entry, key_encodes_scc);
}

bool emit_moi_dense_relay(const MoiDenseRelayPlan &plan, rj_code_arch_t arch,
                          std::string_view diagnostic_subject, std::vector<uint8_t> &new_text,
                          std::vector<ConSanPatchInfo> &patches,
                          std::vector<std::string> &errors) {
  const auto fail = [&](std::string detail) {
    errors.emplace_back("ConSan MOI " + std::string(diagnostic_subject) + " " +
                        std::move(detail));
    return false;
  };
  if (plan.routes.empty() || plan.routes.front().anchor_size != 0u ||
      plan.host_body_offset != new_text.size() ||
      plan.island_offset != plan.host_offset + sizeof(uint32_t))
    return fail("received an incomplete dense-relay plan");
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return fail("has no target profile");

  const uint64_t island_words_count =
      plan.entry_island_words != 0u
          ? plan.entry_island_words
          : moi_dense_call_relay_island_words(plan.routes.size(), plan.derive_key_at_entry,
                                              plan.key_encodes_scc,
                                              plan.clone_local_routes);
  const uint64_t host_words_count = island_words_count + 1u;
  const uint64_t host_bytes = host_words_count * sizeof(uint32_t);
  if (plan.displaced_host_words.size() != host_words_count ||
      plan.host_offset > new_text.size() || host_bytes > new_text.size() - plan.host_offset)
    return fail("has stale relocated-host geometry");

  std::vector<uint32_t> host_words(plan.displaced_host_words.begin(),
                                   plan.displaced_host_words.end());
  if (!append_moi_scc_preserving_indirect_jump(
          host_words, plan.host_body_offset, plan.host_offset + host_bytes,
          plan.indirect_jump.pc_sgpr, plan.indirect_jump.scc_save_sgpr,
          /*capture_scc=*/true, arch))
    return fail("could not return from its relocated host");
  append_words_bytes(new_text, host_words);

  std::optional<std::vector<uint32_t>> island_words;
  if (plan.explicit_key) {
    std::vector<uint32_t> words;
    if (plan.key_encodes_scc)
      words.push_back(build_s_nop(0, arch));
    if (!append_moi_scc_preserving_indirect_jump(
            words, plan.island_offset, plan.dispatcher_offset, plan.indirect_jump.pc_sgpr,
            plan.indirect_jump.scc_save_sgpr,
            /*capture_scc=*/!plan.key_encodes_scc, arch))
      return fail("could not encode its keyed entry island");
    if (plan.requires_indirect_pc_wait) {
      const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
      if (!wait_pc || words.empty())
        return fail("could not encode its keyed entry dependency wait");
      words.insert(words.end() - 1u, *wait_pc);
    }
    island_words = std::move(words);
  } else if (plan.clone_local_routes) {
    std::vector<uint32_t> words;
    words.reserve(island_words_count);
    const uint64_t common_entry_offset =
        plan.island_offset + 2u * plan.routes.size() * sizeof(uint32_t);
    for (const MoiDenseRelayRoute &route : plan.routes) {
      words.push_back(build_s_mov_b32(
          plan.key_sgpr,
          scalar_positive_inline_u32(static_cast<uint16_t>(route.identity)), arch));
      const uint64_t branch_offset = plan.island_offset + words.size() * sizeof(uint32_t);
      const auto branch_delta = compute_sopp_branch_simm16(branch_offset, common_entry_offset);
      if (!branch_delta)
        return fail("could not encode a clone-local route");
      words.push_back(build_s_branch(*branch_delta, arch));
    }
    if (!append_moi_scc_preserving_indirect_jump(
            words, plan.island_offset, plan.dispatcher_offset, plan.indirect_jump.pc_sgpr,
            plan.indirect_jump.scc_save_sgpr, /*capture_scc=*/true, arch))
      return fail("could not encode its common clone-local entry");
    island_words = std::move(words);
  } else {
    island_words = build_moi_dense_barrier_entry_island(
        plan.island_offset, plan.dispatcher_offset,
        {.jump_pc_sgpr = plan.indirect_jump.pc_sgpr,
         .saved_scc_sgpr = plan.indirect_jump.scc_save_sgpr,
         .key_sgpr = plan.key_sgpr,
         .call_return_sgpr = plan.call_return_sgpr},
        plan.derive_key_at_entry, arch);
  }
  if (!island_words || island_words->size() != island_words_count)
    return fail("could not encode its entry island");
  std::memcpy(new_text.data() + plan.island_offset, island_words->data(),
              island_words->size() * sizeof(uint32_t));

  const auto build_call = [&](int16_t simm16) -> std::optional<uint32_t> {
    if (plan.call_form == ConSanDirectCallForm::SCallI64)
      return instrumentation::build_s_call_i64(plan.call_return_sgpr, simm16, arch);
    if (plan.call_form == ConSanDirectCallForm::SCallB64)
      return build_s_call_b64(plan.call_return_sgpr, simm16, arch);
    return std::nullopt;
  };
  std::optional<uint32_t> host_entry;
  if (plan.explicit_key) {
    host_entry =
        plan.key_encodes_scc
            ? instrumentation::build_s_cselect_b32(plan.key_sgpr,
                                                   scalar_positive_inline_u32(1u),
                                                   scalar_positive_inline_u32(0u), arch)
            : std::optional<uint32_t>(build_s_mov_b32(
                  plan.key_sgpr, scalar_positive_inline_u32(0u), arch));
  } else {
    host_entry = build_call(/*next instruction=*/0);
  }
  if (!host_entry)
    return fail("could not encode its relocated-host entry");
  std::memcpy(new_text.data() + plan.host_offset, &*host_entry, sizeof(uint32_t));

  const auto dispatcher_words = build_moi_dense_relay_dispatcher(
      plan.routes, plan.indirect_jump.pc_sgpr, plan.indirect_jump.scc_save_sgpr,
      plan.key_sgpr, plan.call_return_sgpr, plan.derive_key_at_entry, plan.key_encodes_scc,
      plan.match_call_return, plan.clone_local_routes, plan.explicit_key,
      plan.normalize_encoded_scc, plan.dispatcher_offset, plan.island_offset, *target, arch);
  if (!dispatcher_words ||
      dispatcher_words->size() * sizeof(uint32_t) > plan.dispatcher_capacity_bytes ||
      plan.dispatcher_offset > new_text.size() ||
      dispatcher_words->size() * sizeof(uint32_t) > new_text.size() - plan.dispatcher_offset)
    return fail("could not fit its dispatcher");
  std::memcpy(new_text.data() + plan.dispatcher_offset, dispatcher_words->data(),
              dispatcher_words->size() * sizeof(uint32_t));

  for (const MoiDenseRelayRoute &route : plan.routes) {
    if (route.anchor_size == 0u)
      continue;
    if (route.anchor > new_text.size() ||
        route.anchor_size > new_text.size() - route.anchor ||
        route.anchor_size < (plan.explicit_key ? 2u : 1u) * sizeof(uint32_t))
      return fail("could not encode a semantic route anchor");
    std::vector<uint32_t> anchor_words(route.anchor_size / sizeof(uint32_t),
                                       build_s_nop(0, arch));
    if (plan.explicit_key) {
      const uint16_t identity = static_cast<uint16_t>(route.identity);
      const uint16_t tagged_identity = static_cast<uint16_t>(2u * identity);
      const auto route_key =
          plan.key_encodes_scc
              ? instrumentation::build_s_cselect_b32(
                    plan.key_sgpr, scalar_positive_inline_u32(tagged_identity + 1u),
                    scalar_positive_inline_u32(tagged_identity), arch)
              : std::optional<uint32_t>(build_s_mov_b32(
                    plan.key_sgpr, scalar_positive_inline_u32(identity), arch));
      const auto branch_delta =
          compute_sopp_branch_simm16(route.anchor + sizeof(uint32_t), plan.island_offset);
      if (!route_key || !branch_delta)
        return fail("could not encode a keyed semantic route anchor");
      anchor_words[0] = *route_key;
      anchor_words[1] = build_s_branch(*branch_delta, arch);
    } else {
      const uint64_t call_target =
          plan.clone_local_routes
              ? plan.island_offset + 2u * route.identity * sizeof(uint32_t)
              : plan.island_offset;
      const auto call_delta = compute_sopp_branch_simm16(route.anchor, call_target);
      const auto call = call_delta ? build_call(*call_delta) : std::nullopt;
      if (!call)
        return fail("could not encode a semantic call anchor");
      anchor_words.front() = *call;
    }
    std::memcpy(new_text.data() + route.anchor, anchor_words.data(), route.anchor_size);
  }

  ConSanPatchInfo host;
  host.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  host.anchor_offset = plan.host_offset;
  host.original_size = static_cast<uint32_t>(host_bytes);
  host.trampoline_offset = plan.host_body_offset;
  host.trampoline_size = static_cast<uint32_t>(host_words.size() * sizeof(uint32_t));
  host.owner_descriptor_file_offsets.assign(plan.owner_descriptors.begin(),
                                            plan.owner_descriptors.end());
  patches.push_back(std::move(host));
  ConSanPatchInfo dispatcher;
  dispatcher.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  dispatcher.anchor_offset = plan.routes.size() > 1u ? plan.routes[1].anchor : plan.host_offset;
  dispatcher.trampoline_offset = plan.dispatcher_offset;
  dispatcher.trampoline_size =
      static_cast<uint32_t>(dispatcher_words->size() * sizeof(uint32_t));
  if (plan.explicit_key)
    dispatcher.moi_dense_entry_island_offset = plan.island_offset;
  dispatcher.owner_descriptor_file_offsets.assign(plan.owner_descriptors.begin(),
                                                  plan.owner_descriptors.end());
  patches.push_back(std::move(dispatcher));
  return true;
}

/// Emit the owner-local dense entry island, dispatcher, optional relocated
/// host entry, and original-site route anchors shared by all three access
/// engines.
///
/// `PlannedPatch` is deliberately structural: the engines retain distinct
/// evidence-body plans, but expose the same `candidate`, `resources`,
/// `placement`, and `entry_island_offset` facts to this routing mechanism.
/// The selected mode publishes its complete router mechanics before this
/// boundary. Evidence construction and per-site resource admission remain
/// outside this function.
[[nodiscard]] bool emit_moi_dense_access_group(
    std::span<const MoiPlannedAccessPatch *const> group, uint64_t dispatcher_offset,
    const std::map<uint64_t, MoiDenseEntryHost> &entry_hosts, const MoiDenseRouterPlan &router,
    rj_code_arch_t arch, uint64_t dispatcher_words_per_site, std::string_view diagnostic_subject,
    std::vector<uint8_t> &new_text, std::vector<ConSanPatchInfo> &patches,
    std::vector<std::string> &errors) {
  const auto fail = [&](std::string detail) {
    errors.emplace_back("ConSan MOI " + std::string(diagnostic_subject) + " " + std::move(detail));
    return false;
  };
  if (group.empty() || group.front()->candidate == nullptr || !group.front()->entry_island_offset) {
    return fail("received an incomplete dense access route");
  }
  const ConSanTargetProfile *target = consan_target_profile(arch);
  if (target == nullptr)
    return fail("has no target profile");

  const uint16_t pc_sgpr = router.indirect_jump.pc_sgpr;
  const uint16_t scc_save_sgpr = router.indirect_jump.scc_save_sgpr;
  const uint16_t key_sgpr = router.dispatch_key_sgpr;
  const std::optional<uint16_t> call_return_sgpr = router.call_return_sgpr;
  const bool explicit_key = router.explicit_key;
  const bool key_encodes_scc = explicit_key && key_sgpr == scc_save_sgpr;
  const bool collapsed_spill_router = router.collapse_spill_router;
  const uint32_t island_capacity_words = router.entry_island_words;
  const uint16_t call_match_sgpr =
      collapsed_spill_router ? key_sgpr : call_return_sgpr.value_or(0u);
  const uint64_t island_offset = *group.front()->entry_island_offset;

  std::vector<uint32_t> island_words;
  if (key_encodes_scc)
    island_words.push_back(build_s_nop(0, arch));
  if (collapsed_spill_router) {
    const auto save_scc = instrumentation::build_s_cselect_b32(
        scc_save_sgpr, scalar_positive_inline_u32(1), scalar_positive_inline_u32(0), arch);
    if (!save_scc)
      return fail("could not save its dense entry SCC");
    island_words.push_back(*save_scc);
    if (!explicit_key)
      island_words.push_back(build_s_mov_b32(key_sgpr, *call_return_sgpr, arch));
  }
  if (!append_moi_scc_preserving_indirect_jump(
          island_words, island_offset, dispatcher_offset, pc_sgpr, scc_save_sgpr,
          /*capture_scc=*/!collapsed_spill_router && !key_encodes_scc, arch) ||
      island_words.size() > island_capacity_words) {
    return fail("could not encode its dense entry island");
  }
  if (router.requires_indirect_pc_wait) {
    const auto wait_pc = instrumentation::build_s_wait_indirect_pc0(arch);
    if (!wait_pc || island_words.empty())
      return fail("could not encode its dense entry dependency wait");
    island_words.insert(island_words.end() - 1, *wait_pc);
    if (island_words.size() != island_capacity_words)
      return fail("encoded an invalid dense entry island size");
  } else {
    island_words.resize(island_capacity_words, build_s_nop(0, arch));
  }
  const uint64_t island_bytes = island_words.size() * sizeof(uint32_t);
  if (island_offset > new_text.size() || island_bytes > new_text.size() - island_offset)
    return fail("dense entry island exceeds .text");
  std::memcpy(new_text.data() + island_offset, island_words.data(),
              static_cast<size_t>(island_bytes));

  const auto entry_host = entry_hosts.find(dispatcher_offset);
  ConSanPatchInfo island_info;
  island_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  if (entry_host != entry_hosts.end()) {
    island_info.anchor_offset = entry_host->second.host_offset;
    island_info.original_size =
        static_cast<uint32_t>(entry_host->second.displaced_words.size() * sizeof(uint32_t));
    island_info.trampoline_offset = entry_host->second.body_offset;
    island_info.trampoline_size = static_cast<uint32_t>(
        (entry_host->second.displaced_words.size() + router.relocated_entry_return_words) *
        sizeof(uint32_t));
  } else {
    island_info.anchor_offset = group.front()->candidate->anchor();
    island_info.trampoline_offset = island_offset;
    island_info.trampoline_size = static_cast<uint32_t>(island_bytes);
  }
  island_info.owner_descriptor_file_offsets =
      group.front()->resources.owner_descriptor_file_offsets;
  patches.push_back(std::move(island_info));

  std::vector<uint32_t> dispatcher_words;
  if (key_encodes_scc) {
    // The low bit retains the caller's SCC and the remaining inline-constant
    // bits select the route. Comparisons consume the normalized route number
    // in the now-disposable indirect-PC low register.
    dispatcher_words.push_back(
        build_s_lshr_b32(pc_sgpr, key_sgpr, scalar_positive_inline_u32(1), arch));
  }
  const uint16_t route_key_sgpr = key_encodes_scc ? pc_sgpr : key_sgpr;
  const auto append_route_scc_restore = [&](std::vector<uint32_t> &words) {
    if (!key_encodes_scc) {
      const auto restore =
          instrumentation::build_s_cmp_lg_u32(scc_save_sgpr, scalar_positive_inline_u32(0), arch);
      if (!restore)
        return false;
      words.push_back(*restore);
      return true;
    }
    return consan_detail::append_restore_moi_scc_from_route_key(words, {.encoded_sgpr = key_sgpr},
                                                                *target);
  };
  if (entry_host != entry_hosts.end()) {
    if (consan_arch_has_s_call_i64(arch)) {
      if (!append_moi_call_return_match(dispatcher_words, dispatcher_offset,
                                        entry_host->second.host_offset + sizeof(uint32_t), pc_sgpr,
                                        call_match_sgpr, arch)) {
        return fail("could not encode its relocatable entry-host match");
      }
    } else {
      const auto compare =
          instrumentation::build_s_cmp_eq_u32(route_key_sgpr, scalar_positive_inline_u32(0), arch);
      if (!compare)
        return fail("could not encode its dense entry-host comparison");
      dispatcher_words.push_back(*compare);
    }

    std::vector<uint32_t> matched_words;
    if (router.restore_scc_before_route && !append_route_scc_restore(matched_words))
      return fail("could not restore its dense entry-host SCC");
    const uint64_t matched_offset =
        dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
    const uint64_t matched_branch_offset = matched_offset + matched_words.size() * sizeof(uint32_t);
    const auto direct = explicit_key ? compute_sopp_branch_simm16(matched_branch_offset,
                                                                  entry_host->second.body_offset)
                                     : std::nullopt;
    if (direct) {
      matched_words.push_back(build_s_branch(*direct, arch));
    } else if (!append_moi_scc_preserving_indirect_jump(
                   matched_words, matched_offset, entry_host->second.body_offset, pc_sgpr,
                   scc_save_sgpr, /*capture_scc=*/false, arch)) {
      return fail("could not encode its dense entry-host target");
    }
    if (key_encodes_scc && !direct)
      matched_words[matched_words.size() - 2u] = matched_words.front();
    const auto skip =
        instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
    if (!skip)
      return fail("could not encode its dense entry-host branch");
    dispatcher_words.push_back(*skip);
    dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
  }

  for (size_t route = 0; route < group.size(); ++route) {
    const MoiPlannedAccessPatch &planned = *group[route];
    const ConSanMoiCandidate &candidate = *planned.candidate;
    if (consan_arch_has_s_call_i64(arch)) {
      if (!append_moi_call_return_match(dispatcher_words, dispatcher_offset,
                                        candidate.anchor() + sizeof(uint32_t), pc_sgpr,
                                        call_match_sgpr, arch)) {
        return fail("could not encode its relocatable caller match");
      }
    } else {
      const auto compare = instrumentation::build_s_cmp_eq_u32(
          route_key_sgpr, scalar_positive_inline_u32(static_cast<uint16_t>(route + 1u)), arch);
      if (!compare)
        return fail("could not encode its dense dispatch comparison");
      dispatcher_words.push_back(*compare);
    }

    std::vector<uint32_t> matched_words;
    if (router.restore_scc_before_route && !append_route_scc_restore(matched_words))
      return fail("could not restore its dense route SCC");
    if (!explicit_key && !collapsed_spill_router && candidate.size() > sizeof(uint32_t)) {
      const uint32_t adjustment = candidate.size() - sizeof(uint32_t);
      matched_words.push_back(
          build_s_add_u32(*call_return_sgpr, *call_return_sgpr, /*literal=*/255u, arch));
      matched_words.push_back(adjustment);
      matched_words.push_back(build_s_addc_u32(static_cast<uint16_t>(*call_return_sgpr + 1u),
                                               static_cast<uint16_t>(*call_return_sgpr + 1u),
                                               scalar_positive_inline_u32(0), arch));
    }
    const uint64_t matched_offset =
        dispatcher_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
    const uint64_t matched_branch_offset = matched_offset + matched_words.size() * sizeof(uint32_t);
    const auto direct = explicit_key ? compute_sopp_branch_simm16(matched_branch_offset,
                                                                  planned.placement.body_offset)
                                     : std::nullopt;
    if (direct) {
      matched_words.push_back(build_s_branch(*direct, arch));
    } else if (!append_moi_scc_preserving_indirect_jump(
                   matched_words, matched_offset, planned.placement.body_offset, pc_sgpr,
                   scc_save_sgpr, /*capture_scc=*/false, arch)) {
      return fail("could not encode a dense dispatch target");
    }
    if (key_encodes_scc && !direct)
      matched_words[matched_words.size() - 2u] = matched_words.front();
    if (matched_words.size() > static_cast<size_t>(std::numeric_limits<int16_t>::max()))
      return fail("dense dispatch stub is too large");
    const auto skip =
        instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched_words.size()), arch);
    if (!skip)
      return fail("could not encode a dense dispatch branch");
    dispatcher_words.push_back(*skip);
    dispatcher_words.insert(dispatcher_words.end(), matched_words.begin(), matched_words.end());
  }
  dispatcher_words.push_back(build_s_endpgm(arch));
  const uint64_t reserved_words = group.size() * dispatcher_words_per_site;
  if (dispatcher_words.size() > reserved_words || dispatcher_offset > new_text.size() ||
      dispatcher_words.size() * sizeof(uint32_t) > new_text.size() - dispatcher_offset) {
    return fail("dense dispatcher requires " + std::to_string(dispatcher_words.size()) +
                " words for " + std::to_string(group.size()) + " sites, but reserved " +
                std::to_string(reserved_words));
  }
  std::memcpy(new_text.data() + dispatcher_offset, dispatcher_words.data(),
              dispatcher_words.size() * sizeof(uint32_t));

  ConSanPatchInfo dispatcher_info;
  dispatcher_info.kind = ConSanPatchKind::TrampolineMoiIndirectBranchIsland;
  dispatcher_info.anchor_offset = group.front()->candidate->anchor();
  dispatcher_info.trampoline_offset = dispatcher_offset;
  dispatcher_info.trampoline_size =
      static_cast<uint32_t>(dispatcher_words.size() * sizeof(uint32_t));
  if (router.publish_entry_island_offset)
    dispatcher_info.moi_dense_entry_island_offset = island_offset;
  dispatcher_info.owner_descriptor_file_offsets =
      group.front()->resources.owner_descriptor_file_offsets;
  patches.push_back(std::move(dispatcher_info));

  if (entry_host != entry_hosts.end()) {
    std::optional<uint32_t> host_entry;
    if (key_encodes_scc) {
      host_entry = instrumentation::build_s_cselect_b32(key_sgpr, scalar_positive_inline_u32(1),
                                                        scalar_positive_inline_u32(0), arch);
    } else if (explicit_key) {
      host_entry = build_s_mov_b32(key_sgpr, scalar_positive_inline_u32(0), arch);
    } else {
      host_entry = instrumentation::build_s_call_i64(*call_return_sgpr,
                                                     /*next instruction=*/0, arch);
    }
    if (!host_entry || entry_host->second.host_offset > new_text.size() ||
        sizeof(uint32_t) > new_text.size() - entry_host->second.host_offset) {
      return fail("could not encode its dense entry host");
    }
    std::memcpy(new_text.data() + entry_host->second.host_offset, &*host_entry, sizeof(uint32_t));
  }

  for (size_t route = 0; route < group.size(); ++route) {
    const ConSanMoiCandidate &candidate = *group[route]->candidate;
    const uint64_t branch_offset = candidate.anchor() + (explicit_key ? sizeof(uint32_t) : 0u);
    const auto call_delta = compute_sopp_branch_simm16(branch_offset, island_offset);
    const std::optional<uint32_t> call =
        !call_delta || !call_return_sgpr
            ? std::nullopt
            : instrumentation::build_s_call_i64(*call_return_sgpr, *call_delta, arch);
    if ((!explicit_key && !call) ||
        candidate.size() < (explicit_key ? 2u : 1u) * sizeof(uint32_t) ||
        candidate.anchor() > new_text.size() ||
        candidate.size() > new_text.size() - candidate.anchor()) {
      return fail("could not encode a dense call anchor");
    }
    std::vector<uint32_t> anchor_words(candidate.size() / sizeof(uint32_t), build_s_nop(0, arch));
    if (explicit_key) {
      if (!call_delta || route >= 64u)
        return fail("could not encode an explicit dense branch key");
      if (key_encodes_scc) {
        const uint16_t route_key = static_cast<uint16_t>(2u * (route + 1u));
        const auto tagged_key = instrumentation::build_s_cselect_b32(
            key_sgpr, scalar_positive_inline_u32(static_cast<uint16_t>(route_key + 1u)),
            scalar_positive_inline_u32(route_key), arch);
        if (!tagged_key)
          return fail("could not encode a tagged dense branch key");
        anchor_words[0] = *tagged_key;
      } else {
        anchor_words[0] = build_s_mov_b32(
            key_sgpr, scalar_positive_inline_u32(static_cast<uint16_t>(route + 1u)), arch);
      }
      anchor_words[1] = build_s_branch(*call_delta, arch);
    } else {
      anchor_words.front() = *call;
    }
    std::memcpy(new_text.data() + candidate.anchor(), anchor_words.data(), candidate.size());
  }
  return true;
}
