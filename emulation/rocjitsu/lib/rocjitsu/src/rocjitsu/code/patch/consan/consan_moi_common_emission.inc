// Retained intentionally for the structural dense-routing template; compiled
// shared lowering lives in consan_moi_shared_lowering.cpp.

/// Emit the owner-local dense entry island, dispatcher, optional relocated
/// host entry, and original-site route anchors shared by all three access
/// engines.
///
/// `PlannedPatch` is deliberately structural: the engines retain distinct
/// evidence-body plans, but expose the same `candidate`, `resources`,
/// `placement`, and `entry_island_offset` facts to this routing mechanism.
/// The boolean parameter preserves the Record/Replay-versus-Sampled policy for
/// equal indirect-PC and call-return assignments on explicit-key targets.
/// Inline Shadow selects its distinct scalar ABI through the request and
/// owner-bound operating point; its
/// SCC-tagged explicit key and indirect-PC dependency wait remain explicit
/// branches of this shared routing mechanism. Evidence construction and
/// per-site resource admission remain outside this function.
[[nodiscard]] bool emit_moi_dense_access_group(
    std::span<const MoiPlannedAccessPatch *const> group, uint64_t dispatcher_offset,
    const std::map<uint64_t, MoiDenseEntryHost> &entry_hosts, const ConSanRequest &request,
    const ConSanMoiOperatingPoint &point, rj_code_arch_t arch, uint64_t dispatcher_words_per_site,
    bool collapse_spill_router_for_explicit_key, std::string_view diagnostic_subject,
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

  const bool inline_shadow_route = request.moi_engine == ConSanMoiEngine::InlineShadow;
  uint16_t pc_sgpr = 0u;
  uint16_t scc_save_sgpr = 0u;
  uint16_t key_sgpr = 0u;
  std::optional<uint16_t> call_return_sgpr;
  bool explicit_key = false;
  bool key_encodes_scc = false;
  bool collapsed_spill_router = false;
  uint32_t island_capacity_words = 0u;
  uint32_t relocated_entry_return_words = 0u;

  if (inline_shadow_route) {
    const auto scalar_abi = moi_inline_dense_router_scalar_abi(request, point, arch);
    if (!scalar_abi)
      return fail("lost its inline-shadow dense scalar ABI");
    pc_sgpr = scalar_abi->indirect_jump.pc_sgpr;
    scc_save_sgpr = scalar_abi->indirect_jump.scc_save_sgpr;
    key_sgpr = scalar_abi->dispatch_key_sgpr;
    call_return_sgpr = scalar_abi->call_return_sgpr;
    explicit_key = !call_return_sgpr.has_value();
    key_encodes_scc = explicit_key && key_sgpr == scc_save_sgpr;
    island_capacity_words = kMoiInlineShadowIndirectIslandWords;
    relocated_entry_return_words = kMoiInlineShadowIndirectIslandWords;
  } else {
    const bool spill_backed = point.has_compact_moi_scalar_spill();
    if (!point.moi_exec_save_sgpr ||
        (spill_backed && (!point.moi_router_indirect_pc_sgpr || !point.moi_router_call))) {
      return fail("lost its recording dense scalar ABI");
    }
    const uint16_t base = *point.moi_exec_save_sgpr;
    const std::optional<MoiSpecialStateSgprs> special_state =
        moi_special_state_sgprs(request, point);
    if (!special_state)
      return fail("lost its dense special-state scalar ABI");
    pc_sgpr = spill_backed ? *point.moi_router_indirect_pc_sgpr : base;
    scc_save_sgpr = special_state->scc_save_sgpr;
    key_sgpr =
        spill_backed ? point.moi_router_call->dispatch_key_sgpr : static_cast<uint16_t>(base + 5u);
    call_return_sgpr =
        spill_backed ? point.moi_router_call->call_return_sgpr : static_cast<uint16_t>(base + 6u);
    explicit_key = !consan_arch_has_s_call_i64(arch);
    collapsed_spill_router = spill_backed && *call_return_sgpr == pc_sgpr &&
                             (!explicit_key || collapse_spill_router_for_explicit_key);
    island_capacity_words = moi_record_replay_entry_island_words(spill_backed);
    relocated_entry_return_words = kMoiRecordReplayIndirectIslandWords;
  }
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
  if (inline_shadow_route) {
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
        (entry_host->second.displaced_words.size() + relocated_entry_return_words) *
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
    if ((inline_shadow_route || explicit_key) && !append_route_scc_restore(matched_words))
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
    if ((inline_shadow_route || explicit_key) && !append_route_scc_restore(matched_words))
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
  if (inline_shadow_route)
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
