[[nodiscard]] bool append_inline_shadow_diagnostic_words(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate, const ConSanOptions &options,
    rj_code_arch_t arch, const ConSanMoiReportBufferLayout &layout, uint16_t old_value_vgpr,
    uint16_t old_value_hi_vgpr, uint16_t current_value_vgpr, uint16_t current_field_vgpr,
    uint16_t lds_byte_offset_vgpr, uint32_t static_byte_offset, uint32_t byte_count,
    bool special_state_already_saved = false,
    std::optional<uint16_t> diagnostic_lane_mask_sgpr = std::nullopt,
    bool capture_first_diagnostic_only = false, bool compact_local_shadow = false,
    bool workgroup_local_shadow = false, bool generation_tagged_local_shadow = false,
    std::optional<uint16_t> prior_byte_provenance_vgpr = std::nullopt,
    std::optional<uint16_t> current_byte_provenance_vgpr = std::nullopt,
    std::optional<uint16_t> relative_cell_index_vgpr = std::nullopt) {
  if (!options.moi_exec_save_sgpr || layout.diagnostic_capacity == 0)
    return true;
  if (prior_byte_provenance_vgpr.has_value() != current_byte_provenance_vgpr.has_value())
    return false;

  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint64_t report_base = *options.moi_report_buffer_address;
  const uint64_t diagnostic_base = report_base + layout.diagnostic_records_offset;

  if (!special_state_already_saved && !append_save_moi_special_state(words, options, arch))
    return false;

  const uint16_t workgroup_key = static_cast<uint16_t>(*options.scratch_vgpr + 19u);
  if (options.moi_track_atomics) {
    if (workgroup_local_shadow) {
      // An LDS shadow is workgroup-local. Its optional generation tag guards
      // reuse of the local cells; it is not the identity used by the global
      // atomic release/token tables. Validate those tables with the persistent
      // workgroup key for every local-shadow representation.
      if (options.moi_workgroup_key_vgpr) {
        words.push_back(build_v_mov_b32_e32(
            workgroup_key, vector_source_vgpr(*options.moi_workgroup_key_vgpr), arch));
      } else if (options.moi_persistent_workgroup_key_sgpr) {
        words.push_back(
            build_v_mov_b32_e32(workgroup_key, *options.moi_persistent_workgroup_key_sgpr, arch));
      } else {
        return false;
      }
    } else if (!append_extract_exact_shadow_generation(
                   words, workgroup_key, current_value_vgpr,
                   static_cast<uint16_t>(current_value_vgpr + 1u), tmp_vgpr, arch)) {
      return false;
    }
  }

  // Conflict diagnostics are the cold path. Each predicate below can empty
  // EXEC, but merely leaving the remaining vector and memory instructions
  // predicated off still makes scalar execution walk the complete record
  // builder. Large clean kernels can encounter these probes billions of
  // times. Branch directly to the common state restoration whenever no lane
  // remains, preserving identical behavior for every actual conflict.
  std::vector<size_t> empty_diagnostic_branches;
  const auto skip_diagnostic_if_empty = [&]() {
    const auto branch = instrumentation::build_s_cbranch_execz(0, arch);
    if (!branch)
      return false;
    empty_diagnostic_branches.push_back(words.size());
    words.push_back(*branch);
    return true;
  };

  const uint32_t mov_zero = build_v_mov_b32_e32(tmp_vgpr, scalar_positive_inline_u32(0), arch);
  const auto prior_nonempty =
      instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(old_value_vgpr), tmp_vgpr, arch);
  const auto narrow_nonempty =
      instrumentation::build_s_and_saveexec_b64(*options.moi_exec_save_sgpr, kRdna4VccLo, arch);
  if (!prior_nonempty || !narrow_nonempty)
    return false;
  words.push_back(mov_zero);
  words.push_back(*prior_nonempty);
  words.push_back(*narrow_nonempty);
  if (!skip_diagnostic_if_empty())
    return false;

  const uint16_t address_lo_vgpr = *options.scratch_vgpr;
  // Predicate save destinations are scratch: after each empty-mask branch,
  // only the narrowed EXEC matters. Reuse one pair instead of consuming the
  // VCC/SCC preservation window at higher offsets as predicates are reordered.
  const uint16_t predicate_exec_save_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u);
  if (!options.moi_inline_workgroup_shadow || generation_tagged_local_shadow) {
    if (!append_extract_exact_shadow_generation(words, tmp_vgpr, old_value_vgpr, old_value_hi_vgpr,
                                                address_lo_vgpr, arch) ||
        !append_extract_exact_shadow_generation(words, current_field_vgpr, current_value_vgpr,
                                                static_cast<uint16_t>(current_value_vgpr + 1u),
                                                address_lo_vgpr, arch)) {
      return false;
    }
    const auto same_workgroup = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
    const auto narrow_same_workgroup =
        instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kRdna4VccLo, arch);
    if (!same_workgroup || !narrow_same_workgroup)
      return false;
    words.push_back(*same_workgroup);
    words.push_back(*narrow_same_workgroup);
    if (!skip_diagnostic_if_empty())
      return false;
  }

  if (prior_byte_provenance_vgpr) {
    const auto prior_byte_mask = instrumentation::build_v_and_b32_literal(
        tmp_vgpr, consan_moi_exact_byte_cell::byte_mask_mask, *prior_byte_provenance_vgpr, arch);
    const auto current_byte_mask = instrumentation::build_v_and_b32_literal(
        current_field_vgpr, consan_moi_exact_byte_cell::byte_mask_mask,
        *current_byte_provenance_vgpr, arch);
    const auto overlapping_bytes = instrumentation::build_v_and_b32(
        current_field_vgpr, vector_source_vgpr(tmp_vgpr), current_field_vgpr, arch);
    const auto overlap_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0u), current_field_vgpr, arch);
    const auto narrow_overlap =
        instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kRdna4VccLo, arch);
    if (!prior_byte_mask || !current_byte_mask || !overlapping_bytes || !overlap_nonzero ||
        !narrow_overlap) {
      return false;
    }
    words.insert(words.end(), prior_byte_mask->begin(), prior_byte_mask->end());
    words.insert(words.end(), current_byte_mask->begin(), current_byte_mask->end());
    words.push_back(*overlapping_bytes);
    words.push_back(*overlap_nonzero);
    words.push_back(*narrow_overlap);
    if (!skip_diagnostic_if_empty())
      return false;
  }

  // Reject read/read traffic before extracting either owner or epoch. Preserve
  // owner-before-epoch ordering for the remaining lanes: most repeated
  // accesses by one workitem are ordered, while a same-site access from a
  // different physical lane still represents a distinct participant.
  const auto current_kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  if (current_kind != ConSanMoiShadowAccessKind::Write) {
    const auto prior_kind = instrumentation::build_v_and_b32_literal(
        tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), old_value_vgpr,
        arch);
    const auto kind_ne = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(static_cast<uint32_t>(current_kind)), tmp_vgpr, arch);
    const auto narrow_kind_conflict =
        instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kRdna4VccLo, arch);
    if (!prior_kind || !kind_ne || !narrow_kind_conflict)
      return false;
    words.insert(words.end(), prior_kind->begin(), prior_kind->end());
    words.push_back(*kind_ne);
    words.push_back(*narrow_kind_conflict);
    if (!skip_diagnostic_if_empty())
      return false;
  }

  if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch) ||
      !append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch)) {
    return false;
  }
  const auto owner_ne = instrumentation::build_v_cmp_ne_u32_vcc(
      vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
  if (!owner_ne)
    return false;
  if (prior_byte_provenance_vgpr) {
    const uint16_t candidate_exec = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u);
    const uint16_t different_owner_exec = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u);
    const uint16_t predicate_exec = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 6u);
    const auto save_candidates =
        instrumentation::build_s_mov_b64(candidate_exec, kRdna4ExecLo, arch);
    const auto narrow_different_owner =
        instrumentation::build_s_and_saveexec_b64(predicate_exec, kRdna4VccLo, arch);
    const auto save_different_owner =
        instrumentation::build_s_mov_b64(different_owner_exec, kRdna4ExecLo, arch);
    const auto restore_candidates =
        instrumentation::build_s_mov_b64(kRdna4ExecLo, candidate_exec, arch);
    const auto owner_eq = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
    const auto narrow_same_owner =
        instrumentation::build_s_and_saveexec_b64(predicate_exec, kRdna4VccLo, arch);
    const auto prior_instruction = instrumentation::build_v_lshrrev_b32(
        tmp_vgpr,
        scalar_positive_inline_u32(consan_moi_exact_shadow::instruction_offset_shift - 32u),
        old_value_hi_vgpr, arch);
    const auto current_instruction = instrumentation::build_v_mov_b32_literal(
        current_field_vgpr, static_cast<uint32_t>(candidate.text_offset), arch);
    const auto same_instruction = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
    const auto narrow_same_instruction =
        instrumentation::build_s_and_saveexec_b64(predicate_exec, kRdna4VccLo, arch);
    const auto prior_lane = instrumentation::build_v_lshrrev_b32(
        tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_byte_cell::lane_plus_one_shift),
        *prior_byte_provenance_vgpr, arch);
    const auto current_lane = instrumentation::build_v_lshrrev_b32(
        current_field_vgpr,
        scalar_positive_inline_u32(consan_moi_exact_byte_cell::lane_plus_one_shift),
        *current_byte_provenance_vgpr, arch);
    const auto different_lane = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
    const auto narrow_different_lane =
        instrumentation::build_s_and_saveexec_b64(predicate_exec, kRdna4VccLo, arch);
    const auto union_conflicts =
        instrumentation::build_s_xor_b64(kRdna4ExecLo, different_owner_exec, kRdna4ExecLo, arch);
    if (!save_candidates || !narrow_different_owner || !save_different_owner ||
        !restore_candidates || !owner_eq || !narrow_same_owner || !prior_instruction ||
        !current_instruction || !same_instruction || !narrow_same_instruction || !prior_lane ||
        !current_lane || !different_lane || !narrow_different_lane || !union_conflicts) {
      return false;
    }
    words.push_back(*save_candidates);
    words.push_back(*owner_ne);
    words.push_back(*narrow_different_owner);
    words.push_back(*save_different_owner);
    words.push_back(*restore_candidates);
    words.push_back(*owner_eq);
    words.push_back(*narrow_same_owner);
    words.push_back(*prior_instruction);
    words.insert(words.end(), current_instruction->begin(), current_instruction->end());
    words.push_back(*same_instruction);
    words.push_back(*narrow_same_instruction);
    words.push_back(*prior_lane);
    words.push_back(*current_lane);
    words.push_back(*different_lane);
    words.push_back(*narrow_different_lane);
    words.push_back(*union_conflicts);
  } else {
    const auto narrow_conflict =
        instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kRdna4VccLo, arch);
    if (!narrow_conflict)
      return false;
    words.push_back(*owner_ne);
    words.push_back(*narrow_conflict);
  }
  if (!skip_diagnostic_if_empty())
    return false;

  const bool has_epoch = options.moi_epoch_vgpr || options.moi_persistent_epoch_sgpr ||
                         options.automatic_moi_private_epoch;
  if (has_epoch) {
    if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch) ||
        !append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch)) {
      return false;
    }
    const auto epoch_eq = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
    const auto narrow_same_epoch =
        instrumentation::build_s_and_saveexec_b64(predicate_exec_save_sgpr, kRdna4VccLo, arch);
    if (!epoch_eq || !narrow_same_epoch)
      return false;
    words.push_back(*epoch_eq);
    words.push_back(*narrow_same_epoch);
    if (!skip_diagnostic_if_empty())
      return false;
  }

  if (options.moi_track_atomics) {
    const uint16_t base = *options.scratch_vgpr;
    const uint16_t exec_base = *options.moi_exec_save_sgpr;
    const uint16_t conflict_exec = static_cast<uint16_t>(exec_base + 2u);
    const uint16_t predicate_exec = static_cast<uint16_t>(exec_base + 6u);
    // The wave-coalesced exact-shadow caller keeps +7:+12 live across this
    // helper as its saved address, packed current value, and exact-byte
    // provenance. A multi-address wave returns to the partition loop after
    // token filtering, so borrowing those registers here corrupts the next
    // partition's publication. Keep access-time token state in the cold
    // diagnostic-only tail instead.
    const uint16_t token_version = static_cast<uint16_t>(base + 15u);
    const uint16_t token_kind = static_cast<uint16_t>(base + 16u);
    const uint16_t token_epoch = static_cast<uint16_t>(base + 17u);
    const uint16_t source_version = static_cast<uint16_t>(base + 18u);
    const uint16_t token_hash = static_cast<uint16_t>(base + 14u);
    const uint16_t source_address = static_cast<uint16_t>(base + 21u);
    const uint16_t prior_owner = static_cast<uint16_t>(base + 23u);
    const uint16_t consumer_owner =
        options.moi_owner_vgpr ? *options.moi_owner_vgpr : static_cast<uint16_t>(base + 24u);
    if (!options.moi_owner_vgpr) {
      if (!options.moi_persistent_owner_sgpr)
        return false;
      words.push_back(
          build_v_mov_b32_e32(consumer_owner, *options.moi_persistent_owner_sgpr, arch));
    }
    if (!append_extract_exact_shadow_field(words, current_field_vgpr, current_value_vgpr,
                                           consan_moi_exact_shadow::owner_shift,
                                           consan_moi_exact_shadow::max_owner, arch) ||
        !append_extract_exact_shadow_field(words, prior_owner, old_value_vgpr,
                                           consan_moi_exact_shadow::owner_shift,
                                           consan_moi_exact_shadow::max_owner, arch))
      return false;

    const auto filter_ordered_pair = [&](uint16_t token_consumer_owner,
                                         uint16_t token_producer_owner,
                                         uint16_t consumer_packed_value,
                                         uint16_t producer_packed_value) {
      const auto save_conflict =
          instrumentation::build_s_mov_b64(conflict_exec, kRdna4ExecLo, arch);
      if (!save_conflict)
        return false;
      words.push_back(*save_conflict);
      if (!append_inline_acquired_token_slot_address(
              words,
              *options.moi_report_buffer_address + layout.inline_acquired_epoch_token_slots_offset,
              layout.inline_acquired_epoch_token_capacity, workgroup_key, token_consumer_owner,
              token_producer_owner, address_lo_vgpr, token_hash,
              static_cast<uint16_t>(address_lo_vgpr + 1u),
              /*release_sequence=*/false, arch))
        return false;

      const auto narrow = [&]() {
        const auto instruction =
            instrumentation::build_s_and_saveexec_b64(predicate_exec, kRdna4VccLo, arch);
        if (!instruction)
          return false;
        words.push_back(*instruction);
        return true;
      };
      const auto require_equal = [&](size_t offset, uint16_t expected) {
        if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr, offset, tmp_vgpr, arch))
          return false;
        const auto equal =
            instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), tmp_vgpr, arch);
        if (!equal)
          return false;
        words.push_back(*equal);
        return narrow();
      };
      const auto require_scalar_equal = [&](size_t offset, uint16_t expected) {
        if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr, offset, tmp_vgpr, arch))
          return false;
        const auto equal = instrumentation::build_v_cmp_eq_u32_vcc(expected, tmp_vgpr, arch);
        if (!equal)
          return false;
        words.push_back(*equal);
        return narrow();
      };
      const auto require_zero = [&](size_t offset) {
        if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr, offset, tmp_vgpr, arch))
          return false;
        const auto zero =
            instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
        if (!zero)
          return false;
        words.push_back(*zero);
        return narrow();
      };

      // The acquiring atomic can populate this direct-mapped slot immediately
      // before the same wave reaches its ordered LDS access. Use a coherent
      // device-scope version read at both ends of the snapshot; an ordinary
      // flat load may otherwise reuse the zero observed while reserving the
      // slot and miss the completed token publication.
      if (!append_atomic_load_u32(words, address_lo_vgpr, token_version, arch))
        return false;
      const auto token_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
          scalar_positive_inline_u32(0), token_version, arch);
      const auto token_parity =
          instrumentation::build_v_and_b32_literal(tmp_vgpr, 1u, token_version, arch);
      if (!token_nonzero || !token_parity)
        return false;
      words.push_back(*token_nonzero);
      if (!narrow())
        return false;
      words.insert(words.end(), token_parity->begin(), token_parity->end());
      const auto token_even =
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
      if (!token_even)
        return false;
      words.push_back(*token_even);
      if (!narrow() ||
          !require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
                         workgroup_key) ||
          !require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
                         token_consumer_owner) ||
          !require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
                         token_producer_owner))
        return false;
      if (!append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one), token_epoch,
              arch) ||
          !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                          offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind),
                                          token_kind, arch) ||
          !require_scalar_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id),
                                *options.moi_dispatch_id_sgpr) ||
          !require_scalar_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) +
                                    sizeof(uint32_t),
                                static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u)) ||
          !append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address),
              source_address, arch) ||
          !append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address) +
                  sizeof(uint32_t),
              static_cast<uint16_t>(source_address + 1u), arch) ||
          !append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version),
              source_version, arch) ||
          !append_load_u32_vgpr_at_offset(
              words, address_lo_vgpr,
              offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_epoch_plus_one), token_hash,
              arch) ||
          !require_zero(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reservation_version)) ||
          !append_atomic_load_u32(words, address_lo_vgpr, static_cast<uint16_t>(base + 14u), arch))
        return false;
      const auto source_address_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
          scalar_positive_inline_u32(0), source_address, arch);
      const auto source_version_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
          scalar_positive_inline_u32(0), source_version, arch);
      const auto source_version_parity =
          instrumentation::build_v_and_b32_literal(tmp_vgpr, 1u, source_version, arch);
      if (!source_address_nonzero || !source_version_nonzero || !source_version_parity)
        return false;
      words.push_back(*source_address_nonzero);
      if (!narrow())
        return false;
      words.push_back(*source_version_nonzero);
      if (!narrow())
        return false;
      words.insert(words.end(), source_version_parity->begin(), source_version_parity->end());
      const auto source_version_even =
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
      if (!source_version_even)
        return false;
      words.push_back(*source_version_even);
      if (!narrow())
        return false;
      const auto token_unchanged = instrumentation::build_v_cmp_eq_u32_vcc(
          vector_source_vgpr(token_version), static_cast<uint16_t>(base + 14u), arch);
      if (!token_unchanged)
        return false;
      words.push_back(*token_unchanged);
      if (!narrow() ||
          !append_extract_exact_shadow_field(words, source_version, consumer_packed_value,
                                             consan_moi_exact_shadow::epoch_shift,
                                             consan_moi_exact_shadow::max_epoch, arch))
        return false;
      const auto max_epoch = instrumentation::build_v_mov_b32_literal(
          tmp_vgpr, consan_moi_exact_shadow::max_epoch, arch);
      const auto consumer_below_max = instrumentation::build_v_cmp_ne_u32_vcc(
          vector_source_vgpr(source_version), tmp_vgpr, arch);
      if (!max_epoch || !consumer_below_max)
        return false;
      words.insert(words.end(), max_epoch->begin(), max_epoch->end());
      words.push_back(*consumer_below_max);
      if (!narrow())
        return false;
      const auto add_consumer_one = instrumentation::build_v_add_u32(
          source_version, scalar_positive_inline_u32(1), source_version, arch);
      if (!add_consumer_one)
        return false;
      words.insert(words.end(), add_consumer_one->begin(), add_consumer_one->end());
      const auto token_too_new = instrumentation::build_v_cmp_gt_u32_vcc(
          vector_source_vgpr(token_hash), source_version, arch);
      if (!token_too_new)
        return false;
      words.push_back(*token_too_new);
      const auto remove_token_too_new =
          instrumentation::build_s_andn2_b64(kRdna4ExecLo, kRdna4ExecLo, kRdna4VccLo, arch);
      if (!remove_token_too_new)
        return false;
      words.push_back(*remove_token_too_new);
      if (!append_extract_exact_shadow_field(words, token_hash, producer_packed_value,
                                             consan_moi_exact_shadow::epoch_shift,
                                             consan_moi_exact_shadow::max_epoch, arch))
        return false;
      const auto producer_max_epoch = instrumentation::build_v_mov_b32_literal(
          tmp_vgpr, consan_moi_exact_shadow::max_epoch, arch);
      const auto producer_below_max =
          instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(token_hash), tmp_vgpr, arch);
      if (!producer_max_epoch || !producer_below_max)
        return false;
      words.insert(words.end(), producer_max_epoch->begin(), producer_max_epoch->end());
      words.push_back(*producer_below_max);
      if (!narrow())
        return false;
      const auto add_producer_one = instrumentation::build_v_add_u32(
          token_hash, scalar_positive_inline_u32(1), token_hash, arch);
      if (!add_producer_one)
        return false;
      words.insert(words.end(), add_producer_one->begin(), add_producer_one->end());
      const auto token_insufficient = instrumentation::build_v_cmp_gt_u32_vcc(
          vector_source_vgpr(token_hash), token_epoch, arch);
      if (!token_insufficient)
        return false;
      words.push_back(*token_insufficient);
      const auto remove_insufficient =
          instrumentation::build_s_andn2_b64(kRdna4ExecLo, kRdna4ExecLo, kRdna4VccLo, arch);
      if (!remove_insufficient)
        return false;
      words.push_back(*remove_insufficient);

      // Direct and inherited tokens are committed only after a stable source
      // release has been acquired and the guest RMW has completed. The token
      // is therefore the durable causal fact. Release-sequence tokens carry
      // ancestry between releases and never authorize an ordinary access.
      const auto access_authorized = instrumentation::build_v_cmp_gt_u32_vcc(
          scalar_positive_inline_u32(
              static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence)),
          token_kind, arch);
      const auto remove_ordered =
          instrumentation::build_s_andn2_b64(kRdna4ExecLo, conflict_exec, kRdna4VccLo, arch);
      if (!access_authorized || !remove_ordered)
        return false;
      words.push_back(*access_authorized);
      words.push_back(*remove_ordered);
      return true;
    };

    // A conflicting pair is ordered when either access happens-before the
    // other. The LDS exchange is an observation mechanism, not the memory
    // model's clock: a logically later access can become the stored
    // representative before an earlier wave finishes its own exchange.
    // Check both directed acquired-token identities before reporting.
    if (!filter_ordered_pair(current_field_vgpr, prior_owner, current_value_vgpr, old_value_vgpr) ||
        !filter_ordered_pair(prior_owner, current_field_vgpr, old_value_vgpr, current_value_vgpr))
      return false;
  }

  // Writes do not have the read/read kind predicate, and atomic ordering can
  // remove the last candidate after all of the common filters. Cover both
  // cases with one final gate before constructing a diagnostic record.
  if (!skip_diagnostic_if_empty())
    return false;

  const bool compact_local_diagnostic =
      options.moi_inline_workgroup_shadow && !options.moi_track_atomics;
  // The first-diagnostic claim uses SLOT as both the returned value and the
  // low half of the compare-swap data tuple. Keep it in the dead tail of the
  // ordinary transaction window on every target: exact-range construction
  // reuses current_field_vgpr, so aliasing SLOT with that temporary redirects
  // later record stores through a byte offset or negative range endpoint.
  // CDNA3/4 additionally require the tuple to start at an even VGPR.
  constexpr uint16_t diagnostic_tuple_offset =
      consan_detail::inline_shadow_transaction_scratch_count(/*has_exec_save=*/true,
                                                             /*track_atomics=*/false) -
      2u;
  const uint16_t diagnostic_tuple_candidate =
      static_cast<uint16_t>(*options.scratch_vgpr + diagnostic_tuple_offset);
  const uint16_t cdna_slot_vgpr = static_cast<uint16_t>(diagnostic_tuple_candidate & ~1u);
  const uint16_t slot_vgpr = arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4
                                 ? cdna_slot_vgpr
                                 : diagnostic_tuple_candidate;
  const auto save_conflict_exec = instrumentation::build_s_mov_b64(
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kRdna4ExecLo, arch);
  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      tmp_vgpr, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u),
      scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      tmp_vgpr, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 3u),
      vector_source_vgpr(tmp_vgpr), arch);
  const auto first_active_lane =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
  const auto narrow_representative = instrumentation::build_s_and_saveexec_b64(
      static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kRdna4VccLo, arch);
  const auto saved_exec_wait = instrumentation::build_salu_to_valu_dependency_wait(arch);
  if (!save_conflict_exec || !saved_exec_wait || !mbcnt_lo || !mbcnt_hi || !first_active_lane ||
      !narrow_representative)
    return false;
  words.push_back(*save_conflict_exec);
  words.push_back(*saved_exec_wait);
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_active_lane);
  words.push_back(*narrow_representative);

  if (capture_first_diagnostic_only) {
    // A concurrency fault can make the same static conflict execute in every
    // affected workgroup. Preserve one complete, actionable first-fault
    // record instead of turning those repetitions into report overflow. The
    // returned old count is also the slot index: only the 0 -> 1 winner keeps
    // EXEC and writes slot zero.
    const uint64_t count_address = report_base + offsetof(ConSanMoiReportHeader, diagnostic_count);
    const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
        *options.scratch_vgpr, static_cast<uint32_t>(count_address), arch);
    const auto mov_address_hi =
        instrumentation::build_v_mov_b32_literal(static_cast<uint16_t>(*options.scratch_vgpr + 1u),
                                                 static_cast<uint32_t>(count_address >> 32u), arch);
    const uint32_t mov_claimed =
        build_v_mov_b32_e32(slot_vgpr, scalar_positive_inline_u32(1), arch);
    const uint32_t mov_unclaimed = build_v_mov_b32_e32(static_cast<uint16_t>(slot_vgpr + 1u),
                                                       scalar_positive_inline_u32(0), arch);
    const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(
        *options.scratch_vgpr, slot_vgpr, slot_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
        arch);
    const auto claim_won =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), slot_vgpr, arch);
    const auto narrow_winner = instrumentation::build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kRdna4VccLo, arch);
    if (!mov_address_lo || !mov_address_hi || !claim || !claim_won || !narrow_winner)
      return false;
    words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
    words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());
    words.push_back(mov_claimed);
    words.push_back(mov_unclaimed);
    words.insert(words.end(), claim->begin(), claim->end());
    if (!append_moi_global_atomic_wait(words, arch))
      return false;
    words.push_back(*claim_won);
    words.push_back(*narrow_winner);
  } else {
    if (!append_atomic_fetch_add_one_u32(
            words, report_base + offsetof(ConSanMoiReportHeader, diagnostic_count), slot_vgpr,
            *options.scratch_vgpr, arch))
      return false;
    const auto mov_capacity =
        instrumentation::build_v_mov_b32_literal(tmp_vgpr, layout.diagnostic_capacity, arch);
    const auto slot_in_capacity =
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(tmp_vgpr), slot_vgpr, arch);
    const auto narrow_capacity = instrumentation::build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 4u), kRdna4VccLo, arch);
    if (!mov_capacity || !slot_in_capacity || !narrow_capacity)
      return false;
    words.insert(words.end(), mov_capacity->begin(), mov_capacity->end());
    words.push_back(*slot_in_capacity);
    words.push_back(*narrow_capacity);
  }

  const uint16_t compact_address_vgpr =
      arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4
          ? *options.scratch_vgpr
          : static_cast<uint16_t>(*options.scratch_vgpr + 11u);
  if (compact_local_diagnostic &&
      !append_dynamic_diagnostic_record_address(words, diagnostic_base, slot_vgpr,
                                                compact_address_vgpr, arch)) {
    return false;
  }

  const auto store_literal = [&](size_t offset, uint32_t value) {
    if (compact_local_diagnostic) {
      // This path claims the freshly zeroed slot exactly once. Avoid writing
      // fields whose ABI value is already zero, and use the one-word inline
      // encoding for small constants. The full diagnostic record remains
      // byte-for-byte compatible while hot exact probes stay compact enough
      // for large code objects.
      if (capture_first_diagnostic_only && value == 0u)
        return true;
      const auto store = instrumentation::build_flat_store_b32(compact_address_vgpr, tmp_vgpr, arch,
                                                               static_cast<uint32_t>(offset));
      if (!store)
        return false;
      if (value <= 64u) {
        words.push_back(build_v_mov_b32_e32(tmp_vgpr, scalar_positive_inline_u32(value), arch));
      } else {
        const auto mov_value = instrumentation::build_v_mov_b32_literal(tmp_vgpr, value, arch);
        if (!mov_value)
          return false;
        words.insert(words.end(), mov_value->begin(), mov_value->end());
      }
      words.insert(words.end(), store->begin(), store->end());
      return true;
    }
    return append_dynamic_diagnostic_store_u32_literal(words, diagnostic_base + offset, value,
                                                       slot_vgpr, *options.scratch_vgpr, arch);
  };
  const auto store_vgpr = [&](size_t offset, uint16_t value_vgpr) {
    if (compact_local_diagnostic) {
      const auto store = instrumentation::build_flat_store_b32(compact_address_vgpr, value_vgpr,
                                                               arch, static_cast<uint32_t>(offset));
      if (!store)
        return false;
      words.insert(words.end(), store->begin(), store->end());
      return true;
    }
    return append_dynamic_diagnostic_store_u32_vgpr(words, diagnostic_base + offset, value_vgpr,
                                                    slot_vgpr, *options.scratch_vgpr, arch);
  };
  const auto store_scalar = [&](size_t offset, uint16_t scalar_src) {
    if (compact_local_diagnostic) {
      words.push_back(build_v_mov_b32_e32(tmp_vgpr, scalar_src, arch));
      const auto store = instrumentation::build_flat_store_b32(compact_address_vgpr, tmp_vgpr, arch,
                                                               static_cast<uint32_t>(offset));
      if (!store)
        return false;
      words.insert(words.end(), store->begin(), store->end());
      return true;
    }
    return append_dynamic_diagnostic_store_u32_scalar_src(
        words, diagnostic_base + offset, scalar_src, slot_vgpr, *options.scratch_vgpr, arch);
  };
  const bool stored_generation =
      options.moi_track_atomics
          ? options.moi_dispatch_id_sgpr &&
                store_scalar(offsetof(ConSanMoiDiagnosticRecord, generation),
                             *options.moi_dispatch_id_sgpr) &&
                store_scalar(offsetof(ConSanMoiDiagnosticRecord, generation) + sizeof(uint32_t),
                             static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u))
          : store_literal(offsetof(ConSanMoiDiagnosticRecord, generation),
                          static_cast<uint32_t>(options.moi_report_generation)) &&
                store_literal(offsetof(ConSanMoiDiagnosticRecord, generation) + sizeof(uint32_t),
                              static_cast<uint32_t>(options.moi_report_generation >> 32u));
  if (!stored_generation ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, kind),
                     static_cast<uint32_t>(ConSanMoiDiagnosticKind::AccessConflict)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, backend),
                     static_cast<uint32_t>(ConSanMoiEngine::InlineShadow)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, second_instruction_offset),
                     static_cast<uint32_t>(candidate.text_offset)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, second_access_kind),
                     static_cast<uint32_t>(current_kind)) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lane_mask), 0u) ||
      !store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lane_mask) + sizeof(uint32_t), 0u) ||
      !store_scalar(offsetof(ConSanMoiDiagnosticRecord, second_lane_mask),
                    diagnostic_lane_mask_sgpr.value_or(
                        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u))) ||
      !store_scalar(offsetof(ConSanMoiDiagnosticRecord, second_lane_mask) + sizeof(uint32_t),
                    static_cast<uint16_t>(
                        diagnostic_lane_mask_sgpr.value_or(*options.moi_exec_save_sgpr + 2u) + 1u)))
    return false;
  if (options.moi_track_atomics &&
      !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, reserved), workgroup_key))
    return false;

  if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch) ||
      !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_owner_id), tmp_vgpr) ||
      !append_extract_exact_shadow_field(words, tmp_vgpr, current_value_vgpr,
                                         consan_moi_exact_shadow::owner_shift,
                                         consan_moi_exact_shadow::max_owner, arch) ||
      !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, second_owner_id), tmp_vgpr)) {
    return false;
  }

  const auto prior_inst_shift = instrumentation::build_v_lshrrev_b32(
      tmp_vgpr,
      scalar_positive_inline_u32(compact_local_shadow
                                     ? consan_moi_exact_shadow::compact_token_shift
                                     : consan_moi_exact_shadow::instruction_offset_shift - 32u),
      compact_local_shadow ? old_value_vgpr : old_value_hi_vgpr, arch);
  const auto prior_kind = instrumentation::build_v_and_b32_literal(
      tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), old_value_vgpr,
      arch);
  if (!prior_inst_shift || !prior_kind)
    return false;
  words.push_back(*prior_inst_shift);
  if (compact_local_shadow) {
    // The compact record address has already been formed from SLOT. The slot
    // index is dead on this path, so use it to materialize CDNA4's literal
    // without clobbering the in-place token payload in TMP.
    const auto tag_token = instrumentation::build_v_add_u32_literal(
        tmp_vgpr, slot_vgpr, consan_moi_exact_shadow::compact_diagnostic_token_tag, tmp_vgpr, arch);
    if (!tag_token)
      return false;
    words.insert(words.end(), tag_token->begin(), tag_token->end());
  }
  if (!store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_instruction_offset), tmp_vgpr)) {
    return false;
  }
  words.insert(words.end(), prior_kind->begin(), prior_kind->end());
  if (!store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_access_kind), tmp_vgpr)) {
    return false;
  }
  if (has_epoch) {
    if (!append_extract_exact_shadow_field(words, tmp_vgpr, old_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch) ||
        !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_epoch), tmp_vgpr) ||
        !append_extract_exact_shadow_field(words, tmp_vgpr, current_value_vgpr,
                                           consan_moi_exact_shadow::epoch_shift,
                                           consan_moi_exact_shadow::max_epoch, arch) ||
        !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, epoch), tmp_vgpr)) {
      return false;
    }
  }

  if (prior_byte_provenance_vgpr) {
    if (static_byte_offset != 0u) {
      if (!append_compute_effective_lds_byte_offset(words, current_value_vgpr, lds_byte_offset_vgpr,
                                                    static_byte_offset, arch)) {
        return false;
      }
    } else {
      words.push_back(
          build_v_mov_b32_e32(current_value_vgpr, vector_source_vgpr(lds_byte_offset_vgpr), arch));
    }
    const auto align_cell = instrumentation::build_v_and_b32_literal(
        current_value_vgpr, ~(consan_moi_exact_shadow::granule_bytes - 1u), current_value_vgpr,
        arch);
    if (!align_cell)
      return false;
    words.insert(words.end(), align_cell->begin(), align_cell->end());
    if (relative_cell_index_vgpr) {
      // CDNA materializes the scale literal in the supplied helper VGPR. The
      // compact local path has already converted SLOT into a fixed diagnostic
      // record address, so SLOT is dead while address_lo_vgpr must remain
      // intact for all following field stores.
      const uint16_t scale_literal_vgpr =
          compact_local_diagnostic &&
                  (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
              ? slot_vgpr
              : address_lo_vgpr;
      const auto scale_cell = instrumentation::build_v_mul_lo_u32_literal(
          tmp_vgpr, scale_literal_vgpr, consan_moi_exact_shadow::granule_bytes,
          *relative_cell_index_vgpr, arch);
      const auto add_cell = instrumentation::build_v_add_u32(
          current_value_vgpr, vector_source_vgpr(tmp_vgpr), current_value_vgpr, arch);
      if (!scale_cell || !add_cell)
        return false;
      words.insert(words.end(), scale_cell->begin(), scale_cell->end());
      words.insert(words.end(), add_cell->begin(), add_cell->end());
    }

    const auto store_exact_range = [&](uint16_t provenance_vgpr, size_t offset_field,
                                       size_t count_field) {
      if (!append_extract_exact_shadow_field(
              words, current_field_vgpr, provenance_vgpr,
              consan_moi_exact_byte_cell::byte_offset_shift,
              consan_moi_low_bit_mask(consan_moi_exact_byte_cell::byte_offset_bits), arch)) {
        return false;
      }
      const auto add_start = instrumentation::build_v_add_u32(
          current_field_vgpr, vector_source_vgpr(current_value_vgpr), current_field_vgpr, arch);
      if (!add_start)
        return false;
      words.insert(words.end(), add_start->begin(), add_start->end());
      if (!store_vgpr(offset_field, current_field_vgpr) ||
          !append_extract_exact_shadow_field(
              words, tmp_vgpr, provenance_vgpr,
              consan_moi_exact_byte_cell::byte_end_minus_one_shift,
              consan_moi_low_bit_mask(consan_moi_exact_byte_cell::byte_end_minus_one_bits), arch)) {
        return false;
      }
      const auto end = instrumentation::build_v_add_u32(tmp_vgpr, scalar_positive_inline_u32(1u),
                                                        tmp_vgpr, arch);
      if (!end ||
          !append_extract_exact_shadow_field(
              words, current_field_vgpr, provenance_vgpr,
              consan_moi_exact_byte_cell::byte_offset_shift,
              consan_moi_low_bit_mask(consan_moi_exact_byte_cell::byte_offset_bits), arch)) {
        return false;
      }
      words.insert(words.end(), end->begin(), end->end());
      // CDNA4 materializes this literal in the supplied helper VGPR. The
      // compact local path keeps its diagnostic-record address in
      // address_lo_vgpr, while the already-consumed atomic slot is dead here.
      // The external path rebuilds its address for each field and still needs
      // SLOT as the dynamic diagnostic index.
      const uint16_t negative_start_literal_vgpr =
          compact_local_diagnostic &&
                  (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4)
              ? slot_vgpr
              : address_lo_vgpr;
      const auto negative_start = instrumentation::build_v_mul_lo_u32_literal(
          current_field_vgpr, negative_start_literal_vgpr, std::numeric_limits<uint32_t>::max(),
          current_field_vgpr, arch);
      const auto count = instrumentation::build_v_add_u32(
          tmp_vgpr, vector_source_vgpr(current_field_vgpr), tmp_vgpr, arch);
      if (!negative_start || !count)
        return false;
      words.insert(words.end(), negative_start->begin(), negative_start->end());
      words.insert(words.end(), count->begin(), count->end());
      return store_vgpr(count_field, tmp_vgpr);
    };
    if (!store_exact_range(*prior_byte_provenance_vgpr,
                           offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_offset),
                           offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_count)) ||
        !store_exact_range(*current_byte_provenance_vgpr,
                           offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_offset),
                           offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_count))) {
      return false;
    }
  } else {
    uint16_t diagnostic_offset_vgpr = lds_byte_offset_vgpr;
    if (static_byte_offset != 0) {
      if (!append_compute_effective_lds_byte_offset(words, current_value_vgpr, lds_byte_offset_vgpr,
                                                    static_byte_offset, arch))
        return false;
      diagnostic_offset_vgpr = current_value_vgpr;
    }
    if (!store_vgpr(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_offset),
                    diagnostic_offset_vgpr) ||
        !store_literal(offsetof(ConSanMoiDiagnosticRecord, first_lds_byte_count), byte_count) ||
        !store_vgpr(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_offset),
                    diagnostic_offset_vgpr) ||
        !store_literal(offsetof(ConSanMoiDiagnosticRecord, second_lds_byte_count), byte_count))
      return false;
  }

  const size_t restore_offset = words.size();
  for (const size_t branch_offset : empty_diagnostic_branches) {
    const int64_t delta =
        static_cast<int64_t>(restore_offset) - static_cast<int64_t>(branch_offset + 1u);
    if (delta < std::numeric_limits<int16_t>::min() || delta > std::numeric_limits<int16_t>::max())
      return false;
    const auto branch = instrumentation::build_s_cbranch_execz(static_cast<int16_t>(delta), arch);
    if (!branch)
      return false;
    words[branch_offset] = *branch;
  }

  const auto restore_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, *options.moi_exec_save_sgpr, arch);
  if (!restore_exec)
    return false;
  words.push_back(*restore_exec);
  return append_restore_moi_special_state(words, options, arch);
}

// Publish one exact slot through an odd-version reservation. EXEC on entry is
// the set of lane publishers selected by the caller. EXEC on return contains
// only successfully committed publishers whose prior slot belonged to the
// same full hardware dispatch; those are the only lanes allowed to diagnose.
// Every address group retries bounded reservation contention. Uniform groups
// publish through one representative; the outer partition loop serializes
// metadata-distinct lanes and each representative uses the same retry
// protocol. Every entering lane that still cannot commit increments
// undercoverage exactly once.
[[nodiscard]] bool append_versioned_exact_shadow_transaction(
    std::vector<uint32_t> &words, const ConSanOptions &options, rj_code_arch_t arch,
    uint16_t address_lo_vgpr, uint16_t current_low_vgpr, uint16_t old_value_vgpr,
    uint16_t publisher_exec_sgpr, uint16_t committed_exec_sgpr, bool retry_contention) {
  const bool literal_dispatch_id =
      instrumentation::is_consan_rdna_arch(arch) && !options.moi_dispatch_id_sgpr;
  if (!options.scratch_vgpr || !options.moi_exec_save_sgpr ||
      (!options.moi_dispatch_id_sgpr && !literal_dispatch_id) || !options.moi_report_buffer_address)
    return false;

  const uint16_t exec_base = *options.moi_exec_save_sgpr;
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(exec_base + 2u);
  const uint16_t empty_exec_sgpr = exec_base;
  const uint16_t ready_exec_sgpr = temporary_exec_sgpr;
  // +8:+9 retain the guest VCC across the complete probe. Keep the scalar
  // retry counter in the diagnostic-temporary region, which is consumed only
  // after the transaction has finished.
  const uint16_t retry_count_sgpr = static_cast<uint16_t>(exec_base + 20u);
  const uint16_t low_dispatch_exec_sgpr = static_cast<uint16_t>(exec_base + 18u);
  const uint16_t saved_address_lo_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t saved_address_hi_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  const uint16_t saved_current_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 4u);
  const uint16_t saved_current_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 5u);
  const uint16_t prior_dispatch_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 6u);
  const uint16_t prior_dispatch_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 7u);
  const uint16_t ready_version_vgpr = static_cast<uint16_t>(old_value_vgpr + 8u);
  const uint16_t cas_new_vgpr = static_cast<uint16_t>(old_value_vgpr + 9u);
  const uint16_t cas_expected_vgpr = consan_detail::inline_shadow_cas_expected_vgpr(old_value_vgpr);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);

  const auto narrow_vcc = [&]() -> bool {
    const auto narrow =
        instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
    if (!narrow)
      return false;
    words.push_back(*narrow);
    return true;
  };
  const auto require_equal_literal = [&](uint16_t value_vgpr, uint32_t literal) -> bool {
    const auto compare = instrumentation::build_v_cmp_eq_u32_vcc(
        scalar_positive_inline_u32(literal), value_vgpr, arch);
    if (!compare)
      return false;
    words.push_back(*compare);
    return narrow_vcc();
  };
  const auto require_not_equal_literal = [&](uint16_t value_vgpr, uint32_t literal) -> bool {
    const auto compare = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(literal), value_vgpr, arch);
    if (!compare)
      return false;
    words.push_back(*compare);
    return narrow_vcc();
  };
  const auto restore_slot_address = [&]() {
    words.push_back(
        build_v_mov_b32_e32(address_lo_vgpr, vector_source_vgpr(saved_address_lo_vgpr), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(address_lo_vgpr + 1u),
                                        vector_source_vgpr(saved_address_hi_vgpr), arch));
  };

  constexpr uint32_t kContentionRetryLimit = 2048u;
  if (retry_contention) {
    constexpr uint16_t kScalarLiteralSource = 255u;
    words.push_back(build_s_mov_b32(retry_count_sgpr, kScalarLiteralSource, arch));
    words.push_back(kContentionRetryLimit);
  }
  if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, version),
                                      ready_version_vgpr, arch)) {
    return false;
  }
  const size_t retry_begin = words.size();

  if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, packed_access),
                                      old_value_vgpr, arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, packed_access) +
                                          sizeof(uint32_t),
                                      static_cast<uint16_t>(old_value_vgpr + 1u), arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id),
                                      prior_dispatch_low_vgpr, arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) +
                                          sizeof(uint32_t),
                                      prior_dispatch_high_vgpr, arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, byte_provenance),
                                      tmp_vgpr, arch) ||
      !append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, version),
                                      cas_expected_vgpr, arch)) {
    return false;
  }

  // A changed read, odd predecessor, or terminal version is unusable. Keep
  // narrowing instead of branching so a mixed wave can still publish valid
  // independent slots.
  const auto same_version = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(ready_version_vgpr), cas_expected_vgpr, arch);
  const auto parity =
      instrumentation::build_v_and_b32_literal(cas_new_vgpr, 1u, ready_version_vgpr, arch);
  const auto max_ready = instrumentation::build_v_mov_b32_literal(
      cas_new_vgpr, kConSanMoiInlineExactMaxReadyVersion, arch);
  const auto below_terminal = instrumentation::build_v_cmp_ne_u32_vcc(
      vector_source_vgpr(cas_new_vgpr), ready_version_vgpr, arch);
  if (!same_version || !parity || !max_ready || !below_terminal)
    return false;
  words.push_back(*same_version);
  if (!narrow_vcc())
    return false;
  words.insert(words.end(), parity->begin(), parity->end());
  if (!require_equal_literal(cas_new_vgpr, 0))
    return false;
  words.insert(words.end(), max_ready->begin(), max_ready->end());
  words.push_back(*below_terminal);
  if (!narrow_vcc())
    return false;
  const auto save_stable_exec =
      instrumentation::build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
  if (!save_stable_exec)
    return false;
  words.push_back(*save_stable_exec);

  // Version zero is empty only when every payload word is zero.
  if (!require_equal_literal(ready_version_vgpr, 0) || !require_equal_literal(old_value_vgpr, 0) ||
      !require_equal_literal(static_cast<uint16_t>(old_value_vgpr + 1u), 0) ||
      !require_equal_literal(prior_dispatch_low_vgpr, 0) ||
      !require_equal_literal(prior_dispatch_high_vgpr, 0) || !require_equal_literal(tmp_vgpr, 0)) {
    return false;
  }
  const auto save_empty_exec =
      instrumentation::build_s_mov_b64(empty_exec_sgpr, kRdna4ExecLo, arch);
  const auto restore_stable_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, committed_exec_sgpr, arch);
  if (!save_empty_exec || !restore_stable_exec)
    return false;
  words.push_back(*save_empty_exec);
  words.push_back(*restore_stable_exec);

  // A nonempty ready predecessor must itself be structurally valid. Owner and
  // epoch zero are legal packed values; access kind, workgroup key, dispatch,
  // and exact-byte provenance are the discriminants.
  if (!require_not_equal_literal(ready_version_vgpr, 0) ||
      !require_not_equal_literal(tmp_vgpr, 0)) {
    return false;
  }
  const auto provenance_reserved = instrumentation::build_v_and_b32_literal(
      cas_new_vgpr, ~consan_moi_exact_byte_cell::packed_mask, tmp_vgpr, arch);
  const auto provenance_mask = instrumentation::build_v_and_b32_literal(
      cas_new_vgpr, consan_moi_exact_byte_cell::byte_mask_mask, tmp_vgpr, arch);
  if (!provenance_reserved || !provenance_mask)
    return false;
  words.insert(words.end(), provenance_reserved->begin(), provenance_reserved->end());
  if (!require_equal_literal(cas_new_vgpr, 0))
    return false;
  words.insert(words.end(), provenance_mask->begin(), provenance_mask->end());
  for (uint32_t byte_mask = 0; byte_mask <= consan_moi_exact_byte_cell::byte_mask_mask;
       ++byte_mask) {
    if (!consan_moi_exact_byte_cell_mask_is_contiguous(byte_mask) &&
        !require_not_equal_literal(cas_new_vgpr, byte_mask)) {
      return false;
    }
  }
  const auto require_canonical_provenance_field = [&](uint32_t field_shift, uint32_t field_lookup) {
    const auto mask = instrumentation::build_v_and_b32_literal(
        cas_new_vgpr, consan_moi_exact_byte_cell::byte_mask_mask, tmp_vgpr, arch);
    const auto scale_mask = instrumentation::build_v_lshlrev_b32(
        cas_new_vgpr, scalar_positive_inline_u32(1u), cas_new_vgpr, arch);
    const auto lookup =
        instrumentation::build_v_mov_b32_literal(cas_expected_vgpr, field_lookup, arch);
    const auto select_expected = instrumentation::build_v_lshrrev_b32(
        cas_expected_vgpr, vector_source_vgpr(cas_new_vgpr), cas_expected_vgpr, arch);
    const auto bound_expected =
        instrumentation::build_v_and_b32_literal(cas_expected_vgpr, 0x3u, cas_expected_vgpr, arch);
    const auto unpack_actual = instrumentation::build_v_lshrrev_b32(
        cas_new_vgpr, scalar_positive_inline_u32(field_shift), tmp_vgpr, arch);
    const auto bound_actual =
        instrumentation::build_v_and_b32_literal(cas_new_vgpr, 0x3u, cas_new_vgpr, arch);
    const auto matches = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(cas_expected_vgpr), cas_new_vgpr, arch);
    if (!mask || !scale_mask || !lookup || !select_expected || !bound_expected || !unpack_actual ||
        !bound_actual || !matches) {
      return false;
    }
    words.insert(words.end(), mask->begin(), mask->end());
    words.push_back(*scale_mask);
    words.insert(words.end(), lookup->begin(), lookup->end());
    words.push_back(*select_expected);
    words.insert(words.end(), bound_expected->begin(), bound_expected->end());
    words.push_back(*unpack_actual);
    words.insert(words.end(), bound_actual->begin(), bound_actual->end());
    words.push_back(*matches);
    return narrow_vcc();
  };
  if (!require_canonical_provenance_field(consan_moi_exact_byte_cell::byte_offset_shift,
                                          consan_moi_exact_byte_cell::byte_offset_by_mask_lookup) ||
      !require_canonical_provenance_field(
          consan_moi_exact_byte_cell::byte_end_minus_one_shift,
          consan_moi_exact_byte_cell::byte_end_minus_one_by_mask_lookup)) {
    return false;
  }
  const auto provenance_lane = instrumentation::build_v_and_b32_literal(
      cas_new_vgpr, consan_moi_exact_byte_cell::lane_plus_one_mask, tmp_vgpr, arch);
  const auto unpack_provenance_lane = instrumentation::build_v_lshrrev_b32(
      cas_new_vgpr, scalar_positive_inline_u32(consan_moi_exact_byte_cell::lane_plus_one_shift),
      cas_new_vgpr, arch);
  if (!provenance_lane || !unpack_provenance_lane)
    return false;
  words.insert(words.end(), provenance_lane->begin(), provenance_lane->end());
  if (!require_not_equal_literal(cas_new_vgpr, 0))
    return false;
  words.push_back(*unpack_provenance_lane);
  const auto zero_based_lane = instrumentation::build_v_add_u32_literal(
      cas_new_vgpr, tmp_vgpr, std::numeric_limits<uint32_t>::max(), cas_new_vgpr, arch);
  if (!zero_based_lane)
    return false;
  words.insert(words.end(), zero_based_lane->begin(), zero_based_lane->end());
  const auto provenance_lane_in_range = instrumentation::build_v_cmp_gt_u32_vcc(
      scalar_positive_inline_u32(consan_moi_exact_byte_cell::maximum_lane + 1u), cas_new_vgpr,
      arch);
  if (!provenance_lane_in_range)
    return false;
  words.push_back(*provenance_lane_in_range);
  if (!narrow_vcc())
    return false;
  const auto kind = instrumentation::build_v_and_b32_literal(
      tmp_vgpr, static_cast<uint32_t>(consan_moi_exact_shadow::access_kind_mask), old_value_vgpr,
      arch);
  if (!kind)
    return false;
  words.insert(words.end(), kind->begin(), kind->end());
  if (!require_not_equal_literal(tmp_vgpr, 0))
    return false;
  const auto kind_in_range = instrumentation::build_v_cmp_gt_u32_vcc(
      scalar_positive_inline_u32(static_cast<uint32_t>(ConSanMoiShadowAccessKind::Atomic) + 1u),
      tmp_vgpr, arch);
  if (!kind_in_range)
    return false;
  words.push_back(*kind_in_range);
  if (!narrow_vcc() ||
      !append_extract_exact_shadow_generation(words, tmp_vgpr, old_value_vgpr,
                                              static_cast<uint16_t>(old_value_vgpr + 1u),
                                              cas_new_vgpr, arch) ||
      !require_not_equal_literal(tmp_vgpr, 0)) {
    return false;
  }
  const auto save_ready_fields_exec =
      instrumentation::build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
  if (!save_ready_fields_exec)
    return false;
  words.push_back(*save_ready_fields_exec);

  // Form the nonzero 64-bit dispatch predicate without truncating it: the low
  // nonzero and low-zero/high-nonzero masks are disjoint, so XOR is their union.
  if (!require_not_equal_literal(prior_dispatch_low_vgpr, 0))
    return false;
  const auto save_low_dispatch_exec =
      instrumentation::build_s_mov_b64(low_dispatch_exec_sgpr, kRdna4ExecLo, arch);
  const auto restore_ready_fields_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, committed_exec_sgpr, arch);
  if (!save_low_dispatch_exec || !restore_ready_fields_exec)
    return false;
  words.push_back(*save_low_dispatch_exec);
  words.push_back(*restore_ready_fields_exec);
  if (!require_equal_literal(prior_dispatch_low_vgpr, 0) ||
      !require_not_equal_literal(prior_dispatch_high_vgpr, 0)) {
    return false;
  }
  const auto save_high_dispatch_exec =
      instrumentation::build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
  const auto union_ready_dispatch = instrumentation::build_s_xor_b64(
      ready_exec_sgpr, low_dispatch_exec_sgpr, committed_exec_sgpr, arch);
  const auto union_valid =
      instrumentation::build_s_xor_b64(kRdna4ExecLo, empty_exec_sgpr, ready_exec_sgpr, arch);
  if (!save_high_dispatch_exec || !union_ready_dispatch || !union_valid)
    return false;
  words.push_back(*save_high_dispatch_exec);
  words.push_back(*union_ready_dispatch);
  words.push_back(*union_valid);

  const auto claim_new = instrumentation::build_v_add_u32(
      cas_new_vgpr, scalar_positive_inline_u32(1), ready_version_vgpr, arch);
  const uint32_t claim_expected =
      build_v_mov_b32_e32(cas_expected_vgpr, vector_source_vgpr(ready_version_vgpr), arch);
  if (!claim_new || !append_add_literal_field(words, address_lo_vgpr,
                                              offsetof(ConSanMoiInlineExactShadowSlot, version),
                                              tmp_vgpr, arch)) {
    return false;
  }
  words.insert(words.end(), claim_new->begin(), claim_new->end());
  words.push_back(claim_expected);
  const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(
      address_lo_vgpr, cas_new_vgpr, cas_new_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
      arch);
  if (!claim)
    return false;
  words.insert(words.end(), claim->begin(), claim->end());
  if (!append_moi_global_atomic_wait(words, arch))
    return false;
  const auto claim_won = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(cas_expected_vgpr), cas_new_vgpr, arch);
  if (!claim_won)
    return false;
  words.push_back(*claim_won);
  if (!narrow_vcc())
    return false;
  if (retry_contention) {
    const auto save_claimed =
        instrumentation::build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
    const auto select_failed = instrumentation::build_s_andn2_b64(kRdna4ExecLo, publisher_exec_sgpr,
                                                                  committed_exec_sgpr, arch);
    const auto decrement = instrumentation::build_s_sub_u32(retry_count_sgpr, retry_count_sgpr,
                                                            scalar_positive_inline_u32(1), arch);
    const auto retries_remain =
        instrumentation::build_s_cmp_lg_u32(retry_count_sgpr, scalar_positive_inline_u32(0), arch);
    const auto restore_claimed =
        instrumentation::build_s_mov_b64(kRdna4ExecLo, committed_exec_sgpr, arch);
    if (!save_claimed || !select_failed || !decrement || !retries_remain || !restore_claimed) {
      return false;
    }
    words.push_back(*save_claimed);
    words.push_back(*select_failed);
    // Refresh the version only on contending lanes. The ordinary two-read
    // snapshot is substantially cheaper on the successful path. Feed this
    // coherent device-scope value directly into the next retry rather than
    // discarding it and beginning with another potentially stale flat load.
    restore_slot_address();
    if (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) {
      const auto invalidate = arch == ROCJITSU_CODE_ARCH_CDNA3 ? build_cdna3_buffer_inv_sc1(arch)
                                                               : build_cdna4_buffer_inv_sc1(arch);
      if (!invalidate)
        return false;
      words.insert(words.end(), invalidate->begin(), invalidate->end());
    }
    if (!append_add_literal_field(words, address_lo_vgpr,
                                  offsetof(ConSanMoiInlineExactShadowSlot, version), tmp_vgpr,
                                  arch) ||
        !append_atomic_load_u32(words, address_lo_vgpr, ready_version_vgpr, arch)) {
      return false;
    }
    restore_slot_address();
    words.push_back(build_s_sleep(/*delay=*/1, arch));
    words.push_back(*decrement);
    words.push_back(*retries_remain);
    restore_slot_address();
    const auto retries_exhausted = instrumentation::build_s_cbranch_scc0(/*simm16=*/1, arch);
    if (!retries_exhausted)
      return false;
    words.push_back(*retries_exhausted);
    const size_t retry_branch = words.size();
    const int64_t retry_delta =
        static_cast<int64_t>(retry_begin) - static_cast<int64_t>(retry_branch + 1u);
    if (retry_delta < std::numeric_limits<int16_t>::min() ||
        retry_delta > std::numeric_limits<int16_t>::max()) {
      return false;
    }
    const auto retry =
        instrumentation::build_s_cbranch_execnz(static_cast<int16_t>(retry_delta), arch);
    if (!retry)
      return false;
    words.push_back(*retry);
    words.push_back(*restore_claimed);
  }

  // Qualify the predecessor's complete dispatch identity while its payload
  // registers are still live. The resulting scalar mask survives publication;
  // the vector dispatch pair can then retain exact-byte provenance for the
  // diagnostic path without growing the scratch window.
  const auto save_claimed_publishers =
      instrumentation::build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
  if (!save_claimed_publishers)
    return false;
  words.push_back(*save_claimed_publishers);
  uint16_t dispatch_low_source = 0;
  uint16_t dispatch_high_source = 0;
  if (options.moi_dispatch_id_sgpr) {
    dispatch_low_source = *options.moi_dispatch_id_sgpr;
    dispatch_high_source = static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u);
  } else {
    const auto low = instrumentation::build_v_mov_b32_literal(
        tmp_vgpr, static_cast<uint32_t>(options.moi_report_dispatch_id), arch);
    if (!low)
      return false;
    words.insert(words.end(), low->begin(), low->end());
    dispatch_low_source = vector_source_vgpr(tmp_vgpr);
  }
  const auto same_dispatch_low =
      instrumentation::build_v_cmp_eq_u32_vcc(dispatch_low_source, prior_dispatch_low_vgpr, arch);
  if (!same_dispatch_low)
    return false;
  words.push_back(*same_dispatch_low);
  if (!narrow_vcc())
    return false;
  if (!options.moi_dispatch_id_sgpr) {
    const auto high = instrumentation::build_v_mov_b32_literal(
        tmp_vgpr, static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u), arch);
    if (!high)
      return false;
    words.insert(words.end(), high->begin(), high->end());
    dispatch_high_source = vector_source_vgpr(tmp_vgpr);
  }
  const auto same_dispatch_high =
      instrumentation::build_v_cmp_eq_u32_vcc(dispatch_high_source, prior_dispatch_high_vgpr, arch);
  const auto save_same_dispatch =
      instrumentation::build_s_mov_b64(low_dispatch_exec_sgpr, kRdna4ExecLo, arch);
  const auto restore_claimed_publishers =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, committed_exec_sgpr, arch);
  if (!same_dispatch_high || !save_same_dispatch || !restore_claimed_publishers)
    return false;
  words.push_back(*same_dispatch_high);
  if (!narrow_vcc())
    return false;
  words.push_back(*save_same_dispatch);
  words.push_back(*restore_claimed_publishers);

  restore_slot_address();
  if (!append_load_u32_vgpr_at_offset(words, address_lo_vgpr,
                                      offsetof(ConSanMoiInlineExactShadowSlot, byte_provenance),
                                      prior_dispatch_low_vgpr, arch)) {
    return false;
  }
  words.push_back(
      build_v_mov_b32_e32(prior_dispatch_high_vgpr, vector_source_vgpr(current_low_vgpr), arch));
  restore_slot_address();

  if (!append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, packed_access),
                                       saved_current_low_vgpr, arch) ||
      !append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, packed_access) +
                                           sizeof(uint32_t),
                                       saved_current_high_vgpr, arch)) {
    return false;
  }
  if (!append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, byte_provenance),
                                       current_low_vgpr, arch)) {
    return false;
  }
  if (options.moi_dispatch_id_sgpr) {
    words.push_back(build_v_mov_b32_e32(current_low_vgpr, *options.moi_dispatch_id_sgpr, arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(current_low_vgpr + 1u),
                                        static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u),
                                        arch));
  } else {
    const auto low = instrumentation::build_v_mov_b32_literal(
        current_low_vgpr, static_cast<uint32_t>(options.moi_report_dispatch_id), arch);
    const auto high = instrumentation::build_v_mov_b32_literal(
        static_cast<uint16_t>(current_low_vgpr + 1u),
        static_cast<uint32_t>(options.moi_report_dispatch_id >> 32u), arch);
    if (!low || !high)
      return false;
    words.insert(words.end(), low->begin(), low->end());
    words.insert(words.end(), high->begin(), high->end());
  }
  if (!append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id),
                                       current_low_vgpr, arch) ||
      !append_store_u32_vgpr_at_offset(words, address_lo_vgpr,
                                       offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) +
                                           sizeof(uint32_t),
                                       static_cast<uint16_t>(current_low_vgpr + 1u), arch)) {
    return false;
  }
  const auto store_drain = instrumentation::build_s_wait_global_store0(arch);
  if (!store_drain)
    return false;
  words.push_back(*store_drain);

  const auto commit_new = instrumentation::build_v_add_u32(
      cas_new_vgpr, scalar_positive_inline_u32(2), ready_version_vgpr, arch);
  const auto commit_expected = instrumentation::build_v_add_u32(
      cas_expected_vgpr, scalar_positive_inline_u32(1), ready_version_vgpr, arch);
  if (!commit_new || !commit_expected ||
      !append_add_literal_field(words, address_lo_vgpr,
                                offsetof(ConSanMoiInlineExactShadowSlot, version), tmp_vgpr,
                                arch)) {
    return false;
  }
  words.insert(words.end(), commit_new->begin(), commit_new->end());
  words.insert(words.end(), commit_expected->begin(), commit_expected->end());
  const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
      address_lo_vgpr, cas_new_vgpr, cas_new_vgpr, /*return_old_value=*/true, kRdna4ScopeDevice,
      arch);
  if (!commit)
    return false;
  words.insert(words.end(), commit->begin(), commit->end());
  if (!append_moi_global_atomic_wait(words, arch))
    return false;
  const auto commit_won = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(cas_expected_vgpr), cas_new_vgpr, arch);
  if (!commit_won)
    return false;
  words.push_back(*commit_won);
  if (!narrow_vcc())
    return false;
  const auto save_committed =
      instrumentation::build_s_mov_b64(committed_exec_sgpr, kRdna4ExecLo, arch);
  if (!save_committed)
    return false;
  words.push_back(*save_committed);

  // Count successful publications before excluding cross-dispatch priors from
  // diagnosis. The event counter therefore remains a successful-commit count.
  if (!append_atomic_fetch_add_one_u32(words,
                                       *options.moi_report_buffer_address +
                                           offsetof(ConSanMoiReportHeader, event_counter),
                                       tmp_vgpr, address_lo_vgpr, arch)) {
    return false;
  }
  restore_slot_address();

  const auto restore_publishers =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, publisher_exec_sgpr, arch);
  const auto select_failed = instrumentation::build_s_andn2_b64(kRdna4ExecLo, publisher_exec_sgpr,
                                                                committed_exec_sgpr, arch);
  const auto restore_committed =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, committed_exec_sgpr, arch);
  if (!restore_publishers || !select_failed || !restore_committed)
    return false;
  words.push_back(*restore_publishers);
  words.push_back(*select_failed);
  if (!append_atomic_fetch_add_one_u32(
          words,
          *options.moi_report_buffer_address +
              offsetof(ConSanMoiReportHeader, inline_undercoverage_count),
          tmp_vgpr, address_lo_vgpr, arch)) {
    return false;
  }
  words.push_back(*restore_committed);
  restore_slot_address();
  words.push_back(
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch));
  words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(current_low_vgpr + 1u),
                                      vector_source_vgpr(saved_current_high_vgpr), arch));
  const auto select_same_dispatch = instrumentation::build_s_and_b64(
      kRdna4ExecLo, committed_exec_sgpr, low_dispatch_exec_sgpr, arch);
  if (!select_same_dispatch)
    return false;
  words.push_back(*select_same_dispatch);
  return true;
}

// Build the target-neutral exact-byte provenance for one four-byte shadow
// cell. The caller supplies the relative cell index when a wide access is
// traversed by a runtime loop. MASK remains separate for wave partitioning;
// PROVENANCE carries the redundant range endpoints used by diagnostics and
// coherent snapshot validation. The representative lane is added only after
// the caller elects the publishing lane.
[[nodiscard]] bool append_exact_byte_cell_provenance_base(
    std::vector<uint32_t> &words, rj_code_arch_t arch, uint16_t base_offset_vgpr,
    uint32_t static_byte_offset, uint32_t relative_cell_index,
    std::optional<uint16_t> relative_cell_index_vgpr, uint32_t byte_count, uint16_t provenance_vgpr,
    uint16_t mask_vgpr, uint16_t high_vgpr, uint16_t temporary_vgpr) {
  if (byte_count == 0u)
    return false;

  if (static_byte_offset != 0u) {
    const auto effective_start = instrumentation::build_v_add_u32_literal(
        mask_vgpr, provenance_vgpr,
        static_byte_offset & (consan_moi_exact_shadow::granule_bytes - 1u), base_offset_vgpr, arch);
    if (!effective_start)
      return false;
    words.insert(words.end(), effective_start->begin(), effective_start->end());
  } else {
    words.push_back(build_v_mov_b32_e32(mask_vgpr, vector_source_vgpr(base_offset_vgpr), arch));
  }
  const auto start = instrumentation::build_v_and_b32_literal(
      mask_vgpr, consan_moi_exact_shadow::granule_bytes - 1u, mask_vgpr, arch);
  const auto access_end = instrumentation::build_v_add_u32_literal(high_vgpr, temporary_vgpr,
                                                                   byte_count, mask_vgpr, arch);
  if (!start || !access_end)
    return false;
  words.insert(words.end(), start->begin(), start->end());
  words.insert(words.end(), access_end->begin(), access_end->end());

  if (relative_cell_index_vgpr) {
    const auto cell_bytes = instrumentation::build_v_mul_lo_u32_literal(
        temporary_vgpr, provenance_vgpr, consan_moi_exact_shadow::granule_bytes,
        *relative_cell_index_vgpr, arch);
    if (!cell_bytes)
      return false;
    words.insert(words.end(), cell_bytes->begin(), cell_bytes->end());
    // The loop bound covers the worst runtime alignment. Clamp an iteration's
    // cell start to the access end before subtraction so favorable alignments
    // produce an empty mask instead of wrapping to a phantom full cell.
    const auto clamped_cell_bytes = instrumentation::build_v_min_u32(
        temporary_vgpr, vector_source_vgpr(high_vgpr), temporary_vgpr, arch);
    const auto negative_cell_bytes = instrumentation::build_v_mul_lo_u32_literal(
        temporary_vgpr, provenance_vgpr, std::numeric_limits<uint32_t>::max(), temporary_vgpr,
        arch);
    if (!clamped_cell_bytes || !negative_cell_bytes)
      return false;
    words.push_back(*clamped_cell_bytes);
    words.insert(words.end(), negative_cell_bytes->begin(), negative_cell_bytes->end());
    const auto remaining = instrumentation::build_v_add_u32(
        high_vgpr, vector_source_vgpr(temporary_vgpr), high_vgpr, arch);
    const auto first_cell = instrumentation::build_v_min_u32_literal(
        temporary_vgpr, 1u, *relative_cell_index_vgpr, arch);
    const auto low_shift = instrumentation::build_v_lshlrev_b32(
        temporary_vgpr, scalar_positive_inline_u32(1u), temporary_vgpr, arch);
    if (!remaining || !first_cell || !low_shift)
      return false;
    words.insert(words.end(), remaining->begin(), remaining->end());
    words.insert(words.end(), first_cell->begin(), first_cell->end());
    words.push_back(*low_shift);
    const auto low = instrumentation::build_v_lshrrev_b32(
        temporary_vgpr, vector_source_vgpr(temporary_vgpr), mask_vgpr, arch);
    if (!low)
      return false;
    words.push_back(*low);
  } else if (relative_cell_index != 0u) {
    if (relative_cell_index >
        std::numeric_limits<uint32_t>::max() / consan_moi_exact_shadow::granule_bytes) {
      return false;
    }
    const auto cell_bytes = instrumentation::build_v_mov_b32_literal(
        temporary_vgpr, relative_cell_index * consan_moi_exact_shadow::granule_bytes, arch);
    const auto clamped_cell_bytes = instrumentation::build_v_min_u32(
        temporary_vgpr, vector_source_vgpr(high_vgpr), temporary_vgpr, arch);
    const auto negative_cell_bytes = instrumentation::build_v_mul_lo_u32_literal(
        temporary_vgpr, provenance_vgpr, std::numeric_limits<uint32_t>::max(), temporary_vgpr,
        arch);
    const auto remaining = instrumentation::build_v_add_u32(
        high_vgpr, vector_source_vgpr(temporary_vgpr), high_vgpr, arch);
    if (!cell_bytes || !clamped_cell_bytes || !negative_cell_bytes || !remaining)
      return false;
    words.insert(words.end(), cell_bytes->begin(), cell_bytes->end());
    words.push_back(*clamped_cell_bytes);
    words.insert(words.end(), negative_cell_bytes->begin(), negative_cell_bytes->end());
    words.insert(words.end(), remaining->begin(), remaining->end());
    words.push_back(build_v_mov_b32_e32(temporary_vgpr, scalar_positive_inline_u32(0u), arch));
  } else {
    words.push_back(build_v_mov_b32_e32(temporary_vgpr, vector_source_vgpr(mask_vgpr), arch));
  }

  const auto high = instrumentation::build_v_min_u32_literal(
      high_vgpr, consan_moi_exact_shadow::granule_bytes, high_vgpr, arch);
  if (!high)
    return false;
  words.insert(words.end(), high->begin(), high->end());

  const auto one = instrumentation::build_v_mov_b32_literal(provenance_vgpr, 1u, arch);
  const auto low_bits = instrumentation::build_v_lshlrev_b32(
      provenance_vgpr, vector_source_vgpr(high_vgpr), provenance_vgpr, arch);
  const auto low_bits_minus_one = instrumentation::build_v_add_u32_literal(
      provenance_vgpr, mask_vgpr, std::numeric_limits<uint32_t>::max(), provenance_vgpr, arch);
  const auto four_bits = instrumentation::build_v_mov_b32_literal(mask_vgpr, 0xfu, arch);
  const auto high_bits = instrumentation::build_v_lshlrev_b32(
      mask_vgpr, vector_source_vgpr(temporary_vgpr), mask_vgpr, arch);
  const auto exact_mask = instrumentation::build_v_and_b32(
      mask_vgpr, vector_source_vgpr(provenance_vgpr), mask_vgpr, arch);
  if (!one || !low_bits || !low_bits_minus_one || !four_bits || !high_bits || !exact_mask)
    return false;
  words.insert(words.end(), one->begin(), one->end());
  words.push_back(*low_bits);
  words.insert(words.end(), low_bits_minus_one->begin(), low_bits_minus_one->end());
  words.insert(words.end(), four_bits->begin(), four_bits->end());
  words.push_back(*high_bits);
  words.push_back(*exact_mask);

  const auto end_minus_one = instrumentation::build_v_add_u32_literal(
      high_vgpr, provenance_vgpr, std::numeric_limits<uint32_t>::max(), high_vgpr, arch);
  const auto pack_end = instrumentation::build_v_lshlrev_b32(
      high_vgpr, scalar_positive_inline_u32(consan_moi_exact_byte_cell::byte_end_minus_one_shift),
      high_vgpr, arch);
  const auto pack_start = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(consan_moi_exact_byte_cell::byte_offset_shift),
      temporary_vgpr, arch);
  const auto pack_range = instrumentation::build_v_add_u32(
      provenance_vgpr, vector_source_vgpr(high_vgpr), temporary_vgpr, arch);
  const auto pack_mask = instrumentation::build_v_add_u32(
      provenance_vgpr, vector_source_vgpr(mask_vgpr), provenance_vgpr, arch);
  if (!end_minus_one || !pack_end || !pack_start || !pack_range || !pack_mask)
    return false;
  words.insert(words.end(), end_minus_one->begin(), end_minus_one->end());
  words.push_back(*pack_end);
  words.push_back(*pack_start);
  words.insert(words.end(), pack_range->begin(), pack_range->end());
  words.insert(words.end(), pack_mask->begin(), pack_mask->end());
  return true;
}

[[nodiscard]] bool append_exact_byte_representative_lane(std::vector<uint32_t> &words,
                                                         rj_code_arch_t arch,
                                                         uint16_t provenance_vgpr,
                                                         uint16_t lane_vgpr) {
  const auto lane_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      lane_vgpr, kScalarInlineNegativeOneOperand, scalar_positive_inline_u32(0), arch);
  const auto lane_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      lane_vgpr, kScalarInlineNegativeOneOperand, vector_source_vgpr(lane_vgpr), arch);
  const auto lane_plus_one =
      instrumentation::build_v_add_u32(lane_vgpr, scalar_positive_inline_u32(1u), lane_vgpr, arch);
  const auto pack_lane = instrumentation::build_v_lshlrev_b32(
      lane_vgpr, scalar_positive_inline_u32(consan_moi_exact_byte_cell::lane_plus_one_shift),
      lane_vgpr, arch);
  const auto add_lane = instrumentation::build_v_add_u32(
      provenance_vgpr, vector_source_vgpr(lane_vgpr), provenance_vgpr, arch);
  if (!lane_lo || !lane_hi || !lane_plus_one || !pack_lane || !add_lane)
    return false;
  words.insert(words.end(), lane_lo->begin(), lane_lo->end());
  words.insert(words.end(), lane_hi->begin(), lane_hi->end());
  words.insert(words.end(), lane_plus_one->begin(), lane_plus_one->end());
  words.push_back(*pack_lane);
  words.insert(words.end(), add_lane->begin(), add_lane->end());
  return true;
}

// Partition the incoming wave by effective exact-shadow address. Each address
// group with uniform metadata and exact byte coverage publishes through one
// representative lane. A group whose metadata is not uniform submits every
// lane to the versioned transaction rather than selecting arbitrary evidence;
// one CAS winner may publish and every loser poisons coverage. The loop removes
// at least one lane on every iteration, so runtime work is bounded by the
// hardware wave size while generated code size remains constant.
[[nodiscard]] bool append_wave_coalesced_exact_shadow_swap(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate, const ConSanOptions &options,
    rj_code_arch_t arch, const ConSanMoiReportBufferLayout &layout, uint16_t address_lo_vgpr,
    uint16_t current_low_vgpr, uint16_t current_high_vgpr, uint16_t old_value_vgpr,
    uint16_t lds_byte_offset_vgpr, uint32_t static_byte_offset, uint32_t byte_count,
    uint32_t relative_cell_index, std::optional<uint16_t> relative_cell_index_vgpr,
    const VgprSpillSequence *spill, std::optional<uint16_t> spill_backed_lds_byte_offset_source,
    std::vector<std::string> &errors) {
  if (!options.moi_exec_save_sgpr)
    return false;

  const uint16_t base = *options.moi_exec_save_sgpr;
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(base + 2u);
  const uint16_t publisher_exec_sgpr = static_cast<uint16_t>(base + 4u);
  const uint16_t committed_exec_sgpr = static_cast<uint16_t>(base + 6u);
  const uint16_t incoming_exec_sgpr = static_cast<uint16_t>(base + 12u);
  const uint16_t pending_exec_sgpr = static_cast<uint16_t>(base + 14u);
  const uint16_t group_exec_sgpr = static_cast<uint16_t>(base + 16u);
  const uint16_t address_key_sgpr = static_cast<uint16_t>(base + 18u);
  const uint16_t value_key_sgpr = static_cast<uint16_t>(base + 19u);
  const uint16_t lane_rank_vgpr = static_cast<uint16_t>(old_value_vgpr + 1u);
  const uint16_t saved_address_lo_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t saved_address_hi_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  const uint16_t saved_current_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 4u);
  const uint16_t saved_current_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 5u);

  const auto save_incoming_exec =
      instrumentation::build_s_mov_b64(incoming_exec_sgpr, kRdna4ExecLo, arch);
  const auto initialize_pending =
      instrumentation::build_s_mov_b64(pending_exec_sgpr, kRdna4ExecLo, arch);
  const auto select_pending =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, pending_exec_sgpr, arch);
  const uint32_t save_address_lo =
      build_v_mov_b32_e32(saved_address_lo_vgpr, vector_source_vgpr(address_lo_vgpr), arch);
  const uint32_t save_address_hi = build_v_mov_b32_e32(
      saved_address_hi_vgpr, vector_source_vgpr(static_cast<uint16_t>(address_lo_vgpr + 1u)), arch);
  const uint32_t save_current_low =
      build_v_mov_b32_e32(saved_current_low_vgpr, vector_source_vgpr(current_low_vgpr), arch);
  const uint32_t save_current_high =
      build_v_mov_b32_e32(saved_current_high_vgpr, vector_source_vgpr(current_high_vgpr), arch);
  const uint32_t restore_address_lo =
      build_v_mov_b32_e32(address_lo_vgpr, vector_source_vgpr(saved_address_lo_vgpr), arch);
  const uint32_t restore_address_hi = build_v_mov_b32_e32(
      static_cast<uint16_t>(address_lo_vgpr + 1u), vector_source_vgpr(saved_address_hi_vgpr), arch);
  const uint32_t restore_current_low =
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch);
  const uint32_t restore_current_high =
      build_v_mov_b32_e32(current_high_vgpr, vector_source_vgpr(saved_current_high_vgpr), arch);
  const auto read_address =
      instrumentation::build_v_readfirstlane_b32(address_key_sgpr, address_lo_vgpr, arch);
  const auto read_address_high = instrumentation::build_v_readfirstlane_b32(
      value_key_sgpr, static_cast<uint16_t>(address_lo_vgpr + 1u), arch);
  const auto read_value =
      instrumentation::build_v_readfirstlane_b32(value_key_sgpr, current_low_vgpr, arch);
  const auto address_equal =
      instrumentation::build_v_cmp_eq_u32_vcc(address_key_sgpr, address_lo_vgpr, arch);
  const auto narrow_address =
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto save_group = instrumentation::build_s_mov_b64(group_exec_sgpr, kRdna4ExecLo, arch);
  const auto value_equal =
      instrumentation::build_v_cmp_eq_u32_vcc(value_key_sgpr, current_low_vgpr, arch);
  const auto narrow_value =
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto read_mask =
      instrumentation::build_v_readfirstlane_b32(value_key_sgpr, lane_rank_vgpr, arch);
  const auto mask_equal =
      instrumentation::build_v_cmp_eq_u32_vcc(value_key_sgpr, lane_rank_vgpr, arch);
  const auto narrow_mask =
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto low_mask_equal =
      instrumentation::build_s_cmp_eq_u32(kRdna4ExecLo, group_exec_sgpr, arch);
  const auto high_mask_equal = instrumentation::build_s_cmp_eq_u32(
      static_cast<uint16_t>(kRdna4ExecLo + 1u), static_cast<uint16_t>(group_exec_sgpr + 1u), arch);
  if (!save_incoming_exec || !initialize_pending || !select_pending || !read_address ||
      !read_address_high || !read_value || !address_equal || !narrow_address || !save_group ||
      !value_equal || !narrow_value || !read_mask || !mask_equal || !narrow_mask ||
      !low_mask_equal || !high_mask_equal)
    return false;

  words.push_back(*save_incoming_exec);
  words.push_back(*initialize_pending);
  words.push_back(save_address_lo);
  words.push_back(save_address_hi);
  words.push_back(save_current_low);
  words.push_back(save_current_high);
  if (options.moi_partition_mask_debug) {
    // Capture the first incoming lane's address before loop temporaries can
    // reuse these SGPRs. InlineShadow leaves all three record counts at zero,
    // so the debug reader can use count fields without corrupting the layout
    // capacities that locate the exact-shadow array.
    const uint64_t report_base = *options.moi_report_buffer_address;
    words.push_back(*read_address);
    words.push_back(*read_address_high);
    if (!append_store_u32_sgpr(words,
                               report_base + offsetof(ConSanMoiReportHeader, access_record_count),
                               address_key_sgpr, current_low_vgpr, address_lo_vgpr, arch) ||
        !append_store_u32_sgpr(words,
                               report_base + offsetof(ConSanMoiReportHeader, barrier_record_count),
                               value_key_sgpr, current_low_vgpr, address_lo_vgpr, arch))
      return false;
  }
  const size_t loop = words.size();
  words.push_back(*select_pending);
  words.push_back(restore_address_lo);
  words.push_back(restore_address_hi);
  words.push_back(restore_current_low);
  words.push_back(restore_current_high);
  words.push_back(*read_address);
  words.push_back(*read_value);
  words.push_back(*address_equal);
  words.push_back(*narrow_address);
  words.push_back(*save_group);
  if (options.moi_partition_mask_debug) {
    // Capture the current group's exchange address before diagnostics can
    // reuse working VGPRs. Each group overwrites this pair, matching the
    // retained final_group mask. The old-value pair has not been populated by
    // the shadow exchange yet, so it is the only safe address scratch here.
    // In particular, do not use saved_address_lo_vgpr: doing so corrupts the
    // per-lane loop invariant before the first exchange and makes the debug
    // path observe different behavior from the production path.
    const uint64_t report_base = *options.moi_report_buffer_address;
    if (!append_store_u32_vgpr(words,
                               report_base + offsetof(ConSanMoiReportHeader, atomic_record_count),
                               address_lo_vgpr, old_value_vgpr, arch) ||
        !append_store_u32_vgpr(words, report_base + offsetof(ConSanMoiReportHeader, flags),
                               static_cast<uint16_t>(address_lo_vgpr + 1u), old_value_vgpr, arch))
      return false;
  }
  words.push_back(*value_equal);
  words.push_back(*narrow_value);
  words.push_back(*low_mask_equal);
  const size_t low_value_fallback_branch = words.size();
  words.push_back(0);
  words.push_back(*high_mask_equal);
  const size_t high_value_fallback_branch = words.size();
  words.push_back(0);
  if (!append_exact_byte_cell_provenance_base(
          words, arch, lds_byte_offset_vgpr, static_byte_offset, relative_cell_index,
          relative_cell_index_vgpr, byte_count, current_low_vgpr, lane_rank_vgpr, current_high_vgpr,
          static_cast<uint16_t>(*options.scratch_vgpr + 4u))) {
    errors.emplace_back(
        "ConSan MOI inline-shadow probe could not encode exact byte-cell provenance");
    return false;
  }
  words.push_back(*read_mask);
  words.push_back(*mask_equal);
  words.push_back(*narrow_mask);
  words.push_back(*low_mask_equal);
  const size_t low_mask_fallback_branch = words.size();
  words.push_back(0);
  words.push_back(*high_mask_equal);
  const size_t high_mask_fallback_branch = words.size();
  words.push_back(0);
  const auto common_mask_nonzero =
      instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0u), lane_rank_vgpr, arch);
  const auto narrow_common_nonzero =
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  if (!common_mask_nonzero || !narrow_common_nonzero)
    return false;
  words.push_back(*common_mask_nonzero);
  words.push_back(*narrow_common_nonzero);

  const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
      lane_rank_vgpr, group_exec_sgpr, scalar_positive_inline_u32(0), arch);
  const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
      lane_rank_vgpr, static_cast<uint16_t>(group_exec_sgpr + 1u),
      vector_source_vgpr(lane_rank_vgpr), arch);
  const auto first_lane =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), lane_rank_vgpr, arch);
  const auto narrow_first =
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto restore_group_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, group_exec_sgpr, arch);
  const auto save_publishers =
      instrumentation::build_s_mov_b64(publisher_exec_sgpr, kRdna4ExecLo, arch);
  const auto use_group_diagnostic_mask =
      instrumentation::build_s_mov_b64(publisher_exec_sgpr, group_exec_sgpr, arch);
  const auto use_committed_diagnostic_mask =
      instrumentation::build_s_mov_b64(publisher_exec_sgpr, kRdna4ExecLo, arch);
  if (!mbcnt_lo || !mbcnt_hi || !first_lane || !narrow_first || !restore_group_exec ||
      !save_publishers || !use_group_diagnostic_mask || !use_committed_diagnostic_mask)
    return false;
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_lane);
  words.push_back(*narrow_first);
  words.push_back(*save_publishers);
  if (!append_exact_byte_representative_lane(words, arch, current_low_vgpr, current_high_vgpr)) {
    return false;
  }
  if (!append_versioned_exact_shadow_transaction(words, options, arch, address_lo_vgpr,
                                                 current_low_vgpr, old_value_vgpr,
                                                 publisher_exec_sgpr, committed_exec_sgpr,
                                                 /*retry_contention=*/true)) {
    return false;
  }
  // A uniform representative stands for the complete metadata-identical
  // address group. Retain that exact group as the diagnostic lane mask.
  words.push_back(*use_group_diagnostic_mask);
  const size_t common_branch = words.size();
  words.push_back(0);

  const size_t fallback = words.size();
  words.push_back(*restore_group_exec);
  if (!append_exact_byte_cell_provenance_base(
          words, arch, lds_byte_offset_vgpr, static_byte_offset, relative_cell_index,
          relative_cell_index_vgpr, byte_count, current_low_vgpr, lane_rank_vgpr, current_high_vgpr,
          static_cast<uint16_t>(*options.scratch_vgpr + 4u))) {
    errors.emplace_back(
        "ConSan MOI inline-shadow fallback could not encode exact byte-cell provenance");
    return false;
  }
  words.push_back(build_v_mov_b32_e32(current_high_vgpr, vector_source_vgpr(lane_rank_vgpr), arch));
  // Metadata-distinct lanes sharing one address cannot all retain the same
  // odd-version reservation concurrently. Publish one representative and
  // shrink group_exec to that lane; the outer pending-mask loop then revisits
  // the remaining lanes and serializes each complete metadata value.
  words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
  words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
  words.push_back(*first_lane);
  words.push_back(*narrow_first);
  words.push_back(*save_group);
  const auto fallback_mask_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
      scalar_positive_inline_u32(0u), current_high_vgpr, arch);
  const auto narrow_fallback_nonzero =
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  if (!fallback_mask_nonzero || !narrow_fallback_nonzero)
    return false;
  words.push_back(*fallback_mask_nonzero);
  words.push_back(*narrow_fallback_nonzero);
  words.push_back(*save_publishers);
  if (!append_exact_byte_representative_lane(words, arch, current_low_vgpr, current_high_vgpr)) {
    return false;
  }
  if (!append_versioned_exact_shadow_transaction(words, options, arch, address_lo_vgpr,
                                                 current_low_vgpr, old_value_vgpr,
                                                 publisher_exec_sgpr, committed_exec_sgpr,
                                                 /*retry_contention=*/true)) {
    return false;
  }
  // Nonuniform metadata cannot be attributed to the whole group. EXEC now
  // contains only committed, same-dispatch publishers admitted to diagnose.
  words.push_back(*use_committed_diagnostic_mask);
  const size_t diagnostic = words.size();

  // Counter updates and version-address CAS operations use the working address
  // pair. Restore it from the loop invariants before diagnostics and the next
  // partition iteration.
  words.push_back(restore_address_lo);
  words.push_back(restore_address_hi);
  // Narrow spill-backed paths deliberately pay one reload per partition only
  // when diagnostics can consume the recovered guest address.
  if (layout.diagnostic_capacity != 0 && spill_backed_lds_byte_offset_source) {
    if (spill == nullptr) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe has no spill plan for its diagnostic LDS address");
      return false;
    }
    const auto reload = consan_detail::append_reload_moi_spilled_vgpr(
        words, *spill, lds_byte_offset_vgpr, *spill_backed_lds_byte_offset_source, arch);
    if (reload != consan_detail::MoiSpilledVgprReloadResult::Appended) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not recover its diagnostic LDS address: " +
          std::string(consan_detail::moi_spilled_vgpr_reload_result_name(reload)));
      return false;
    }
  }

  // The partition operations changed VCC/SCC. Restore the incoming values so
  // the existing diagnostic helper can snapshot and restore the true program
  // state while evaluating every lane in this address group.
  if (!append_restore_moi_special_state(words, options, arch) ||
      !append_inline_shadow_diagnostic_words(
          words, candidate, options, arch, layout, old_value_vgpr,
          static_cast<uint16_t>(old_value_vgpr + 1u), current_low_vgpr, current_high_vgpr,
          lds_byte_offset_vgpr, static_byte_offset, byte_count,
          /*special_state_already_saved=*/true,
          /*diagnostic_lane_mask_sgpr=*/publisher_exec_sgpr,
          /*capture_first_diagnostic_only=*/false,
          /*compact_local_shadow=*/false,
          /*workgroup_local_shadow=*/false,
          /*generation_tagged_local_shadow=*/false,
          /*prior_byte_provenance_vgpr=*/
          static_cast<uint16_t>(old_value_vgpr + 6u),
          /*current_byte_provenance_vgpr=*/
          static_cast<uint16_t>(old_value_vgpr + 7u), relative_cell_index_vgpr))
    return false;

  if (options.moi_partition_mask_debug) {
    // Debug-only side channel for gfx1201 qualification. The report fields are
    // otherwise unused by InlineShadow. Capture before the subtraction while
    // group EXEC is still active; the matching hook log labels the run as
    // non-acceptance telemetry.
    const uint64_t report_base = *options.moi_report_buffer_address;
    if (!append_store_u32_sgpr(words, report_base + offsetof(ConSanMoiReportHeader, dispatch_id),
                               group_exec_sgpr, current_low_vgpr, address_lo_vgpr, arch) ||
        !append_store_u32_sgpr(
            words, report_base + offsetof(ConSanMoiReportHeader, dispatch_id) + sizeof(uint32_t),
            static_cast<uint16_t>(group_exec_sgpr + 1u), current_low_vgpr, address_lo_vgpr, arch))
      return false;
  }

  // The selected group is a subset of pending EXEC, so XOR removes exactly
  // that group. Keep this independent of the gfx12 AND-NOT operand convention;
  // live qualification is the authority for this generated control flow.
  const auto remove_group =
      instrumentation::build_s_xor_b64(pending_exec_sgpr, pending_exec_sgpr, group_exec_sgpr, arch);
  const auto select_remaining =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, pending_exec_sgpr, arch);
  const auto restore_incoming_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, incoming_exec_sgpr, arch);
  if (!remove_group || !select_remaining || !restore_incoming_exec)
    return false;
  words.push_back(*remove_group);
  // Install the remaining mask into EXEC and branch on it. Live gfx1201 debug
  // telemetry proved this traversal reaches the final divergent lane when
  // there is scalar distance after the pending-mask XOR. Keep that dependency
  // explicit: without it, both an immediate EXEC move and an immediate scalar
  // compare observed only the first address group in acceptance runs.
  words.push_back(build_s_nop(0, arch));
  words.push_back(build_s_nop(0, arch));
  words.push_back(*select_remaining);
  const size_t loop_branch = words.size();
  words.push_back(0);
  words.push_back(*restore_incoming_exec);
  // The diagnostic path uses current_high_vgpr as a field temporary. Restore
  // both packed words before returning so a following cell of the same wide
  // access starts from the original owner/epoch/workgroup metadata.
  words.push_back(restore_current_low);
  words.push_back(restore_current_high);
  if (!append_restore_moi_special_state(words, options, arch))
    return false;

  const auto branch_offset = [](size_t from, size_t to) -> std::optional<int16_t> {
    const int64_t delta = static_cast<int64_t>(to) - static_cast<int64_t>(from + 1u);
    if (delta < std::numeric_limits<int16_t>::min() || delta > std::numeric_limits<int16_t>::max())
      return std::nullopt;
    return static_cast<int16_t>(delta);
  };
  const auto low_value_delta = branch_offset(low_value_fallback_branch, fallback);
  const auto high_value_delta = branch_offset(high_value_fallback_branch, fallback);
  const auto low_mask_delta = branch_offset(low_mask_fallback_branch, fallback);
  const auto high_mask_delta = branch_offset(high_mask_fallback_branch, fallback);
  const auto diagnostic_delta = branch_offset(common_branch, diagnostic);
  const auto loop_delta = branch_offset(loop_branch, loop);
  if (!low_value_delta || !high_value_delta || !low_mask_delta || !high_mask_delta ||
      !diagnostic_delta || !loop_delta)
    return false;
  const auto low_value_branch = instrumentation::build_s_cbranch_scc0(*low_value_delta, arch);
  const auto high_value_branch = instrumentation::build_s_cbranch_scc0(*high_value_delta, arch);
  const auto low_mask_branch = instrumentation::build_s_cbranch_scc0(*low_mask_delta, arch);
  const auto high_mask_branch = instrumentation::build_s_cbranch_scc0(*high_mask_delta, arch);
  const auto continue_loop = instrumentation::build_s_cbranch_execnz(*loop_delta, arch);
  if (!low_value_branch || !high_value_branch || !low_mask_branch || !high_mask_branch ||
      !continue_loop)
    return false;
  words[low_value_fallback_branch] = *low_value_branch;
  words[high_value_fallback_branch] = *high_value_branch;
  words[low_mask_fallback_branch] = *low_mask_branch;
  words[high_mask_fallback_branch] = *high_mask_branch;
  words[common_branch] = build_s_branch(*diagnostic_delta, arch);
  words[loop_branch] = *continue_loop;
  return true;
}

// Make one packed-validity local slot ready without eagerly clearing the full
// exact mirror. Atomic OR claims the cell's initializing bit. The winner
// zeroes the exact slot before publishing its ready bit; every participating
// lane then polls that ready bit before the normal exact exchange. Two bits
// per slot allow neighboring cells to share one atomic state word safely.
[[nodiscard]] bool append_workgroup_local_lazy_shadow_ready(
    std::vector<uint32_t> &words, const ConSanOptions &options, rj_code_arch_t arch,
    const ConSanMoiWorkgroupShadowLayout &workgroup_shadow, uint16_t saved_cell_vgpr,
    std::vector<std::string> &errors) {
  if (!workgroup_shadow.lazy_initialization)
    return true;
  if (consan_moi_generation_tagged_workgroup_shadow(workgroup_shadow))
    return true;
  if ((arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_GFX1250) ||
      !options.scratch_vgpr || !options.moi_exec_save_sgpr ||
      workgroup_shadow.validity_size == 0u) {
    errors.emplace_back("ConSan MOI lazy local shadow requires gfx12 packed validity state");
    return false;
  }

  const uint16_t scratch = *options.scratch_vgpr;
  const uint16_t address_vgpr = scratch;
  const uint16_t temporary_vgpr = static_cast<uint16_t>(scratch + 4u);
  const uint16_t state_vgpr = static_cast<uint16_t>(scratch + 5u);
  const uint16_t zero_vgpr = static_cast<uint16_t>(scratch + 6u);
  const uint16_t initializing_mask_vgpr = static_cast<uint16_t>(scratch + 8u);
  const uint16_t ready_mask_vgpr = static_cast<uint16_t>(scratch + 11u);
  const uint16_t validity_address_vgpr = static_cast<uint16_t>(scratch + 12u);
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 14u);
  const uint16_t bounded_exec_sgpr = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 16u);

  const auto state_word = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(4u), saved_cell_vgpr, arch);
  const auto scale_state_word = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(2u), temporary_vgpr, arch);
  const auto validity_address = instrumentation::build_v_add_u32_literal(
      validity_address_vgpr, workgroup_shadow.validity_base, temporary_vgpr, arch);
  const auto state_index =
      instrumentation::build_v_and_b32_literal(initializing_mask_vgpr, 15u, saved_cell_vgpr, arch);
  const auto scale_state_index = instrumentation::build_v_lshlrev_b32(
      initializing_mask_vgpr, scalar_positive_inline_u32(1u), initializing_mask_vgpr, arch);
  const auto initializing_mask = instrumentation::build_v_lshlrev_b32(
      initializing_mask_vgpr, vector_source_vgpr(initializing_mask_vgpr), ready_mask_vgpr, arch);
  const auto ready_mask = instrumentation::build_v_lshlrev_b32(
      ready_mask_vgpr, scalar_positive_inline_u32(1u), initializing_mask_vgpr, arch);
  const auto claim = instrumentation::build_ds_or_rtn_b32(state_vgpr, validity_address_vgpr,
                                                          initializing_mask_vgpr, 0u, arch);
  const auto prior_claim = instrumentation::build_v_and_b32(
      temporary_vgpr, vector_source_vgpr(initializing_mask_vgpr), state_vgpr, arch);
  const auto won_claim =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0u), temporary_vgpr, arch);
  const auto narrow_winners =
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto shadow_byte_offset = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(workgroup_shadow.compact ? 2u : 3u),
      saved_cell_vgpr, arch);
  const auto shadow_address = instrumentation::build_v_add_u32_literal(
      address_vgpr, workgroup_shadow.base, temporary_vgpr, arch);
  const auto clear_slot =
      workgroup_shadow.compact
          ? instrumentation::build_ds_store_b32(address_vgpr, state_vgpr, 0u, arch)
          : instrumentation::build_ds_store_b64(address_vgpr, state_vgpr, 0u, arch);
  const auto publish_ready = instrumentation::build_ds_or_rtn_b32(state_vgpr, validity_address_vgpr,
                                                                  ready_mask_vgpr, 0u, arch);
  const auto restore_participants =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, temporary_exec_sgpr, arch);
  const auto observe_ready =
      instrumentation::build_ds_load_b32(state_vgpr, validity_address_vgpr, 0u, arch);
  const auto observed_ready = instrumentation::build_v_and_b32(
      temporary_vgpr, vector_source_vgpr(ready_mask_vgpr), state_vgpr, arch);
  const auto is_ready = instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(ready_mask_vgpr),
                                                                temporary_vgpr, arch);
  const auto remove_ready =
      instrumentation::build_s_andn2_b64(kRdna4ExecLo, kRdna4ExecLo, kRdna4VccLo, arch);
  const auto skip_initialization = instrumentation::build_s_cbranch_execz(0, arch);
  const auto continue_poll = instrumentation::build_s_cbranch_execnz(0, arch);
  const auto restore_bounded =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, bounded_exec_sgpr, arch);
  if (!state_word || !scale_state_word || !validity_address || !state_index || !scale_state_index ||
      !initializing_mask || !ready_mask || !claim || !prior_claim || !won_claim ||
      !narrow_winners || !shadow_byte_offset || !shadow_address || !clear_slot || !publish_ready ||
      !restore_participants || !observe_ready || !observed_ready || !is_ready || !remove_ready ||
      !skip_initialization || !continue_poll || !restore_bounded) {
    errors.emplace_back(
        "ConSan MOI lazy local shadow could not encode packed gfx12 state transitions");
    return false;
  }

  words.push_back(*state_word);
  words.push_back(*scale_state_word);
  words.insert(words.end(), validity_address->begin(), validity_address->end());
  words.push_back(build_v_mov_b32_e32(ready_mask_vgpr, scalar_positive_inline_u32(1u), arch));
  words.insert(words.end(), state_index->begin(), state_index->end());
  words.push_back(*scale_state_index);
  words.push_back(*initializing_mask);
  words.push_back(*ready_mask);
  // The steady-state path observes ready with an ordinary LDS load and
  // bypasses every atomic state transition. Only not-yet-ready lanes claim.
  words.insert(words.end(), observe_ready->begin(), observe_ready->end());
  if (!append_moi_lds_wait(words, arch))
    return false;
  words.push_back(*observed_ready);
  words.push_back(*is_ready);
  words.push_back(*remove_ready);
  const size_t skip_initialization_index = words.size();
  words.push_back(*skip_initialization);
  words.insert(words.end(), claim->begin(), claim->end());
  if (!append_moi_lds_wait(words, arch))
    return false;
  words.push_back(*prior_claim);
  words.push_back(*won_claim);
  words.push_back(*narrow_winners);
  words.push_back(build_v_mov_b32_e32(state_vgpr, scalar_positive_inline_u32(0u), arch));
  words.push_back(build_v_mov_b32_e32(zero_vgpr, scalar_positive_inline_u32(0u), arch));
  words.push_back(*shadow_byte_offset);
  words.insert(words.end(), shadow_address->begin(), shadow_address->end());
  words.insert(words.end(), clear_slot->begin(), clear_slot->end());
  if (!append_moi_lds_wait(words, arch))
    return false;
  words.insert(words.end(), publish_ready->begin(), publish_ready->end());
  if (!append_moi_lds_wait(words, arch))
    return false;
  words.push_back(*restore_participants);
  const size_t poll_begin = words.size();
  words.insert(words.end(), observe_ready->begin(), observe_ready->end());
  if (!append_moi_lds_wait(words, arch))
    return false;
  words.push_back(*observed_ready);
  words.push_back(*is_ready);
  words.push_back(*remove_ready);
  const size_t poll_branch = words.size();
  const int64_t poll_delta =
      static_cast<int64_t>(poll_begin) - static_cast<int64_t>(poll_branch + 1u);
  if (poll_delta < std::numeric_limits<int16_t>::min() ||
      poll_delta > std::numeric_limits<int16_t>::max()) {
    errors.emplace_back("ConSan MOI lazy local shadow polling branch exceeds SOPP range");
    return false;
  }
  const auto patched_poll =
      instrumentation::build_s_cbranch_execnz(static_cast<int16_t>(poll_delta), arch);
  if (!patched_poll)
    return false;
  words.push_back(*patched_poll);
  const size_t restore_bounded_index = words.size();
  words.push_back(*restore_bounded);
  const int64_t skip_delta = static_cast<int64_t>(restore_bounded_index) -
                             static_cast<int64_t>(skip_initialization_index + 1u);
  if (skip_delta < std::numeric_limits<int16_t>::min() ||
      skip_delta > std::numeric_limits<int16_t>::max()) {
    errors.emplace_back("ConSan MOI lazy local shadow fast-path branch exceeds SOPP range");
    return false;
  }
  const auto patched_skip =
      instrumentation::build_s_cbranch_execz(static_cast<int16_t>(skip_delta), arch);
  if (!patched_skip)
    return false;
  words[skip_initialization_index] = *patched_skip;
  return true;
}

// Publish one exact metadata entry into the bounded shadow interval reserved
// in the owning kernel's LDS allocation. Unlike the global versioned table,
// a DS exchange is naturally scoped to one workgroup and atomically returns
// the complete prior 64-bit entry, so it needs neither dispatch qualification
// nor a multi-instruction reservation protocol.
[[nodiscard]] bool append_workgroup_local_exact_shadow_swap(
    std::vector<uint32_t> &words, const ConSanMoiCandidate &candidate, const ConSanOptions &options,
    rj_code_arch_t arch, const ConSanMoiReportBufferLayout &report_layout,
    const ConSanMoiWorkgroupShadowLayout &workgroup_shadow, uint16_t address_vgpr,
    uint16_t current_low_vgpr, uint16_t current_high_vgpr, uint16_t old_value_vgpr,
    uint16_t lds_byte_offset_vgpr, uint32_t static_byte_offset, uint32_t cell_index,
    uint32_t byte_count, uint16_t diagnostic_lds_byte_offset_vgpr,
    uint32_t diagnostic_static_byte_offset, std::optional<uint16_t> relative_cell_index_vgpr,
    std::vector<std::string> &errors) {
  if (!options.scratch_vgpr || !options.moi_exec_save_sgpr)
    return false;
  if (workgroup_shadow.compact || consan_moi_generation_tagged_workgroup_shadow(workgroup_shadow)) {
    errors.emplace_back(
        "ConSan MOI exact-byte local publish requires one full-width initialized cell");
    return false;
  }

  const uint16_t temporary_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t saved_current_low_vgpr = static_cast<uint16_t>(old_value_vgpr + 4u);
  const uint16_t saved_current_high_vgpr = static_cast<uint16_t>(old_value_vgpr + 5u);
  const uint16_t saved_cell_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t exec_base = *options.moi_exec_save_sgpr;
  const uint16_t incoming_exec_sgpr = static_cast<uint16_t>(exec_base + 12u);
  const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(exec_base + 14u);
  const uint16_t bounded_exec_sgpr = static_cast<uint16_t>(exec_base + 16u);
  const uint16_t valid_workgroup_exec_sgpr = static_cast<uint16_t>(exec_base + 22u);
  const uint16_t provenance_temporary_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  const auto restore_valid_workgroup =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, valid_workgroup_exec_sgpr, arch);
  if (!restore_valid_workgroup) {
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not restore its workgroup lanes");
    return false;
  }

  words.push_back(
      build_v_mov_b32_e32(saved_current_low_vgpr, vector_source_vgpr(current_low_vgpr), arch));
  words.push_back(
      build_v_mov_b32_e32(saved_current_high_vgpr, vector_source_vgpr(current_high_vgpr), arch));
  if (!append_exact_byte_cell_provenance_base(
          words, arch, diagnostic_lds_byte_offset_vgpr, diagnostic_static_byte_offset, cell_index,
          relative_cell_index_vgpr, byte_count, current_low_vgpr, current_high_vgpr,
          provenance_temporary_vgpr, temporary_vgpr)) {
    errors.emplace_back("ConSan MOI local publish could not encode exact byte-cell provenance");
    return false;
  }
  // Worst-case cell loops deliberately include iterations that are empty for
  // some runtime alignments. Remove those lanes before bounds accounting or
  // LDS traffic so only bytes actually touched can consume shadow coverage.
  const auto mask_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0u),
                                                                    current_high_vgpr, arch);
  const auto narrow_nonzero =
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  if (!mask_nonzero || !narrow_nonzero)
    return false;
  words.push_back(*mask_nonzero);
  words.push_back(*narrow_nonzero);

  uint16_t effective_offset_vgpr = lds_byte_offset_vgpr;
  if (static_byte_offset != 0u) {
    if (!append_compute_effective_lds_byte_offset(words, temporary_vgpr, lds_byte_offset_vgpr,
                                                  static_byte_offset, arch))
      return false;
    effective_offset_vgpr = temporary_vgpr;
  }

  const auto cell = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
      effective_offset_vgpr, arch);
  const uint32_t shadow_cell_capacity = workgroup_shadow.size / sizeof(uint64_t);
  const uint32_t final_cell_index = cell_index;
  const uint32_t cell_capacity =
      final_cell_index < shadow_cell_capacity ? shadow_cell_capacity - final_cell_index : 0u;
  const auto capacity = instrumentation::build_v_mov_b32_literal(address_vgpr, cell_capacity, arch);
  const auto in_bounds = instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(address_vgpr),
                                                                 temporary_vgpr, arch);
  const auto save_incoming_exec =
      instrumentation::build_s_mov_b64(incoming_exec_sgpr, kRdna4ExecLo, arch);
  const auto narrow_bounded =
      instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
  const auto save_bounded_exec =
      instrumentation::build_s_mov_b64(bounded_exec_sgpr, kRdna4ExecLo, arch);
  const auto select_out_of_bounds =
      instrumentation::build_s_andn2_b64(kRdna4ExecLo, incoming_exec_sgpr, bounded_exec_sgpr, arch);
  const auto restore_bounded_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, bounded_exec_sgpr, arch);
  const auto shadow_byte_offset = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(3u), temporary_vgpr, arch);
  const uint32_t local_base = workgroup_shadow.base + cell_index * sizeof(uint64_t);
  const auto local_address =
      instrumentation::build_v_add_u32_literal(address_vgpr, local_base, temporary_vgpr, arch);
  const auto exchange = instrumentation::build_ds_storexchg_rtn_b64(old_value_vgpr, address_vgpr,
                                                                    current_low_vgpr, 0u, arch);
  if (!cell || !capacity || !in_bounds || !shadow_byte_offset || !local_address || !exchange) {
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not encode CDNA vector/DS operations");
    return false;
  }
  if (!save_incoming_exec || !narrow_bounded || !save_bounded_exec || !select_out_of_bounds ||
      !restore_bounded_exec || !restore_valid_workgroup) {
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not encode scalar lane masks");
    return false;
  }

  words.push_back(*save_incoming_exec);
  words.push_back(*cell);
  words.push_back(build_v_mov_b32_e32(saved_cell_vgpr, vector_source_vgpr(temporary_vgpr), arch));
  words.insert(words.end(), capacity->begin(), capacity->end());
  words.push_back(*in_bounds);
  words.push_back(*narrow_bounded);
  words.push_back(*save_bounded_exec);
  words.push_back(*select_out_of_bounds);
  const size_t skip_empty_undercoverage_index = words.size();
  words.push_back(0);
  if (!append_atomic_fetch_add_one_u32(
          words,
          *options.moi_report_buffer_address +
              offsetof(ConSanMoiReportHeader, inline_undercoverage_count),
          temporary_vgpr, address_vgpr, arch)) {
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not encode undercoverage accounting");
    return false;
  }
  const size_t restore_bounded_exec_index = words.size();
  words.push_back(*restore_bounded_exec);
  const int64_t skip_empty_undercoverage_delta =
      static_cast<int64_t>(restore_bounded_exec_index) -
      static_cast<int64_t>(skip_empty_undercoverage_index + 1u);
  if (skip_empty_undercoverage_delta < std::numeric_limits<int16_t>::min() ||
      skip_empty_undercoverage_delta > std::numeric_limits<int16_t>::max()) {
    errors.emplace_back("ConSan MOI inline-shadow local undercoverage branch exceeds SOPP range");
    return false;
  }
  const auto skip_empty_undercoverage = instrumentation::build_s_cbranch_execz(
      static_cast<int16_t>(skip_empty_undercoverage_delta), arch);
  if (!skip_empty_undercoverage) {
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not skip empty undercoverage accounting");
    return false;
  }
  words[skip_empty_undercoverage_index] = *skip_empty_undercoverage;
  if (!append_workgroup_local_lazy_shadow_ready(words, options, arch, workgroup_shadow,
                                                saved_cell_vgpr, errors)) {
    return false;
  }
  words.push_back(
      build_v_mov_b32_e32(provenance_temporary_vgpr, vector_source_vgpr(current_low_vgpr), arch));
  if (!append_exact_byte_representative_lane(words, arch, provenance_temporary_vgpr,
                                             current_high_vgpr)) {
    errors.emplace_back("ConSan MOI local publish could not encode its representative lane");
    return false;
  }
  words.push_back(
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch));
  words.push_back(
      build_v_mov_b32_e32(current_high_vgpr, vector_source_vgpr(saved_current_high_vgpr), arch));
  if (!append_add_exact_shadow_generation(words, current_low_vgpr, current_high_vgpr,
                                          provenance_temporary_vgpr, temporary_vgpr, arch)) {
    errors.emplace_back("ConSan MOI local publish could not embed exact byte-cell provenance");
    return false;
  }
  words.push_back(build_v_mov_b32_e32(temporary_vgpr, vector_source_vgpr(saved_cell_vgpr), arch));
  words.push_back(*shadow_byte_offset);
  words.insert(words.end(), local_address->begin(), local_address->end());
  words.insert(words.end(), exchange->begin(), exchange->end());
  if (!append_moi_lds_wait(words, arch)) {
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not encode its LDS result wait");
    return false;
  }

  // Address formation and conflict filtering use condition state. Reinstall
  // the displaced program's state before the diagnostic helper, whose final
  // restoration makes the local path observationally transparent.
  if (!append_restore_moi_special_state(words, options, arch)) {
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not restore special scalar state");
    return false;
  }
  const uint16_t prior_byte_provenance_vgpr = static_cast<uint16_t>(old_value_vgpr + 2u);
  const uint16_t current_byte_provenance_vgpr = static_cast<uint16_t>(old_value_vgpr + 3u);
  if (!append_extract_exact_shadow_generation(words, prior_byte_provenance_vgpr, old_value_vgpr,
                                              static_cast<uint16_t>(old_value_vgpr + 1u),
                                              temporary_vgpr, arch) ||
      !append_extract_exact_shadow_generation(words, current_byte_provenance_vgpr, current_low_vgpr,
                                              current_high_vgpr, temporary_vgpr, arch)) {
    errors.emplace_back("ConSan MOI local publish could not recover exact byte-cell provenance");
    return false;
  }
  if (!append_inline_shadow_diagnostic_words(
          words, candidate, options, arch, report_layout, old_value_vgpr,
          static_cast<uint16_t>(old_value_vgpr + 1u), current_low_vgpr, current_high_vgpr,
          diagnostic_lds_byte_offset_vgpr, diagnostic_static_byte_offset, byte_count,
          /*special_state_already_saved=*/true,
          /*diagnostic_lane_mask_sgpr=*/std::nullopt,
          /*capture_first_diagnostic_only=*/true,
          /*compact_local_shadow=*/false,
          /*workgroup_local_shadow=*/true,
          /*generation_tagged_local_shadow=*/false, prior_byte_provenance_vgpr,
          current_byte_provenance_vgpr, relative_cell_index_vgpr)) {
    errors.emplace_back(
        "ConSan MOI inline-shadow local publish could not encode conflict diagnostics");
    return false;
  }

  words.push_back(
      build_v_mov_b32_e32(current_low_vgpr, vector_source_vgpr(saved_current_low_vgpr), arch));
  words.push_back(
      build_v_mov_b32_e32(current_high_vgpr, vector_source_vgpr(saved_current_high_vgpr), arch));
  words.push_back(*restore_valid_workgroup);
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_shadow_words(
    std::span<const uint8_t> bytes, const ConSanMoiCandidate &candidate,
    const ConSanOptions &options, rj_code_arch_t arch, const ConSanMoiReportBufferLayout &layout,
    const ConSanMoiWorkgroupSources &workgroup_sources,
    std::optional<ConSanMoiWorkgroupShadowLayout> workgroup_shadow,
    std::optional<uint32_t> private_epoch_offset, std::optional<uint32_t> private_owner_offset,
    std::optional<uint16_t> private_owner_shift,
    std::optional<uint32_t> private_workgroup_key_offset, uint16_t workgroup_shadow_compact_token,
    uint16_t scratch_count, const VgprSpillSequence *spill, std::vector<std::string> &errors,
    uint32_t *guest_instruction_word_count = nullptr) {
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  if (!options.moi_exec_save_sgpr ||
      (!options.moi_dispatch_id_sgpr && !instrumentation::is_consan_rdna_arch(arch))) {
    errors.emplace_back(
        "ConSan MOI versioned exact-shadow probe requires EXEC-save and dispatch-ID state");
    return std::nullopt;
  }
  if (options.moi_partition_mask_debug &&
      (!options.moi_exec_save_sgpr || layout.diagnostic_capacity == 0)) {
    errors.emplace_back("ConSan MOI partition-mask debug requires wave-coalesced InlineShadow");
    return std::nullopt;
  }
  const uint16_t normal_scratch_count = inline_shadow_scratch_count(options, candidate, arch);
  const uint16_t spill_scratch_count =
      inline_shadow_spill_backed_scratch_count(options, candidate, arch);
  if (scratch_count != normal_scratch_count &&
      (spill == nullptr || scratch_count != spill_scratch_count)) {
    errors.emplace_back(
        "ConSan MOI inline-shadow probe scratch VGPR count disagrees with its resource plan");
    return std::nullopt;
  }
  if (spill != nullptr &&
      (spill->vgpr_base != *options.scratch_vgpr || spill->vgpr_count != scratch_count)) {
    errors.emplace_back(
        "ConSan MOI inline-shadow probe spill window disagrees with its resource plan");
    return std::nullopt;
  }
  if (static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back("ConSan MOI inline-shadow probe scratch VGPR window exceeds the limit");
    return std::nullopt;
  }
  if (!validate_inline_shadow_exec_save_sgpr(options, arch, errors))
    return std::nullopt;
  auto lds_byte_offset_vgpr = candidate_lds_byte_offset_vgpr(candidate, errors);
  if (!lds_byte_offset_vgpr)
    return std::nullopt;
  if (!options.moi_spill_overlaps_guest_operands &&
      reject_candidate_scratch_range_overlap(candidate, *options.scratch_vgpr, scratch_count,
                                             errors))
    return std::nullopt;
  if (!options.moi_owner_vgpr && !options.moi_persistent_owner_sgpr &&
      !options.automatic_moi_private_epoch) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires persistent owner state");
    return std::nullopt;
  }
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI epoch", errors) ||
      reject_optional_scratch_range_overlap(options.moi_workgroup_key_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI workgroup key", errors))
    return std::nullopt;

  const std::optional<std::vector<ConSanMoiAccessRange>> access_ranges =
      candidate_access_ranges(bytes, candidate, arch);
  if (!access_ranges || access_ranges->empty()) {
    errors.emplace_back("ConSan MOI inline-shadow probe requires a supported LDS access range");
    return std::nullopt;
  }

  const uint16_t address_lo_vgpr = *options.scratch_vgpr;
  const uint16_t address_hi_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 1u);
  const uint16_t low_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t high_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
  const uint16_t tmp_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t old_value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const uint16_t cas_expected_vgpr = consan_detail::inline_shadow_cas_expected_vgpr(old_value_vgpr);
  if (cas_expected_vgpr >= static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count) {
    errors.emplace_back(
        "ConSan MOI inline-shadow probe has no scratch VGPR for the expected CAS value");
    return std::nullopt;
  }
  const bool materialize_flat_offset =
      candidate_requires_gfx1250_flat_offset_materialization(candidate);
  const auto source_is_spilled = [&](uint16_t source) {
    return spill != nullptr && source >= spill->vgpr_base &&
           source < static_cast<uint32_t>(spill->vgpr_base) + spill->vgpr_count;
  };
  const std::optional<uint16_t> spill_backed_lds_byte_offset_source =
      spill != nullptr && !materialize_flat_offset &&
              (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
              source_is_spilled(*lds_byte_offset_vgpr)
          ? lds_byte_offset_vgpr
          : std::nullopt;
  std::optional<uint16_t> saved_lds_byte_offset_vgpr;
  if (materialize_flat_offset) {
    saved_lds_byte_offset_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + scratch_count - 2u);
  } else if (spill_backed_lds_byte_offset_source) {
    // Address construction and local exchange consume the guest offset before
    // the transaction initializes its expected CAS value. The authoritative
    // spill slot reloads the offset again before diagnostics. This phase
    // sharing keeps the minimum native CDNA access window at 16 VGPRs.
    saved_lds_byte_offset_vgpr = cas_expected_vgpr;
  } else if (moi_load_clobbers_address(candidate)) {
    saved_lds_byte_offset_vgpr =
        static_cast<uint16_t>(consan_detail::inline_shadow_loop_counter_vgpr(
                                  *options.scratch_vgpr, options.moi_exec_save_sgpr.has_value(),
                                  options.moi_track_atomics) +
                              inline_shadow_loop_scratch_count(options, candidate));
  }

  const auto kind = consan_moi_shadow_kind_from_access_kind(candidate.kind);
  if (workgroup_shadow && workgroup_shadow->compact &&
      (workgroup_shadow_compact_token == 0u ||
       workgroup_shadow_compact_token > consan_moi_exact_shadow::max_compact_token)) {
    errors.emplace_back("ConSan MOI compact local shadow requires a nonzero bounded site token");
    return std::nullopt;
  }
  const uint32_t low_literal =
      static_cast<uint32_t>(kind) | (workgroup_shadow && workgroup_shadow->compact
                                         ? static_cast<uint32_t>(workgroup_shadow_compact_token)
                                               << consan_moi_exact_shadow::compact_token_shift
                                         : 0u);
  const uint32_t high_literal =
      workgroup_shadow && workgroup_shadow->compact
          ? 0u
          : static_cast<uint32_t>(
                (candidate.text_offset & consan_moi_exact_shadow::max_instruction_offset)
                << (consan_moi_exact_shadow::instruction_offset_shift - 32u));
  const uint64_t exact_shadow_base =
      *options.moi_report_buffer_address + layout.exact_shadow_entries_offset;

  std::vector<uint32_t> words;
  words.reserve(candidate.size / sizeof(uint32_t) + 64u + (options.moi_exec_save_sgpr ? 120u : 0u) +
                (options.moi_epoch_vgpr ? 2u : 0u));
  const auto reload_spill_backed_lds_byte_offset = [&]() {
    if (!spill_backed_lds_byte_offset_source)
      return true;
    if (spill == nullptr) {
      errors.emplace_back("ConSan MOI inline-shadow probe has no spill plan for its LDS address");
      return false;
    }
    const auto reload = consan_detail::append_reload_moi_spilled_vgpr(
        words, *spill, *saved_lds_byte_offset_vgpr, *spill_backed_lds_byte_offset_source, arch);
    if (reload == consan_detail::MoiSpilledVgprReloadResult::Appended)
      return true;
    errors.emplace_back("ConSan MOI inline-shadow probe could not recover its spilled LDS "
                        "address: " +
                        std::string(consan_detail::moi_spilled_vgpr_reload_result_name(reload)));
    return false;
  };
  // A DS load may use the same VGPR for its address and destination. Preserve
  // the effective address before executing the displaced guest instruction;
  // otherwise the loaded payload becomes the exact-shadow cell index.
  if (saved_lds_byte_offset_vgpr) {
    if (!spill_backed_lds_byte_offset_source) {
      words.push_back(build_v_mov_b32_e32(*saved_lds_byte_offset_vgpr,
                                          vector_source_vgpr(*lds_byte_offset_vgpr), arch));
      if (materialize_flat_offset) {
        words.push_back(build_v_mov_b32_e32(
            static_cast<uint16_t>(*saved_lds_byte_offset_vgpr + 1u),
            vector_source_vgpr(static_cast<uint16_t>(*lds_byte_offset_vgpr + 1u)), arch));
      }
    }
    lds_byte_offset_vgpr = saved_lds_byte_offset_vgpr;
  }
  // Publish the shadow before the application access, then execute that access
  // immediately before its continuation. This prevents another wave from
  // observing application data while either side is still in trampoline
  // bookkeeping. Scratch allocation excludes every guest operand, so a
  // deferred store retains its address and payload; an address-clobbering load
  // uses the snapshot above for metadata and produces its application result
  // only after instrumentation is finished.
  // Preserve application condition state before metadata construction. Owner,
  // epoch, address, and partition setup all use VCC/SCC-producing
  // instructions; snapshotting inside the exchange loop would retain those
  // generated values rather than the state at the displaced LDS access.
  const bool preserve_special_state = options.moi_exec_save_sgpr.has_value();
  if (preserve_special_state && !append_save_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not preserve application VCC/SCC");
    return std::nullopt;
  }
  if (materialize_flat_offset) {
    const auto add = instrumentation::build_v_add_u64_signed_i24(*lds_byte_offset_vgpr,
                                                                 *candidate.raw_ioffset, arch);
    if (!add) {
      errors.emplace_back("ConSan MOI inline-shadow probe could not materialize FLAT offset");
      return std::nullopt;
    }
    words.insert(words.end(), add->begin(), add->end());
  }
  if (!append_moi_lds_wait(words, arch))
    return std::nullopt;

  const uint16_t workgroup_key_vgpr = old_value_vgpr;
  const bool generation_tagged_local_shadow =
      workgroup_shadow && consan_moi_generation_tagged_workgroup_shadow(*workgroup_shadow);
  const uint16_t original_exec_save_offset = kConSanMoiInlineOriginalExecSaveOffset;
  ConSanOptions key_options = options;
  // Eagerly initialized and bitmap-backed workgroup-local mirrors are
  // physically unreachable from every other live workgroup and omit the
  // generation field. A generation-tagged local mirror deliberately retains
  // that field so reused, uncleared LDS is distinguishable from current state.
  if (workgroup_shadow) {
    const auto save_original_exec = instrumentation::build_s_mov_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + original_exec_save_offset),
        kRdna4ExecLo, arch);
    if (!save_original_exec) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not retain its original workgroup lanes");
      return std::nullopt;
    }
    words.push_back(*save_original_exec);
  }
  if (private_workgroup_key_offset && (!workgroup_shadow || generation_tagged_local_shadow)) {
    const auto load = instrumentation::build_private_load_b32(workgroup_key_vgpr,
                                                              *private_workgroup_key_offset, arch);
    const auto wait = instrumentation::build_s_wait_private_load0(arch);
    if (!load || !wait) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not load its private workgroup key");
      return std::nullopt;
    }
    words.insert(words.end(), load->begin(), load->end());
    words.push_back(*wait);
    key_options.moi_workgroup_key_vgpr = workgroup_key_vgpr;
  }
  if ((!workgroup_shadow || generation_tagged_local_shadow) && options.moi_exec_save_sgpr &&
      !append_inline_workgroup_key(words, workgroup_sources, key_options, workgroup_key_vgpr,
                                   tmp_vgpr, address_lo_vgpr, original_exec_save_offset, arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not derive workgroup identity");
    return std::nullopt;
  }
  if (generation_tagged_local_shadow) {
    if (!options.moi_dispatch_id_sgpr) {
      errors.emplace_back("ConSan MOI generation-tagged local shadow requires dispatch identity");
      return std::nullopt;
    }
    // Workgroup coordinates alone repeat across dispatches. Fold the native
    // dispatch identity into the bounded generation and reserve zero for an
    // empty/unqualified cell, so LDS reuse cannot manufacture a prior owner.
    const auto mix_dispatch_low = instrumentation::build_v_xor_b32(
        workgroup_key_vgpr, *options.moi_dispatch_id_sgpr, workgroup_key_vgpr, arch);
    const auto mix_dispatch_high = instrumentation::build_v_xor_b32(
        workgroup_key_vgpr, static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u),
        workgroup_key_vgpr, arch);
    const auto bound_generation = instrumentation::build_v_and_b32_literal(
        workgroup_key_vgpr, consan_moi_exact_shadow::max_generation - 1u, workgroup_key_vgpr, arch);
    const auto make_nonzero_generation = instrumentation::build_v_add_u32(
        workgroup_key_vgpr, scalar_positive_inline_u32(1u), workgroup_key_vgpr, arch);
    if (!mix_dispatch_low || !mix_dispatch_high || !bound_generation || !make_nonzero_generation) {
      errors.emplace_back("ConSan MOI local shadow could not derive dispatch-qualified generation");
      return std::nullopt;
    }
    words.push_back(*mix_dispatch_low);
    words.push_back(*mix_dispatch_high);
    words.insert(words.end(), bound_generation->begin(), bound_generation->end());
    words.insert(words.end(), make_nonzero_generation->begin(), make_nonzero_generation->end());
  }
  if ((!workgroup_shadow || generation_tagged_local_shadow) && options.moi_exec_save_sgpr) {
    // The packed key is injective only inside its documented dimensional
    // bounds. Keep the guest running, but make every excluded lane-site
    // encounter observable in the Inline-only header flags/count field so a
    // clean acceptance run cannot silently pass with detector undercoverage.
    const uint16_t exec_base = *options.moi_exec_save_sgpr;
    const auto save_valid_exec = instrumentation::build_s_mov_b64(exec_base, kRdna4ExecLo, arch);
    const auto select_invalid_exec = instrumentation::build_s_andn2_b64(
        kRdna4ExecLo, static_cast<uint16_t>(exec_base + original_exec_save_offset), exec_base,
        arch);
    const auto restore_valid_exec = instrumentation::build_s_mov_b64(kRdna4ExecLo, exec_base, arch);
    std::vector<uint32_t> accounting_words;
    if (!save_valid_exec || !select_invalid_exec || !restore_valid_exec ||
        !append_atomic_fetch_add_one_u32(
            accounting_words,
            *options.moi_report_buffer_address +
                offsetof(ConSanMoiReportHeader, inline_unsupported_count),
            tmp_vgpr, address_lo_vgpr, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not account unsupported workgroup identity");
      return std::nullopt;
    }
    words.push_back(*save_valid_exec);
    words.push_back(*select_invalid_exec);
    words.insert(words.end(), accounting_words.begin(), accounting_words.end());
    words.push_back(*restore_valid_exec);
  }
  if (candidate.in_kernel && candidate.source == ConSanMoiCandidateSource::FlatMaybeGroup) {
    // Likely provenance deliberately admits flat sites whose static dataflow
    // cannot prove one aperture. Resolve that uncertainty with the same high
    // address comparison used by hardware: a 32-bit SHARED_BASE source is the
    // shared aperture's high half. Global lanes bypass shadow bookkeeping and
    // later rejoin for the displaced guest access.
    uint16_t address_hi_source = materialize_flat_offset
                                     ? static_cast<uint16_t>(*lds_byte_offset_vgpr + 1u)
                                     : static_cast<uint16_t>(*candidate.addr_vgpr + 1u);
    if (!materialize_flat_offset && source_is_spilled(address_hi_source)) {
      const uint16_t recovered_address_hi = static_cast<uint16_t>(*options.scratch_vgpr + 1u);
      const auto reload = consan_detail::append_reload_moi_spilled_vgpr(
          words, *spill, recovered_address_hi, address_hi_source, arch);
      if (reload != consan_detail::MoiSpilledVgprReloadResult::Appended) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not recover a spilled flat "
            "address: " +
            std::string(consan_detail::moi_spilled_vgpr_reload_result_name(reload)));
        return std::nullopt;
      }
      address_hi_source = recovered_address_hi;
    }
    const auto in_shared_aperture =
        instrumentation::build_v_cmp_eq_u32_vcc(kScalarOperandSharedBase, address_hi_source, arch);
    const auto retain_shared_lanes = instrumentation::build_s_and_saveexec_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 2u), kRdna4VccLo, arch);
    if (!in_shared_aperture || !retain_shared_lanes) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not resolve a maybe-group flat aperture");
      return std::nullopt;
    }
    words.push_back(*in_shared_aperture);
    words.push_back(*retain_shared_lanes);
  }
  if (workgroup_shadow) {
    const auto retain_valid_workgroup = instrumentation::build_s_mov_b64(
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + 22u), kRdna4ExecLo, arch);
    if (!retain_valid_workgroup) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not retain validated workgroup lanes");
      return std::nullopt;
    }
    words.push_back(*retain_valid_workgroup);
  }

  const auto mov_low = instrumentation::build_v_mov_b32_literal(low_vgpr, low_literal, arch);
  const auto mov_high = instrumentation::build_v_mov_b32_literal(high_vgpr, high_literal, arch);
  if (!mov_low || !mov_high) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode exact-shadow literals");
    return std::nullopt;
  }
  words.insert(words.end(), mov_low->begin(), mov_low->end());
  if (!append_inline_shadow_owner_field(words, options, low_vgpr, tmp_vgpr, high_vgpr, arch,
                                        private_owner_offset, private_owner_shift, errors)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode owner field");
    return std::nullopt;
  }
  if (!append_inline_shadow_epoch_field(words, options, private_epoch_offset, low_vgpr, tmp_vgpr,
                                        arch, errors)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode epoch field");
    return std::nullopt;
  }
  words.insert(words.end(), mov_high->begin(), mov_high->end());
  if ((!workgroup_shadow || generation_tagged_local_shadow) && options.moi_exec_save_sgpr &&
      !append_add_exact_shadow_generation(words, low_vgpr, high_vgpr, workgroup_key_vgpr, tmp_vgpr,
                                          arch)) {
    errors.emplace_back("ConSan MOI inline-shadow probe could not encode workgroup identity");
    return std::nullopt;
  }

  // The global versioned transaction keeps old_value at +5 so its CDNA CAS
  // tuples remain even. The workgroup-local B64 DS exchange instead requires
  // an even return tuple, so rotate only that local result to the unused +6
  // pair on CDNA3/4.
  const uint16_t local_old_value_vgpr = static_cast<uint16_t>(
      old_value_vgpr +
      (arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 1u : 0u));

  for (const ConSanMoiAccessRange &range : *access_ranges) {
    ConSanMoiLdsCellRange cell_range =
        consan_moi_lds_cell_range_for_bytes(range.static_byte_offset, range.byte_count);
    // The register-sourced LDS base can have any byte alignment even when the
    // instruction's static offset is aligned. Emit the target-neutral maximum
    // and let the exact mask remove empty per-lane iterations before coverage
    // accounting and publication.
    cell_range.cell_count = consan_moi_maximum_cell_count_for_unaligned_bytes(range.byte_count);
    const bool use_cell_loop =
        cell_range.cell_count > 1u && inline_shadow_loop_scratch_count(options, candidate) != 0u;
    for (uint32_t cell_index = 0; cell_index < cell_range.cell_count; ++cell_index) {
      if (workgroup_shadow) {
        if (use_cell_loop) {
          if (cell_index != 0u)
            continue;
          const uint16_t loop_counter_vgpr = consan_detail::inline_shadow_loop_counter_vgpr(
              *options.scratch_vgpr, options.moi_exec_save_sgpr.has_value(),
              options.moi_track_atomics);
          const uint16_t loop_offset_vgpr = static_cast<uint16_t>(loop_counter_vgpr + 1u);
          if (static_cast<uint32_t>(loop_offset_vgpr) >=
              static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count) {
            errors.emplace_back(
                "ConSan MOI inline-shadow local cell loop exceeds its scratch plan");
            return std::nullopt;
          }
          if (!reload_spill_backed_lds_byte_offset()) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not recover its spilled LDS address");
            return std::nullopt;
          }
          if (range.static_byte_offset == 0u) {
            words.push_back(build_v_mov_b32_e32(loop_offset_vgpr,
                                                vector_source_vgpr(*lds_byte_offset_vgpr), arch));
          } else if (!append_compute_effective_lds_byte_offset(words, loop_offset_vgpr,
                                                               *lds_byte_offset_vgpr,
                                                               range.static_byte_offset, arch)) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not initialize its local cell loop");
            return std::nullopt;
          }
          words.push_back(
              build_v_mov_b32_e32(loop_counter_vgpr, scalar_positive_inline_u32(0), arch));
          const size_t loop_begin = words.size();
          if (!append_workgroup_local_exact_shadow_swap(
                  words, candidate, options, arch, layout, *workgroup_shadow, address_lo_vgpr,
                  low_vgpr, high_vgpr, local_old_value_vgpr, loop_offset_vgpr,
                  /*static_byte_offset=*/0u, /*cell_index=*/0u, range.byte_count,
                  *lds_byte_offset_vgpr, range.static_byte_offset, loop_counter_vgpr, errors)) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode looped local LDS shadow publish");
            return std::nullopt;
          }
          const auto advance_offset = instrumentation::build_v_add_u32(
              loop_offset_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_bytes),
              loop_offset_vgpr, arch);
          const auto advance_counter = instrumentation::build_v_add_u32(
              loop_counter_vgpr, scalar_positive_inline_u32(1), loop_counter_vgpr, arch);
          const auto more_cells = instrumentation::build_v_cmp_gt_u32_vcc(
              scalar_positive_inline_u32(cell_range.cell_count), loop_counter_vgpr, arch);
          const uint16_t loop_exec_save_sgpr =
              static_cast<uint16_t>(*options.moi_exec_save_sgpr + 14u);
          const uint16_t valid_workgroup_exec_sgpr =
              static_cast<uint16_t>(*options.moi_exec_save_sgpr + 22u);
          const auto narrow_more_cells =
              instrumentation::build_s_and_saveexec_b64(loop_exec_save_sgpr, kRdna4VccLo, arch);
          const auto restore_valid_workgroup =
              instrumentation::build_s_mov_b64(kRdna4ExecLo, valid_workgroup_exec_sgpr, arch);
          if (!advance_offset || !advance_counter || !more_cells || !narrow_more_cells ||
              !restore_valid_workgroup) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its local cell loop control");
            return std::nullopt;
          }
          words.insert(words.end(), advance_offset->begin(), advance_offset->end());
          words.insert(words.end(), advance_counter->begin(), advance_counter->end());
          words.push_back(*more_cells);
          // Vector compares only define VCC for active lanes. Branching on raw
          // VCC can therefore observe stale guest bits from inactive lanes and
          // loop forever. Narrow EXEC to the active lanes that still need a
          // cell, branch on that mask, then restore the validated workgroup
          // mask on loop exit.
          words.push_back(*narrow_more_cells);
          const size_t loop_branch = words.size();
          const int64_t loop_delta =
              static_cast<int64_t>(loop_begin) - static_cast<int64_t>(loop_branch + 1u);
          if (loop_delta < std::numeric_limits<int16_t>::min() ||
              loop_delta > std::numeric_limits<int16_t>::max()) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its local cell loop branch");
            return std::nullopt;
          }
          const auto loop =
              instrumentation::build_s_cbranch_execnz(static_cast<int16_t>(loop_delta), arch);
          if (!loop) {
            errors.emplace_back(
                "ConSan MOI inline-shadow probe could not encode its local cell loop branch");
            return std::nullopt;
          }
          words.push_back(*loop);
          words.push_back(*restore_valid_workgroup);
        } else if (!reload_spill_backed_lds_byte_offset() ||
                   !append_workgroup_local_exact_shadow_swap(
                       words, candidate, options, arch, layout, *workgroup_shadow, address_lo_vgpr,
                       low_vgpr, high_vgpr, local_old_value_vgpr, *lds_byte_offset_vgpr,
                       range.static_byte_offset, cell_index, range.byte_count,
                       *lds_byte_offset_vgpr, range.static_byte_offset,
                       /*relative_cell_index_vgpr=*/std::nullopt, errors)) {
          errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode local LDS shadow publish");
          return std::nullopt;
        }
        continue;
      }
      const bool loop_external_cells = use_cell_loop;
      const uint16_t loop_counter_vgpr = consan_detail::inline_shadow_loop_counter_vgpr(
          *options.scratch_vgpr, options.moi_exec_save_sgpr.has_value(), options.moi_track_atomics);
      const uint16_t saved_workgroup_key_vgpr = static_cast<uint16_t>(loop_counter_vgpr + 1u);
      if (loop_external_cells) {
        if (static_cast<uint32_t>(saved_workgroup_key_vgpr) >=
            static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count) {
          errors.emplace_back(
              "ConSan MOI inline-shadow external cell loop exceeds its scratch plan");
          return std::nullopt;
        }
        if (cell_index != 0u)
          continue;
        words.push_back(
            build_v_mov_b32_e32(loop_counter_vgpr, scalar_positive_inline_u32(0), arch));
        words.push_back(build_v_mov_b32_e32(saved_workgroup_key_vgpr,
                                            vector_source_vgpr(workgroup_key_vgpr), arch));
      }
      // Recompute the complete external address at the top of every wide-cell
      // iteration. The versioned transaction deliberately reuses its address
      // pair, so advancing that pair after the transaction is not valid.
      const size_t external_loop_begin = words.size();
      const auto mov_address_lo = instrumentation::build_v_mov_b32_literal(
          address_lo_vgpr, static_cast<uint32_t>(exact_shadow_base), arch);
      const auto mov_address_hi = instrumentation::build_v_mov_b32_literal(
          address_hi_vgpr, static_cast<uint32_t>(exact_shadow_base >> 32u), arch);
      if (!mov_address_lo || !mov_address_hi) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow address");
        return std::nullopt;
      }
      words.insert(words.end(), mov_address_lo->begin(), mov_address_lo->end());
      words.insert(words.end(), mov_address_hi->begin(), mov_address_hi->end());

      const uint32_t entries_per_dispatch_bank =
          layout.exact_shadow_entry_capacity / layout.inline_exact_dispatch_bank_count;
      const uint32_t dispatch_bank_stride =
          entries_per_dispatch_bank * sizeof(ConSanMoiInlineExactShadowSlot);
      std::vector<uint32_t> copy_dispatch_id;
      if (options.moi_dispatch_id_sgpr) {
        copy_dispatch_id.push_back(
            build_v_mov_b32_e32(tmp_vgpr, *options.moi_dispatch_id_sgpr, arch));
      } else {
        const auto literal = instrumentation::build_v_mov_b32_literal(
            tmp_vgpr, static_cast<uint32_t>(options.moi_report_dispatch_id), arch);
        if (!literal) {
          errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode literal dispatch ID");
          return std::nullopt;
        }
        copy_dispatch_id = *literal;
      }
      const auto mix_workgroup_key = instrumentation::build_v_xor_b32(
          tmp_vgpr,
          vector_source_vgpr(loop_external_cells ? saved_workgroup_key_vgpr : workgroup_key_vgpr),
          tmp_vgpr, arch);
      const auto select_dispatch_bank = instrumentation::build_v_and_b32_literal(
          tmp_vgpr, layout.inline_exact_dispatch_bank_count - 1u, tmp_vgpr, arch);
      const auto scale_dispatch_bank = instrumentation::build_v_mul_lo_u32_literal(
          tmp_vgpr, old_value_vgpr, dispatch_bank_stride, tmp_vgpr, arch);
      const auto add_dispatch_bank =
          instrumentation::build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch);
      if (layout.inline_exact_dispatch_bank_count == 0u ||
          (layout.inline_exact_dispatch_bank_count &
           (layout.inline_exact_dispatch_bank_count - 1u)) != 0u ||
          entries_per_dispatch_bank == 0u || dispatch_bank_stride == 0u || !mix_workgroup_key ||
          !select_dispatch_bank || !scale_dispatch_bank || !add_dispatch_bank) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode dispatch bank");
        return std::nullopt;
      }
      words.insert(words.end(), copy_dispatch_id.begin(), copy_dispatch_id.end());
      words.push_back(*mix_workgroup_key);
      words.insert(words.end(), select_dispatch_bank->begin(), select_dispatch_bank->end());
      words.insert(words.end(), scale_dispatch_bank->begin(), scale_dispatch_bank->end());
      words.insert(words.end(), add_dispatch_bank->begin(), add_dispatch_bank->end());

      if (!reload_spill_backed_lds_byte_offset()) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not recover its spilled LDS address");
        return std::nullopt;
      }
      uint16_t effective_lds_byte_offset_vgpr = *lds_byte_offset_vgpr;
      if (range.static_byte_offset != 0) {
        if (!append_compute_effective_lds_byte_offset(words, tmp_vgpr, *lds_byte_offset_vgpr,
                                                      range.static_byte_offset, arch)) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not encode LDS byte offset");
          return std::nullopt;
        }
        effective_lds_byte_offset_vgpr = tmp_vgpr;
      }

      const auto start_cell_shift = instrumentation::build_v_lshrrev_b32(
          tmp_vgpr, scalar_positive_inline_u32(consan_moi_exact_shadow::granule_shift),
          effective_lds_byte_offset_vgpr, arch);
      const auto byte_index_shift = instrumentation::build_v_lshlrev_b32(
          tmp_vgpr, scalar_positive_inline_u32(3), tmp_vgpr, arch);
      const auto address_with_low_cell =
          instrumentation::build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch);
      const auto high_cell_shift = instrumentation::build_v_lshlrev_b32(
          tmp_vgpr, scalar_positive_inline_u32(1), tmp_vgpr, arch);
      const auto address_with_high_cell =
          instrumentation::build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch);
      if (!start_cell_shift || !byte_index_shift || !address_with_low_cell || !high_cell_shift ||
          !address_with_high_cell) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow publish");
        return std::nullopt;
      }
      words.push_back(*start_cell_shift);
      words.push_back(*byte_index_shift);
      words.insert(words.end(), address_with_low_cell->begin(), address_with_low_cell->end());
      words.push_back(*high_cell_shift);
      words.insert(words.end(), address_with_high_cell->begin(), address_with_high_cell->end());
      if (loop_external_cells) {
        const auto scale_cell = instrumentation::build_v_mul_lo_u32_literal(
            tmp_vgpr, old_value_vgpr, sizeof(ConSanMoiInlineExactShadowSlot), loop_counter_vgpr,
            arch);
        const auto add_cell =
            instrumentation::build_v_add_u64_vgpr_offset(address_lo_vgpr, tmp_vgpr, arch);
        if (!scale_cell || !add_cell) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not encode looped cell offset");
          return std::nullopt;
        }
        words.insert(words.end(), scale_cell->begin(), scale_cell->end());
        words.insert(words.end(), add_cell->begin(), add_cell->end());
      } else if (cell_index != 0) {
        const auto add_cell = instrumentation::build_v_add_u64_signed_i24(
            address_lo_vgpr,
            static_cast<int32_t>(cell_index * sizeof(ConSanMoiInlineExactShadowSlot)), arch);
        if (!add_cell) {
          errors.emplace_back("ConSan MOI inline-shadow probe could not encode shadow cell offset");
          return std::nullopt;
        }
        words.insert(words.end(), add_cell->begin(), add_cell->end());
      }
      if (!append_wave_coalesced_exact_shadow_swap(
              words, candidate, options, arch, layout, address_lo_vgpr, low_vgpr, high_vgpr,
              old_value_vgpr, *lds_byte_offset_vgpr, range.static_byte_offset, range.byte_count,
              cell_index,
              loop_external_cells ? std::optional<uint16_t>(loop_counter_vgpr) : std::nullopt,
              spill, spill_backed_lds_byte_offset_source, errors)) {
        errors.emplace_back(
            "ConSan MOI inline-shadow probe could not encode versioned shadow publish");
        return std::nullopt;
      }
      if (loop_external_cells) {
        const auto advance_counter = instrumentation::build_v_add_u32(
            loop_counter_vgpr, scalar_positive_inline_u32(1), loop_counter_vgpr, arch);
        const auto more_cells = instrumentation::build_v_cmp_gt_u32_vcc(
            scalar_positive_inline_u32(cell_range.cell_count), loop_counter_vgpr, arch);
        const uint16_t temporary_exec_sgpr =
            static_cast<uint16_t>(*options.moi_exec_save_sgpr + 14u);
        const uint16_t incoming_exec_sgpr =
            static_cast<uint16_t>(*options.moi_exec_save_sgpr + 12u);
        const auto narrow_more_cells =
            instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
        const auto restore_incoming_exec =
            instrumentation::build_s_mov_b64(kRdna4ExecLo, incoming_exec_sgpr, arch);
        if (!advance_counter || !more_cells || !narrow_more_cells || !restore_incoming_exec) {
          errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode its external cell loop");
          return std::nullopt;
        }
        words.insert(words.end(), advance_counter->begin(), advance_counter->end());
        words.push_back(*more_cells);
        words.push_back(*narrow_more_cells);
        const size_t loop_branch = words.size();
        const int64_t loop_delta =
            static_cast<int64_t>(external_loop_begin) - static_cast<int64_t>(loop_branch + 1u);
        if (loop_delta < std::numeric_limits<int16_t>::min() ||
            loop_delta > std::numeric_limits<int16_t>::max()) {
          errors.emplace_back(
              "ConSan MOI inline-shadow probe external cell loop exceeds SOPP range");
          return std::nullopt;
        }
        const auto loop =
            instrumentation::build_s_cbranch_execnz(static_cast<int16_t>(loop_delta), arch);
        if (!loop) {
          errors.emplace_back(
              "ConSan MOI inline-shadow probe could not encode its external cell loop branch");
          return std::nullopt;
        }
        words.push_back(*loop);
        words.push_back(*restore_incoming_exec);
      }
    }
  }
  if (workgroup_shadow) {
    const std::optional<uint16_t> visible_evidence_sgpr =
        inline_shadow_visible_evidence_sgpr(options);
    std::optional<size_t> skip_visible_evidence_index;
    if (visible_evidence_sgpr) {
      const auto already_published = instrumentation::build_s_cmp_lg_u32(
          *visible_evidence_sgpr, scalar_positive_inline_u32(0), arch);
      const auto skip_visible_evidence = instrumentation::build_s_cbranch_scc1(0, arch);
      if (!already_published || !skip_visible_evidence) {
        errors.emplace_back("ConSan MOI inline-shadow probe could not test visible evidence");
        return std::nullopt;
      }
      words.push_back(*already_published);
      skip_visible_evidence_index = words.size();
      words.push_back(*skip_visible_evidence);
    }
    const uint16_t exec_base = *options.moi_exec_save_sgpr;
    const uint16_t active_exec_sgpr = static_cast<uint16_t>(exec_base + 12u);
    const uint16_t temporary_exec_sgpr = static_cast<uint16_t>(exec_base + 14u);
    const auto save_active_exec =
        instrumentation::build_s_mov_b64(active_exec_sgpr, kRdna4ExecLo, arch);
    const auto mbcnt_lo = instrumentation::build_v_mbcnt_lo_u32_b32(
        tmp_vgpr, active_exec_sgpr, scalar_positive_inline_u32(0), arch);
    const auto mbcnt_hi = instrumentation::build_v_mbcnt_hi_u32_b32(
        tmp_vgpr, static_cast<uint16_t>(active_exec_sgpr + 1u), vector_source_vgpr(tmp_vgpr), arch);
    const auto first_active =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), tmp_vgpr, arch);
    const auto narrow_first =
        instrumentation::build_s_and_saveexec_b64(temporary_exec_sgpr, kRdna4VccLo, arch);
    const auto saved_exec_wait = instrumentation::build_salu_to_valu_dependency_wait(arch);
    if (!save_active_exec || !saved_exec_wait || !mbcnt_lo || !mbcnt_hi || !first_active ||
        !narrow_first) {
      errors.emplace_back("ConSan MOI inline-shadow probe could not elect visible evidence");
      return std::nullopt;
    }
    words.push_back(*save_active_exec);
    words.push_back(*saved_exec_wait);
    words.insert(words.end(), mbcnt_lo->begin(), mbcnt_lo->end());
    words.insert(words.end(), mbcnt_hi->begin(), mbcnt_hi->end());
    words.push_back(*first_active);
    words.push_back(*narrow_first);
    if (!append_publish_visible_evidence_if_zero(
            words,
            *options.moi_report_buffer_address + offsetof(ConSanMoiReportHeader, event_counter),
            tmp_vgpr, address_lo_vgpr, temporary_exec_sgpr, arch)) {
      errors.emplace_back("ConSan MOI inline-shadow probe could not publish visible evidence");
      return std::nullopt;
    }
    if (visible_evidence_sgpr) {
      words.push_back(build_s_mov_b32(*visible_evidence_sgpr, scalar_positive_inline_u32(1), arch));
      const int64_t skip_delta = static_cast<int64_t>(words.size()) -
                                 static_cast<int64_t>(*skip_visible_evidence_index + 1u);
      if (skip_delta < std::numeric_limits<int16_t>::min() ||
          skip_delta > std::numeric_limits<int16_t>::max()) {
        errors.emplace_back("ConSan MOI inline-shadow visible-evidence branch exceeds SOPP range");
        return std::nullopt;
      }
      words[*skip_visible_evidence_index] =
          *instrumentation::build_s_cbranch_scc1(static_cast<int16_t>(skip_delta), arch);
    }
  }
  if (options.moi_exec_save_sgpr) {
    const auto restore_original_exec = instrumentation::build_s_mov_b64(
        kRdna4ExecLo,
        static_cast<uint16_t>(*options.moi_exec_save_sgpr + original_exec_save_offset), arch);
    if (!restore_original_exec) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not restore workgroup-filtered state");
      return std::nullopt;
    }
    words.push_back(*restore_original_exec);
    if (preserve_special_state && !append_restore_moi_special_state(words, options, arch)) {
      errors.emplace_back(
          "ConSan MOI inline-shadow probe could not restore workgroup-filtered state");
      return std::nullopt;
    }
  }
  const bool reserve_two_address_replay_scratch =
      arch == ROCJITSU_CODE_ARCH_GFX1250 &&
      two_address_native_lds_offset_scale(candidate.mnemonic).value_or(0u) > 8u;
  const auto guest_words = build_moi_relocated_guest_access_words(
      bytes, candidate, arch, *lds_byte_offset_vgpr,
      reserve_two_address_replay_scratch ? std::optional<uint16_t>(static_cast<uint16_t>(
                                               *options.scratch_vgpr + scratch_count - 1u))
                                         : std::nullopt,
      errors);
  if (!guest_words)
    return std::nullopt;
  if (guest_instruction_word_count)
    *guest_instruction_word_count = static_cast<uint32_t>(guest_words->size());
  words.insert(words.end(), guest_words->begin(), guest_words->end());
  return words;
}

[[nodiscard]] bool append_moi_scc_preserving_indirect_jump(
    std::vector<uint32_t> &words, uint64_t words_text_offset, uint64_t target_text_offset,
    uint16_t pc_sgpr, uint16_t saved_scc_sgpr, bool capture_scc, rj_code_arch_t arch);
