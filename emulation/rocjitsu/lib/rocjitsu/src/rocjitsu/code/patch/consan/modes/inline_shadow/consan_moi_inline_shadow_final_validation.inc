void validate_inline_exact_shadow_semantics(const FinalValidationEnvironment &environment,
                                            const ConSanFinalValidationInput &result,
                                            uint64_t expected_moi_report_dispatch_id,
                                            std::vector<std::string> &errors) {
  const auto is_exact_patch = [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  };
  if (!std::ranges::any_of(result.patches, is_exact_patch))
    return;
  if (!environment.require_target(errors, "ConSan exact-shadow proof found no target profile"))
    return;
  const rj_code_arch_t arch = environment.arch();
  const bool has_exact_patch_without_dispatch =
      std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
        return is_exact_patch(patch) && !patch.workgroup_shadow &&
               (!patch.moi_dispatch_identity_source ||
                !patch.moi_dispatch_identity_source->is_well_formed());
      });
  if (!environment.replacement_text.available || has_exact_patch_without_dispatch) {
    errors.emplace_back(
        "ConSan final validation found exact-shadow publication without dispatch-ID state");
    return;
  }

  if (!environment.decoder) {
    errors.emplace_back(
        "ConSan final validation could not create an exact-shadow semantic decoder");
    return;
  }
  const std::span<const uint8_t> text = environment.replacement_text.bytes;

  for (const ConSanPatchInfo &patch : result.patches) {
    if (!is_exact_patch(patch))
      continue;
    const uint64_t body_offset =
        patch.trampoline_size != 0u ? patch.trampoline_offset : patch.anchor_offset;
    const uint64_t body_size =
        patch.trampoline_size != 0u ? patch.trampoline_size : patch.original_size;
    if (!patch.scratch_vgpr || body_offset > text.size() || body_size > text.size() - body_offset ||
        body_size % sizeof(uint32_t) != 0) {
      errors.emplace_back(
          "ConSan final validation found invalid exact-shadow generated-body bounds");
      continue;
    }

    const std::span<const uint8_t> body = text.subspan(body_offset, body_size);
    size_t decoded_cas_count = 0;
    size_t decoded_local_exchange_count = 0;
    size_t decoded_local_exchange_b32_count = 0;
    size_t decoded_local_exchange_b64_count = 0;
    size_t decoded_backward_local_loop_count = 0;
    bool decoded_nonbackward_local_loop = false;
    bool decoded_legacy_swap = false;
    size_t cursor = 0;
    while (cursor < body.size()) {
      std::array<uint32_t, 4> words{};
      std::memcpy(words.data(), body.data() + cursor,
                  std::min(words.size() * sizeof(uint32_t), body.size() - cursor));
      std::unique_ptr<Instruction> instruction =
          decode_bounded_instruction(*environment.decoder,
                                     std::span<const uint32_t>(words).first(std::min(
                                         words.size(), (body.size() - cursor) / sizeof(uint32_t))),
                                     body_offset + cursor);
      if (!instruction) {
        errors.emplace_back(
            "ConSan final validation could not re-decode exact-shadow generated body");
        break;
      }
      if (!instruction || instruction->size() <= 0 ||
          static_cast<size_t>(instruction->size()) > body.size() - cursor) {
        errors.emplace_back(
            "ConSan final validation found invalid exact-shadow instruction extent");
        break;
      }
      const std::string_view mnemonic = instruction->mnemonic();
      decoded_cas_count +=
          (mnemonic == "flat_atomic_cmpswap" || mnemonic.starts_with("flat_atomic_cmpswap_b32"))
              ? 1u
              : 0u;
      const bool local_exchange_b32 = mnemonic.starts_with("ds_storexchg_rtn_b32");
      const bool local_exchange_b64 =
          mnemonic.starts_with("ds_storexchg_rtn_b64") || mnemonic.starts_with("ds_wrxchg_rtn_b64");
      decoded_local_exchange_b32_count += local_exchange_b32 ? 1u : 0u;
      decoded_local_exchange_b64_count += local_exchange_b64 ? 1u : 0u;
      decoded_local_exchange_count += local_exchange_b32 || local_exchange_b64 ? 1u : 0u;
      if (mnemonic == "s_cbranch_execnz") {
        const int16_t displacement = static_cast<int16_t>(words.front());
        decoded_backward_local_loop_count += displacement < 0 ? 1u : 0u;
        decoded_nonbackward_local_loop |= displacement >= 0;
      }
      decoded_legacy_swap |=
          mnemonic == "flat_atomic_swap_x2" || mnemonic.starts_with("flat_atomic_swap_b64");
      cursor += static_cast<size_t>(instruction->size());
    }

    const uint16_t scratch = *patch.scratch_vgpr;
    std::vector<uint32_t> body_words(body.size() / sizeof(uint32_t));
    std::memcpy(body_words.data(), body.data(), body.size());
    const auto count_pattern = [&body_words](std::span<const uint32_t> pattern) {
      size_t count = 0;
      auto position = body_words.cbegin();
      while ((position = std::search(position, body_words.cend(), pattern.begin(),
                                     pattern.end())) != body_words.cend()) {
        ++count;
        position += static_cast<std::ptrdiff_t>(pattern.size());
      }
      return count;
    };
    if (patch.workgroup_shadow) {
      const ConSanMoiWorkgroupShadowLayout &workgroup_shadow = *patch.workgroup_shadow;
      const auto intent = std::ranges::find_if(
          result.observation_plan().probe_intents, [&](const ConSanProbeIntent &candidate) {
            return candidate.kind == ConSanProbeIntentKind::ExactShadowAccess &&
                   candidate.physical_site.original_text_offset == patch.anchor_offset;
          });
      const ConSanProgramSite *access =
          intent == result.observation_plan().probe_intents.end()
              ? nullptr
              : result.program_inventory.program_site(intent->source_site);
      const uint32_t minimum_cell_count =
          access == nullptr || !access->has_access()
              ? 0u
              : consan_moi_maximum_cell_count_for_unaligned_bytes(access->decoded_width_bits / 8u);
      const size_t decoded_local_exchange_cell_count = decoded_local_exchange_b64_count;
      const bool needs_publication_loop = minimum_cell_count > decoded_local_exchange_cell_count;
      // A bitmap-backed lazy mirror contributes one readiness-poll loop per
      // logical exchange cell. A wide access whose cells share one generated
      // exchange must additionally retain its outer publication loop; merely
      // finding any backward branch would let a bitmap poll mask its removal.
      const bool bitmap_backed_lazy_shadow =
          workgroup_shadow.lazy_initialization && workgroup_shadow.validity_size != 0u;
      const size_t minimum_backward_loop_count =
          (bitmap_backed_lazy_shadow ? decoded_local_exchange_cell_count : 0u) +
          (needs_publication_loop ? 1u : 0u);
      const uint16_t alignment = environment.target_profile->flat_compare_swap_data_pair_alignment;
      const uint16_t old_value_candidate = static_cast<uint16_t>(scratch + 5u);
      const uint16_t old_value_offset =
          static_cast<uint16_t>(5u + (alignment - old_value_candidate % alignment) % alignment);
      const auto expected_local_exchange_b64 = instrumentation::build_ds_storexchg_rtn_b64(
          static_cast<uint16_t>(scratch + old_value_offset), scratch,
          static_cast<uint16_t>(scratch + 2u), 0u, arch);
      // Exact-byte publication reserves a named tuple in the mode-local
      // scratch ABI. Validation reconstructs its instruction independently.
      constexpr uint16_t diagnostic_claim_offset =
          consan_moi_impl::inline_register_layout::ExactShadow::diagnostic_claim_tuple;
      const auto expected_first_diagnostic_claim = instrumentation::build_flat_atomic_cmpswap_b32(
          scratch, static_cast<uint16_t>(scratch + diagnostic_claim_offset),
          static_cast<uint16_t>(scratch + diagnostic_claim_offset),
          /*return_old_value=*/true,
          /*scope=*/2, arch);
      if (!expected_local_exchange_b64 || !expected_first_diagnostic_claim || decoded_legacy_swap ||
          decoded_cas_count != decoded_local_exchange_cell_count ||
          decoded_local_exchange_count == 0u || access == nullptr || !access->has_access() ||
          decoded_backward_local_loop_count < minimum_backward_loop_count ||
          decoded_nonbackward_local_loop || decoded_local_exchange_b32_count != 0u ||
          count_pattern(*expected_local_exchange_b64) != decoded_local_exchange_b64_count ||
          count_pattern(*expected_first_diagnostic_claim) != decoded_cas_count) {
        errors.emplace_back(
            "ConSan final validation found invalid workgroup-local exact-shadow publication "
            "semantics: decoded_cas=" +
            std::to_string(decoded_cas_count) +
            " decoded_local_exchange=" + std::to_string(decoded_local_exchange_count) +
            " decoded_local_cells=" + std::to_string(decoded_local_exchange_cell_count) +
            " minimum_cells=" + std::to_string(minimum_cell_count) +
            " backward_loops=" + std::to_string(decoded_backward_local_loop_count) +
            " minimum_backward_loops=" + std::to_string(minimum_backward_loop_count) +
            " nonbackward_loop=" + std::to_string(decoded_nonbackward_local_loop) +
            " legacy_swap=" + std::to_string(decoded_legacy_swap) + " decoded_exchange_b32=" +
            std::to_string(decoded_local_exchange_b32_count) + " expected_exchange_b64=" +
            std::to_string(expected_local_exchange_b64 ? count_pattern(*expected_local_exchange_b64)
                                                       : 0u) +
            " expected_claim=" +
            std::to_string(expected_first_diagnostic_claim
                               ? count_pattern(*expected_first_diagnostic_claim)
                               : 0u));
      }
      continue;
    }

    const auto expected_cas = instrumentation::build_flat_atomic_cmpswap_b32(
        scratch,
        static_cast<uint16_t>(
            scratch + consan_moi_impl::inline_register_layout::ExactShadow::diagnostic_claim_tuple),
        static_cast<uint16_t>(
            scratch + consan_moi_impl::inline_register_layout::ExactShadow::diagnostic_claim_tuple),
        /*return_old_value=*/true, /*scope=*/2, arch);
    const auto expected_dispatch_low_store = instrumentation::build_flat_store_b32(
        scratch,
        static_cast<uint16_t>(
            scratch + consan_moi_impl::inline_register_layout::ExactShadow::dispatch_id_low),
        arch, offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id));
    const auto expected_dispatch_high_store = instrumentation::build_flat_store_b32(
        scratch,
        static_cast<uint16_t>(
            scratch + consan_moi_impl::inline_register_layout::ExactShadow::dispatch_id_high),
        arch, offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) + sizeof(uint32_t));
    if (!expected_cas || !expected_dispatch_low_store || !expected_dispatch_high_store) {
      errors.emplace_back(
          "ConSan final validation could not encode exact-shadow semantic expectations");
      continue;
    }

    const size_t exact_cas_count = count_pattern(*expected_cas);
    const size_t transaction_count = exact_cas_count / 2u;
    std::vector<uint32_t> capture_low;
    std::vector<uint32_t> capture_high;
    if (!patch.moi_dispatch_identity_source ||
        !patch.moi_dispatch_identity_source->is_well_formed()) {
      errors.emplace_back(
          "ConSan final validation found exact-shadow patch without a committed dispatch-ID "
          "source");
      continue;
    }
    const ConSanMoiDispatchIdentitySource &dispatch = *patch.moi_dispatch_identity_source;
    bool dispatch_source_matches_request = true;
    if (dispatch.sgpr) {
      capture_low.push_back(
          build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 2u), *dispatch.sgpr, arch));
      capture_high.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 3u),
                                                 static_cast<uint16_t>(*dispatch.sgpr + 1u), arch));
    } else if (dispatch.vgpr) {
      capture_low.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 2u),
                                                vector_source_vgpr(*dispatch.vgpr), arch));
      capture_high.push_back(build_v_mov_b32_e32(
          static_cast<uint16_t>(scratch + 3u),
          vector_source_vgpr(static_cast<uint16_t>(*dispatch.vgpr + 1u)), arch));
    } else if (dispatch.private_offset) {
      const auto low = instrumentation::build_private_load_b32(static_cast<uint16_t>(scratch + 2u),
                                                               *dispatch.private_offset, arch);
      const auto high = instrumentation::build_private_load_b32(
          static_cast<uint16_t>(scratch + 3u), *dispatch.private_offset + SpillManager::kSlotBytes,
          arch);
      const auto wait = instrumentation::build_s_wait_private_load0(arch);
      if (!low || !high || !wait) {
        errors.emplace_back(
            "ConSan final validation could not encode private dispatch-ID expectations");
        continue;
      }
      capture_low = *low;
      capture_high = *high;
    } else if (dispatch.literal) {
      dispatch_source_matches_request = *dispatch.literal == expected_moi_report_dispatch_id;
      const auto low = instrumentation::build_v_mov_b32_literal(
          static_cast<uint16_t>(scratch + 2u), static_cast<uint32_t>(*dispatch.literal), arch);
      const auto high = instrumentation::build_v_mov_b32_literal(
          static_cast<uint16_t>(scratch + 3u), static_cast<uint32_t>(*dispatch.literal >> 32u),
          arch);
      if (!low || !high) {
        errors.emplace_back(
            "ConSan final validation could not encode literal dispatch-ID expectations");
        continue;
      }
      capture_low = *low;
      capture_high = *high;
    }
    // One composite-key partition body owns each generated cell. Its odd
    // claim and even commit are the two required CAS operations; distinct
    // address/access/mask groups revisit the same body at runtime rather than
    // requiring a duplicated uniform/fallback transaction in the image.
    if (!dispatch_source_matches_request || decoded_legacy_swap || decoded_cas_count < 2u ||
        decoded_cas_count % 2u != 0u || exact_cas_count != decoded_cas_count ||
        count_pattern(*expected_dispatch_low_store) != transaction_count ||
        count_pattern(*expected_dispatch_high_store) != transaction_count ||
        count_pattern(capture_low) < transaction_count ||
        count_pattern(capture_high) < transaction_count) {
      errors.emplace_back(
          "ConSan final validation found invalid versioned exact-shadow publication semantics: "
          "anchor=" +
          std::to_string(patch.anchor_offset) + " scratch=v" + std::to_string(scratch) +
          " decoded_cas=" + std::to_string(decoded_cas_count) + " exact_cas=" +
          std::to_string(exact_cas_count) + " transactions=" + std::to_string(transaction_count) +
          " dispatch_low_stores=" + std::to_string(count_pattern(*expected_dispatch_low_store)) +
          " dispatch_high_stores=" + std::to_string(count_pattern(*expected_dispatch_high_store)) +
          " capture_low=" + std::to_string(count_pattern(capture_low)) +
          " capture_high=" + std::to_string(count_pattern(capture_high)) +
          " dispatch_source_matches_request=" + std::to_string(dispatch_source_matches_request) +
          " owner_count=" + std::to_string(patch.owner_descriptor_file_offsets.size()) +
          " legacy_swap=" + std::to_string(decoded_legacy_swap));
    }
  }
  return;
}

void validate_inline_release_transaction_semantics(const FinalValidationEnvironment &environment,
                                                   const ConSanFinalValidationInput &result,
                                                   std::vector<std::string> &errors) {
  constexpr uint16_t kVccLo = 106u;
  const auto is_release_transaction = [&](const ConSanPatchInfo &patch) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiInlineAtomicOrdering)
      return false;
    const auto event = std::ranges::find_if(
        result.program_inventory.sync().sync_events, [&](const ConSanSyncEvent &item) {
          return item.kind == ConSanSyncKind::Atomic && item.text_offset() == patch.anchor_offset;
        });
    if (event == result.program_inventory.sync().sync_events.end())
      return false;
    const ConSanSyncSequence *sequence =
        result.program_inventory.sync().find_unique_sequence_containing(event->semantic_id);
    return sequence != nullptr &&
           (sequence->memory_role == ConSanSyncMemoryRole::Release ||
            sequence->memory_role == ConSanSyncMemoryRole::AcquireRelease ||
            sequence->memory_role == ConSanSyncMemoryRole::SequentiallyConsistent);
  };
  if (!std::ranges::any_of(result.patches, is_release_transaction))
    return;
  if (!environment.replacement_text.available) {
    errors.emplace_back(
        "ConSan final validation found release transaction without executable text");
    return;
  }

  if (!environment.require_target(
          errors, "ConSan release-transaction proof found no target profile",
          "ConSan final validation could not create a release-transaction semantic decoder"))
    return;
  const rj_code_arch_t arch = environment.arch();
  const std::span<const uint8_t> text = environment.replacement_text.bytes;

  const auto atomic_site_for = [&](uint64_t text_offset) -> const ConSanAtomicSite * {
    for (const ConSanProgramSite &decoded : result.program_inventory.program_sites()) {
      if (decoded.text_offset() == text_offset) {
        if (const ConSanAtomicSite *site = decoded.get_if<ConSanAtomicSite>())
          return site;
      }
    }
    return nullptr;
  };

  for (const ConSanPatchInfo &patch : result.patches) {
    if (!is_release_transaction(patch))
      continue;
    if (!patch.scratch_vgpr || !result.moi_operating_point.moi_exec_save_sgpr ||
        patch.trampoline_offset > text.size() ||
        patch.trampoline_size > text.size() - patch.trampoline_offset ||
        patch.trampoline_size % sizeof(uint32_t) != 0) {
      errors.emplace_back(
          "ConSan final validation found invalid versioned release transaction bounds");
      continue;
    }
    const std::span<const uint8_t> body =
        text.subspan(patch.trampoline_offset, patch.trampoline_size);
    size_t cursor = 0;
    bool decode_failed = false;
    while (cursor < body.size()) {
      std::array<uint32_t, 4> words{};
      std::memcpy(words.data(), body.data() + cursor,
                  std::min(words.size() * sizeof(uint32_t), body.size() - cursor));
      std::unique_ptr<Instruction> instruction =
          decode_bounded_instruction(*environment.decoder,
                                     std::span<const uint32_t>(words).first(std::min(
                                         words.size(), (body.size() - cursor) / sizeof(uint32_t))),
                                     patch.trampoline_offset + cursor);
      if (!instruction) {
        decode_failed = true;
      }
      if (!instruction || instruction->size() <= 0 ||
          static_cast<size_t>(instruction->size()) > body.size() - cursor) {
        decode_failed = true;
        break;
      }
      cursor += static_cast<size_t>(instruction->size());
    }
    if (decode_failed) {
      errors.emplace_back(
          "ConSan final validation could not re-decode versioned release transaction");
      continue;
    }

    std::vector<uint32_t> body_words(body.size() / sizeof(uint32_t));
    std::memcpy(body_words.data(), body.data(), body.size());
    const auto positions = [&](std::span<const uint32_t> pattern) {
      std::vector<size_t> result_positions;
      auto position = body_words.cbegin();
      while ((position = std::search(position, body_words.cend(), pattern.begin(),
                                     pattern.end())) != body_words.cend()) {
        result_positions.push_back(
            static_cast<size_t>(std::distance(body_words.cbegin(), position)));
        position += static_cast<std::ptrdiff_t>(pattern.size());
      }
      return result_positions;
    };
    const uint16_t scratch = *patch.scratch_vgpr;
    const uint16_t alignment = environment.target_profile->flat_compare_swap_data_pair_alignment;
    const uint16_t snapshot_candidate = static_cast<uint16_t>(scratch + 5u);
    const uint16_t snapshot_address = static_cast<uint16_t>(
        snapshot_candidate + (alignment - snapshot_candidate % alignment) % alignment);
    using SnapshotLayout = consan_moi_impl::inline_register_layout::AtomicCausalSnapshot;
    const uint16_t temporary = static_cast<uint16_t>(scratch + SnapshotLayout::temporary);
    const uint16_t version_cas_candidate = static_cast<uint16_t>(scratch + 5u);
    const uint16_t version_cas_value =
        static_cast<uint16_t>(version_cas_candidate - version_cas_candidate % alignment);
    const auto version_cas = instrumentation::build_flat_atomic_cmpswap_b32(
        scratch, version_cas_value, version_cas_value, /*return_old_value=*/true, /*scope=*/2,
        arch);
    uint16_t owner_vgpr = 0;
    if (patch.private_state_layout || patch.persistent_sgpr_state.owner()) {
      const auto resource_plan = std::ranges::find_if(result.resource_plans, [&](const auto &plan) {
        return plan.site_kind == ConSanResourceSiteKind::Atomic &&
               plan.text_offset == patch.anchor_offset && plan.scratch_vgpr == patch.scratch_vgpr;
      });
      if (resource_plan == result.resource_plans.end() || resource_plan->scratch_vgpr_count < 4u) {
        errors.emplace_back(
            "ConSan final validation could not resolve release-transaction scalar owner state");
        continue;
      }
      // Scalar-persistent and owner-private atomic emission both materialize
      // owner/epoch in the four-register tail reserved by the selected
      // resource plan. Derive the owner from that plan rather than duplicating
      // an architecture-specific scratch-window size in validation.
      owner_vgpr = static_cast<uint16_t>(scratch + resource_plan->scratch_vgpr_count - 4u);
    } else if (patch.moi_vgpr_state) {
      owner_vgpr = patch.moi_vgpr_state->owner_epoch.owner;
    } else {
      errors.emplace_back(
          "ConSan final validation could not resolve release-transaction owner state");
      continue;
    }
    const auto owner_store = instrumentation::build_flat_store_b32(
        scratch, owner_vgpr, arch, offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id));
    const auto wait_store = instrumentation::build_s_wait_global_store0(arch);
    if (!version_cas || !owner_store || !wait_store) {
      errors.emplace_back(
          "ConSan final validation could not encode release-transaction expectations");
      continue;
    }
    const std::vector<size_t> cas_positions = positions(*version_cas);
    const std::vector<size_t> owner_positions = positions(*owner_store);
    const size_t owner_between =
        cas_positions.size() >= 2u
            ? std::ranges::count_if(owner_positions,
                                    [&](size_t position) {
                                      return cas_positions.front() < position &&
                                             position < cas_positions.back();
                                    })
            : 0u;
    // A claimed AcquireRelease transaction emits five unrolled acquired-token
    // consumer stores (one direct and four bounded ancestors) before writing
    // the successor release owner. All six stores intentionally use scratch+0
    // and ABI offset 4, so their encodings are identical; release-only
    // transactions retain the single-store shape.
    const ConSanAtomicSite *site = atomic_site_for(patch.anchor_offset);
    const bool compare_exchange = site && consan_atomic_is_compare_exchange(*site);
    const bool claimed_compare_exchange = compare_exchange && cas_positions.size() == 3u;
    const size_t publication_begin =
        claimed_compare_exchange
            ? cas_positions[1]
            : (cas_positions.empty() ? std::numeric_limits<size_t>::max() : cas_positions.front());
    const size_t publisher_owner_count =
        cas_positions.size() >= 2u ? std::ranges::count_if(owner_positions,
                                                           [&](size_t position) {
                                                             return publication_begin < position &&
                                                                    position < cas_positions.back();
                                                           })
                                   : 0u;
    bool valid = ((!compare_exchange && cas_positions.size() == 2u) || claimed_compare_exchange) &&
                 (owner_between == 1u || owner_between == 6u) &&
                 (!claimed_compare_exchange || publisher_owner_count == 1u);

    const auto version_before = instrumentation::build_flat_load_b32(
        snapshot_address, static_cast<uint16_t>(scratch + SnapshotLayout::token_version_before),
        arch, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version));
    const auto version_after = instrumentation::build_flat_load_b32(
        snapshot_address, temporary, arch,
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version));
    bool token_loads_valid = version_before && version_after &&
                             !positions(*version_before).empty() &&
                             !positions(*version_after).empty();
    for (size_t offset : kConSanMoiInlineAcquiredEpochTokenPayloadOffsets) {
      const auto load =
          instrumentation::build_flat_load_b32(snapshot_address, temporary, arch, offset);
      token_loads_valid &= load && !positions(*load).empty();
    }
    valid &= token_loads_valid;

    const auto require_snapshot_store = [&](size_t offset, uint16_t source) {
      const auto store =
          instrumentation::build_flat_store_b32(snapshot_address, source, arch, offset);
      if (!store)
        return false;
      const std::vector<size_t> store_positions = positions(*store);
      return store_positions.size() == 1u && !cas_positions.empty() &&
             publication_begin < store_positions.front() &&
             std::ranges::any_of(
                 owner_positions,
                 [&](size_t owner_position) {
                   return store_positions.front() < owner_position &&
                          owner_position < cas_positions.back();
                 });
    };
    bool snapshot_stores_valid = require_snapshot_store(
        offsetof(ConSanMoiInlineCausalSnapshot, entry_count),
        static_cast<uint16_t>(scratch + SnapshotLayout::count(alignment > 1u)));
    snapshot_stores_valid &=
        require_snapshot_store(offsetof(ConSanMoiInlineCausalSnapshot, flags),
                               static_cast<uint16_t>(scratch + SnapshotLayout::flags));
    for (uint16_t i = 0; i < 4u; ++i) {
      const size_t entry_offset = offsetof(ConSanMoiInlineCausalSnapshot, entries) +
                                  i * sizeof(ConSanMoiInlineCausalSnapshotEntry);
      snapshot_stores_valid &= require_snapshot_store(
          entry_offset, static_cast<uint16_t>(scratch + SnapshotLayout::owners + i));
      snapshot_stores_valid &=
          require_snapshot_store(entry_offset + sizeof(uint32_t),
                                 static_cast<uint16_t>(scratch + SnapshotLayout::epochs + i));
    }
    valid &= snapshot_stores_valid;
    const size_t wait_store_count = positions(std::array<uint32_t, 1>{*wait_store}).size();
    valid &= wait_store_count >= 2u;

    if (compare_exchange) {
      if (!site->data_vgpr || !site->destination_vgpr) {
        valid = false;
      } else {
        const uint16_t value_word_count = static_cast<uint16_t>(site->width_bits / 32u);
        const uint16_t compare_vgpr = static_cast<uint16_t>(*site->data_vgpr + value_word_count);
        const auto success = instrumentation::build_v_cmp_eq_u32_vcc(
            vector_source_vgpr(compare_vgpr), *site->destination_vgpr, arch);
        const auto narrow = instrumentation::build_s_and_saveexec_b64(
            *result.moi_operating_point.moi_exec_save_sgpr, kVccLo, arch);
        if (!success || !narrow) {
          valid = false;
        } else {
          std::vector<uint32_t> success_sequence = {*success, *narrow};
          if (site->width_bits == 64u) {
            const auto success_high = instrumentation::build_v_cmp_eq_u32_vcc(
                vector_source_vgpr(static_cast<uint16_t>(compare_vgpr + 1u)),
                static_cast<uint16_t>(*site->destination_vgpr + 1u), arch);
            if (!success_high) {
              valid = false;
            } else {
              success_sequence.push_back(*success_high);
              success_sequence.push_back(*narrow);
            }
          }
          const std::vector<size_t> success_positions = positions(success_sequence);
          const bool guest_offset_valid =
              patch.relocated_guest_instruction_offset &&
              *patch.relocated_guest_instruction_offset >= patch.trampoline_offset &&
              (*patch.relocated_guest_instruction_offset - patch.trampoline_offset) %
                      sizeof(uint32_t) ==
                  0u;
          const size_t guest_position =
              guest_offset_valid ? static_cast<size_t>((*patch.relocated_guest_instruction_offset -
                                                        patch.trampoline_offset) /
                                                       sizeof(uint32_t))
                                 : body_words.size();
          // The release slot must already be odd when the guest CAS exposes
          // its new value. Both complete post-guest outcome qualifications
          // precede the failed-comparison rollback, and only then may
          // successor metadata be staged and committed.
          valid &= claimed_compare_exchange && guest_offset_valid &&
                   guest_position < body_words.size() && success_positions.size() == 2u &&
                   cas_positions.front() < guest_position &&
                   guest_position < success_positions.front() &&
                   success_positions.front() < success_positions.back() &&
                   success_positions.back() < cas_positions[1];
        }
      }
    }
    if (!valid) {
      errors.emplace_back("ConSan final validation found invalid versioned release transaction "
                          "semantics (cas=" +
                          std::to_string(cas_positions.size()) +
                          " owner_between=" + std::to_string(owner_between) +
                          " publisher_owner=" + std::to_string(publisher_owner_count) +
                          " token_loads=" + (token_loads_valid ? "yes" : "no") +
                          " snapshot_stores=" + (snapshot_stores_valid ? "yes" : "no") +
                          " waits=" + std::to_string(wait_store_count) + ")");
    }
  }
  return;
}
