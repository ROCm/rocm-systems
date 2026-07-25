namespace {

struct UnmatchedBarrierWait {
  const ConSanBarrierSite *site = nullptr;
  std::optional<uint64_t> owner_descriptor_file_offset;
};

template <typename Container>
void append_unmatched_barrier_waits(const Container &container, bool in_kernel,
                                    const ConSanOptions &options, const ConSanResult &result,
                                    std::vector<UnmatchedBarrierWait> &waits) {
  if (!options.test_kernel_name_filter.empty() &&
      container.name.find(options.test_kernel_name_filter) == std::string::npos)
    return;
  for (const ConSanBarrierSite &site : container.barrier_sites) {
    if (site.operation != ConSanBarrierSite::Operation::Wait || site.size != sizeof(uint32_t) ||
        !site.barrier_id || site.operand_source != ConSanBarrierSite::OperandSource::Immediate)
      continue;
    const ConSanSyncEvent *event = nullptr;
    size_t event_count = 0;
    for (const ConSanSyncEvent &candidate : result.sync_events) {
      if (candidate.kind != ConSanSyncEventKind::Barrier ||
          candidate.operation != ConSanSyncOperation::BarrierWait ||
          candidate.container_name != container.name || candidate.in_kernel != in_kernel ||
          candidate.text_offset != site.text_offset || candidate.size != site.size)
        continue;
      event = &candidate;
      ++event_count;
    }
    if (event_count != 1u || event == nullptr)
      continue;
    const bool belongs_to_sequence =
        std::ranges::any_of(result.sync_sequences, [&](const ConSanSyncSequence &sequence) {
          return sequence.kind == ConSanSyncSequenceKind::Barrier &&
                 sequence.member_event_identities.size() > 1u &&
                 std::ranges::find(sequence.member_event_identities, event->identity) !=
                     sequence.member_event_identities.end();
        });
    if (belongs_to_sequence)
      continue;
    std::optional<uint64_t> owner;
    if constexpr (std::is_same_v<Container, ConSanKernelInfo>)
      owner = container.descriptor_file_offset;
    waits.push_back({&site, owner});
  }
}

void try_apply_unmatched_barrier_wait_abort(std::span<const uint8_t> original_bytes,
                                            const ConSanOptions &options, ConSanResult &result) {
  if (!options.abort_unmatched_barrier_wait || options.fault_dry_run || !result.errors.empty())
    return;
  std::vector<UnmatchedBarrierWait> waits;
  for (const ConSanKernelInfo &kernel : result.kernels)
    append_unmatched_barrier_waits(kernel, true, options, result, waits);
  for (const ConSanFunctionInfo &function : result.functions)
    append_unmatched_barrier_waits(function, false, options, result, waits);
  if (waits.empty())
    return;

  const std::span<const uint8_t> active_bytes =
      result.elf_bytes.empty() ? original_bytes : std::span<const uint8_t>(result.elf_bytes);
  AmdGpuCodeObject active(active_bytes.data(), active_bytes.size());
  const rj_code_arch_t arch = arch_for_target(active.target_id());
  if (arch == ROCJITSU_CODE_ARCH_INVALID)
    return;
  CodeObjectPatcher patcher(active);
  const std::span<const uint8_t> old_text = patcher.text_bytes();
  if (old_text.empty()) {
    result.errors.emplace_back("ConSan unmatched barrier abort found no executable text");
    return;
  }
  std::vector<uint8_t> new_text(old_text.begin(), old_text.end());
  const uint32_t abort_word = build_s_endpgm(arch);
  std::vector<ConSanPatchInfo> patches;
  for (const UnmatchedBarrierWait &wait : waits) {
    const ConSanBarrierSite &site = *wait.site;
    if (site.text_offset > new_text.size() || site.size > new_text.size() - site.text_offset) {
      result.errors.emplace_back("ConSan unmatched barrier abort found an out-of-range wait");
      return;
    }
    const bool overlaps_existing =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          if (patch.original_size == 0)
            return false;
          const uint64_t site_end = site.text_offset + site.size;
          const uint64_t patch_end = patch.anchor_offset + patch.original_size;
          return site.text_offset < patch_end && patch.anchor_offset < site_end;
        });
    if (overlaps_existing) {
      result.errors.emplace_back(
          "ConSan unmatched barrier abort overlaps existing instrumentation");
      return;
    }
    std::memcpy(new_text.data() + site.text_offset, &abort_word, sizeof(abort_word));
    ConSanPatchInfo patch;
    patch.kind = ConSanPatchKind::InlineMalformedBarrierAbort;
    patch.anchor_offset = site.text_offset;
    patch.trampoline_offset = site.text_offset;
    patch.original_size = site.size;
    if (wait.owner_descriptor_file_offset)
      patch.owner_descriptor_file_offsets.push_back(*wait.owner_descriptor_file_offset);
    patches.push_back(std::move(patch));
  }
  if (!patcher.replace_text(new_text, kConSanMaxPatchedFileGrowth)) {
    result.errors.emplace_back("ConSan unmatched barrier abort could not rewrite executable text");
    return;
  }
  result.elf_bytes = patcher.emit();
  result.patches.insert(result.patches.end(), std::make_move_iterator(patches.begin()),
                        std::make_move_iterator(patches.end()));
  result.modified = true;
  result.warnings.emplace_back("ConSan diagnosed and aborted " + std::to_string(waits.size()) +
                               " statically unmatched barrier wait(s)");
}

[[nodiscard]] bool has_rejected_kernel(const ConSanResult &result) {
  return std::ranges::any_of(result.kernels, [](const ConSanKernelInfo &kernel) {
    return kernel.preflight_action == ConSanPreflightAction::Reject;
  });
}

[[nodiscard]] bool has_unsupported_preflight(const ConSanResult &result) {
  return std::ranges::any_of(result.kernels, [](const ConSanKernelInfo &kernel) {
    if (kernel.preflight_action != ConSanPreflightAction::Skip)
      return false;
    return !kernel.has_text_range || !kernel.decoded || kernel.stats.decode_error_count != 0 ||
           kernel.stats.ds_other_count != 0;
  });
}

template <typename Entity>
[[nodiscard]] std::vector<std::string> sorted_entity_names(std::span<const Entity> entities) {
  std::vector<std::string> names;
  names.reserve(entities.size());
  for (const Entity &entity : entities)
    names.push_back(entity.name);
  std::ranges::sort(names);
  return names;
}

struct TextSectionIdentity {
  std::string name;
  uint64_t virtual_address = 0;
  uint64_t size = 0;
};

[[nodiscard]] std::vector<TextSectionIdentity>
text_section_identities(const AmdGpuCodeObject &code_object) {
  std::vector<TextSectionIdentity> identities;
  identities.reserve(code_object.text_sections().size());
  for (const Section *section : code_object.text_sections())
    identities.push_back({section->name(), section->vaddr(), section->size()});
  std::ranges::sort(identities, [](const auto &lhs, const auto &rhs) {
    if (lhs.name != rhs.name)
      return lhs.name < rhs.name;
    return lhs.virtual_address < rhs.virtual_address;
  });
  return identities;
}

struct InventoryRange {
  ByteRange bytes;
  ConSanPatchPhase phase = ConSanPatchPhase::Instrumentation;
  ConSanPatchKind kind = ConSanPatchKind::InlineNopRewrite;
  bool must_be_original = false;
  bool is_trampoline = false;
};

[[nodiscard]] std::vector<std::string>
validate_patch_byte_accounting(const AmdGpuCodeObject &original,
                               const AmdGpuCodeObject &replacement, const ConSanResult &result) {
  std::vector<std::string> errors;
  if (original.text_sections().size() != 1 || replacement.text_sections().size() != 1) {
    errors.emplace_back(
        "ConSan final validation cannot map patch inventory across multiple text sections");
    return errors;
  }
  const Section *original_text = original.text_sections().front();
  const Section *replacement_text = replacement.text_sections().front();
  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(original_text->data()), original_text->size());
  const auto replacement_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(replacement_text->data()), replacement_text->size());
  std::vector<InventoryRange> ranges;
  std::vector<bool> accounted(replacement_bytes.size(), false);

  const auto add_range = [&](uint64_t begin, uint32_t size, ConSanPatchPhase phase,
                             ConSanPatchKind kind, bool must_be_original, bool is_trampoline) {
    if (!errors.empty())
      return;
    if (size == 0)
      return;
    if (begin % sizeof(uint32_t) != 0 || size % sizeof(uint32_t) != 0 ||
        begin > replacement_bytes.size() || size > replacement_bytes.size() - begin) {
      errors.emplace_back("ConSan final validation found a stale or unaligned patch range");
      return;
    }
    ranges.push_back({{begin, begin + size}, phase, kind, must_be_original, is_trampoline});
  };

  for (const ConSanPatchInfo &patch : result.patches) {
    add_range(patch.anchor_offset, patch.original_size, patch.phase, patch.kind,
              /*must_be_original=*/true, /*is_trampoline=*/false);
    add_range(patch.trampoline_offset, patch.trampoline_size, patch.phase, patch.kind,
              /*must_be_original=*/false, /*is_trampoline=*/true);
  }
  if (!errors.empty())
    return errors;

  std::ranges::sort(ranges, [](const InventoryRange &lhs, const InventoryRange &rhs) {
    if (lhs.bytes.begin != rhs.bytes.begin)
      return lhs.bytes.begin < rhs.bytes.begin;
    if (lhs.bytes.end != rhs.bytes.end)
      return lhs.bytes.end > rhs.bytes.end;
    return lhs.phase == ConSanPatchPhase::Mutation && rhs.phase != ConSanPatchPhase::Mutation;
  });
  std::vector<const InventoryRange *> active;
  for (const InventoryRange &next : ranges) {
    std::erase_if(active, [&](const InventoryRange *existing) {
      return existing->bytes.end <= next.bytes.begin;
    });
    const bool nested_in_mutation = std::ranges::any_of(active, [&](const auto *existing) {
      return existing->phase == ConSanPatchPhase::Mutation &&
             existing->bytes.begin <= next.bytes.begin && next.bytes.end <= existing->bytes.end;
    });
    const uint64_t size = next.bytes.end - next.bytes.begin;
    if (next.must_be_original &&
        (next.bytes.begin > original_bytes.size() ||
         size > original_bytes.size() - next.bytes.begin) &&
        !nested_in_mutation) {
      errors.emplace_back("ConSan final validation found a stale or unaligned patch range");
      return errors;
    }
    for (const InventoryRange *existing : active) {
      if (!ranges_overlap(existing->bytes, next.bytes))
        continue;
      const bool exact_alias =
          existing->bytes.begin == next.bytes.begin && existing->bytes.end == next.bytes.end;
      const bool nested_cross_phase =
          existing->phase != next.phase &&
          ((existing->bytes.begin <= next.bytes.begin && next.bytes.end <= existing->bytes.end) ||
           (next.bytes.begin <= existing->bytes.begin && existing->bytes.end <= next.bytes.end));
      const bool nested_sc_relay_donor =
          existing->kind == ConSanPatchKind::TrampolineScBranchRelayDonor &&
          next.kind == ConSanPatchKind::TrampolineScBranchRelayDonor &&
          ((existing->is_trampoline &&
            existing->bytes.end - existing->bytes.begin == 3u * sizeof(uint32_t) &&
            !next.is_trampoline && size >= 6u * sizeof(uint32_t) &&
            existing->bytes.begin == next.bytes.begin + sizeof(uint32_t) &&
            existing->bytes.end == next.bytes.begin + 4u * sizeof(uint32_t)) ||
           (next.is_trampoline && size == 3u * sizeof(uint32_t) && !existing->is_trampoline &&
            existing->bytes.end - existing->bytes.begin >= 6u * sizeof(uint32_t) &&
            next.bytes.begin == existing->bytes.begin + sizeof(uint32_t) &&
            next.bytes.end == existing->bytes.begin + 4u * sizeof(uint32_t)));
      const bool nested_composed_access_atomic =
          existing->phase == ConSanPatchPhase::Instrumentation &&
          next.phase == ConSanPatchPhase::Instrumentation &&
          existing->bytes.begin == next.bytes.begin &&
          (((existing->kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
             existing->kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) &&
            next.kind == ConSanPatchKind::TrampolineMoiAtomicRecord) ||
           ((next.kind == ConSanPatchKind::InlineMoiAccessRecordStore ||
             next.kind == ConSanPatchKind::TrampolineMoiAccessRecordStore) &&
            existing->kind == ConSanPatchKind::TrampolineMoiAtomicRecord));
      const bool nested_composed_sampled_access_atomic =
          existing->phase == ConSanPatchPhase::Instrumentation &&
          next.phase == ConSanPatchPhase::Instrumentation &&
          existing->bytes.begin == next.bytes.begin &&
          (((existing->kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
             existing->kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore) &&
            next.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata) ||
           ((next.kind == ConSanPatchKind::InlineMoiSampledWatchpointStore ||
             next.kind == ConSanPatchKind::TrampolineMoiSampledWatchpointStore) &&
            existing->kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata));
      if (!exact_alias && !nested_cross_phase && !nested_sc_relay_donor &&
          !nested_composed_access_atomic && !nested_composed_sampled_access_atomic) {
        errors.emplace_back(
            "ConSan final validation found partially overlapping patch ranges: existing=" +
            std::to_string(existing->bytes.begin) + "-" + std::to_string(existing->bytes.end) +
            " kind=" + std::to_string(static_cast<unsigned>(existing->kind)) +
            " next=" + std::to_string(next.bytes.begin) + "-" + std::to_string(next.bytes.end) +
            " kind=" + std::to_string(static_cast<unsigned>(next.kind)));
        return errors;
      }
    }
    active.push_back(&next);
    std::fill(accounted.begin() + static_cast<ptrdiff_t>(next.bytes.begin),
              accounted.begin() + static_cast<ptrdiff_t>(next.bytes.end), true);
  }

  const size_t common_size = std::min(original_bytes.size(), replacement_bytes.size());
  for (size_t i = 0; i < common_size; ++i) {
    if (original_bytes[i] != replacement_bytes[i] && !accounted[i]) {
      uint64_t previous_end = 0;
      uint64_t next_begin = replacement_bytes.size();
      for (const InventoryRange &range : ranges) {
        if (range.bytes.end <= i)
          previous_end = std::max(previous_end, range.bytes.end);
        if (range.bytes.begin > i)
          next_begin = std::min(next_begin, range.bytes.begin);
      }
      errors.emplace_back(
          "ConSan final validation found an unaccounted executable byte change at " +
          std::to_string(i) + " from " + std::to_string(original_bytes[i]) + " to " +
          std::to_string(replacement_bytes[i]) + " between ranges ending at " +
          std::to_string(previous_end) + " and beginning at " + std::to_string(next_begin));
      return errors;
    }
  }

  const uint32_t padding_nop = build_s_nop(0, arch_for_target(replacement.target_id()));
  size_t offset = original_bytes.size();
  while (offset < replacement_bytes.size()) {
    if (accounted[offset]) {
      ++offset;
      continue;
    }
    if (offset % sizeof(uint32_t) != 0 || sizeof(uint32_t) > replacement_bytes.size() - offset) {
      errors.emplace_back("ConSan final validation found unaccounted appended text bytes");
      return errors;
    }
    uint32_t word = 0;
    std::memcpy(&word, replacement_bytes.data() + offset, sizeof(word));
    if (word != padding_nop) {
      uint64_t previous_end = original_bytes.size();
      uint64_t next_begin = replacement_bytes.size();
      std::optional<InventoryRange> previous_range;
      std::optional<InventoryRange> next_range;
      for (const InventoryRange &range : ranges) {
        if (range.bytes.end <= offset && range.bytes.end >= previous_end) {
          previous_end = std::max(previous_end, range.bytes.end);
          previous_range = range;
        }
        if (range.bytes.begin > offset && range.bytes.begin <= next_begin) {
          next_begin = std::min(next_begin, range.bytes.begin);
          next_range = range;
        }
      }
      std::string samples;
      size_t sample_count = 0;
      for (size_t candidate = original_bytes.size();
           candidate + sizeof(uint32_t) <= replacement_bytes.size() && sample_count < 8u;
           candidate += sizeof(uint32_t)) {
        uint32_t candidate_word = 0;
        std::memcpy(&candidate_word, replacement_bytes.data() + candidate, sizeof(candidate_word));
        if (!accounted[candidate] && candidate_word != padding_nop) {
          samples += (samples.empty() ? "" : ",") + std::to_string(candidate) + ":" +
                     std::to_string(candidate_word);
          ++sample_count;
        }
      }
      errors.emplace_back(
          "ConSan final validation found unaccounted appended executable code at " +
          std::to_string(offset) + " word " + std::to_string(word) + " between ranges ending at " +
          std::to_string(previous_end) +
          (previous_range ? " kind " + std::to_string(static_cast<unsigned>(previous_range->kind))
                          : "") +
          " and beginning at " + std::to_string(next_begin) +
          (next_range ? " kind " + std::to_string(static_cast<unsigned>(next_range->kind)) : "") +
          " samples " + samples);
      return errors;
    }
    offset += sizeof(uint32_t);
  }
  return errors;
}

[[nodiscard]] bool decode_inventory_range(std::span<const uint8_t> text, uint64_t begin,
                                          uint32_t size, Decoder &decoder, std::string &error) {
  if (size == 0)
    return true;
  if (begin > text.size() || size > text.size() - begin) {
    error = "range is outside replacement text";
    return false;
  }
  const std::span<const uint8_t> range = text.subspan(begin, size);
  size_t offset = 0;
  while (offset < range.size()) {
    const size_t remaining = range.size() - offset;
    if (remaining < sizeof(uint32_t)) {
      error = "range ends in a partial instruction word";
      return false;
    }
    std::array<uint32_t, 4> words{};
    std::memcpy(words.data(), range.data() + offset,
                std::min(remaining, words.size() * sizeof(uint32_t)));
    std::unique_ptr<Instruction> instruction;
    try {
      instruction.reset(decoder.decode(words.data(), begin + offset));
    } catch (const std::exception &exception) {
      error = "decoder rejected instruction: " + std::string(exception.what());
      return false;
    }
    if (!instruction) {
      error = "decoder returned null";
      return false;
    }
    const int instruction_size = instruction->size();
    if (instruction_size <= 0 || instruction_size % static_cast<int>(sizeof(uint32_t)) != 0 ||
        static_cast<size_t>(instruction_size) > remaining) {
      error = "decoded instruction does not fit its inventoried range";
      return false;
    }
    offset += static_cast<size_t>(instruction_size);
    // Proof rewrites intentionally terminate before all bytes in the replaced
    // source encoding. Bytes after an unconditional terminal are unreachable
    // and do not need to form another instruction.
    if (instruction->mnemonic() == std::string_view("s_endpgm") ||
        instruction->mnemonic() == std::string_view("s_trap"))
      return true;
  }
  return true;
}

[[nodiscard]] std::vector<std::string> validate_patch_decoding(const AmdGpuCodeObject &replacement,
                                                               const ConSanResult &result) {
  std::vector<std::string> errors;
  if (replacement.text_sections().size() != 1)
    return errors;
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    errors.emplace_back("ConSan final validation could not create a replacement decoder");
    return errors;
  }
  const Section *text_section = replacement.text_sections().front();
  const auto text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(text_section->data()), text_section->size());
  for (const ConSanPatchInfo &patch : result.patches) {
    std::string error;
    if (!decode_inventory_range(text, patch.anchor_offset, patch.original_size, *decoder, error)) {
      errors.emplace_back("ConSan final validation could not re-decode patch anchor: " + error);
      return errors;
    }
    if (!decode_inventory_range(text, patch.trampoline_offset, patch.trampoline_size, *decoder,
                                error)) {
      errors.emplace_back("ConSan final validation could not re-decode generated body at " +
                          std::to_string(patch.trampoline_offset) + " (size " +
                          std::to_string(patch.trampoline_size) + ", anchor " +
                          std::to_string(patch.anchor_offset) + "): " + error);
      return errors;
    }
  }
  return errors;
}

[[nodiscard]] std::vector<std::string>
validate_sc_indirect_island_semantics(const AmdGpuCodeObject &replacement,
                                      const ConSanResult &result) {
  std::vector<std::string> errors;
  if (replacement.text_sections().size() != 1)
    return errors;
  const auto replacement_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(replacement.text_sections().front()->data()),
      replacement.text_sections().front()->size());
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  for (const ConSanPatchInfo &island : result.patches) {
    if (island.kind != ConSanPatchKind::TrampolineScIndirectBranchIsland)
      continue;
    const auto body = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
      return (patch.kind == ConSanPatchKind::LocalCaveLdsLoadCheckTrap ||
              patch.kind == ConSanPatchKind::LocalCaveLdsStoreCheckTrap ||
              patch.kind == ConSanPatchKind::LocalCaveFlatLoadCheckTrap ||
              patch.kind == ConSanPatchKind::LocalCaveFlatStoreCheckTrap) &&
             patch.anchor_offset == island.anchor_offset && patch.indirect_return_offset;
    });
    if (body == result.patches.end() || !body->indirect_pc_sgpr || !body->indirect_saved_scc_sgpr ||
        !body->indirect_saved_vcc_sgpr) {
      errors.emplace_back("ConSan indirect-island proof found incomplete scalar state metadata");
      continue;
    }
    const uint16_t pc = *body->indirect_pc_sgpr;
    const uint16_t scc_save = *body->indirect_saved_scc_sgpr;
    const uint16_t vcc_save = *body->indirect_saved_vcc_sgpr;
    const bool is_flat_body = body->kind == ConSanPatchKind::LocalCaveFlatLoadCheckTrap ||
                              body->kind == ConSanPatchKind::LocalCaveFlatStoreCheckTrap;
    const uint16_t vcc_save_width =
        is_flat_body || arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ? 2u
                                                                                             : 1u;
    const auto ranges_overlap = [](uint16_t lhs_base, uint16_t lhs_width, uint16_t rhs_base,
                                   uint16_t rhs_width) {
      return static_cast<uint32_t>(lhs_base) < static_cast<uint32_t>(rhs_base) + rhs_width &&
             static_cast<uint32_t>(rhs_base) < static_cast<uint32_t>(lhs_base) + lhs_width;
    };
    const bool disjoint = !ranges_overlap(scc_save, 1u, pc, 2u) &&
                          !ranges_overlap(scc_save, 1u, vcc_save, vcc_save_width) &&
                          !ranges_overlap(pc, 2u, vcc_save, vcc_save_width);
    const uint16_t required = std::max<uint16_t>(
        std::max<uint16_t>(static_cast<uint16_t>(pc + 2u), static_cast<uint16_t>(scc_save + 1u)),
        static_cast<uint16_t>(vcc_save + vcc_save_width));
    std::vector<uint32_t> expected_entry;
    std::vector<uint32_t> expected_return;
    const bool encoded =
        append_lds_check_trap_indirect_jump(expected_entry, island.trampoline_offset,
                                            body->trampoline_offset, pc, scc_save, arch) &&
        append_lds_check_trap_indirect_return(expected_return, *body->indirect_return_offset,
                                              body->anchor_offset + body->original_size, pc,
                                              scc_save, arch);
    const auto matches = [&](uint64_t offset, std::span<const uint32_t> words) {
      const uint64_t size = words.size() * sizeof(uint32_t);
      return offset <= replacement_text.size() && size <= replacement_text.size() - offset &&
             std::memcmp(replacement_text.data() + offset, words.data(), size) == 0;
    };
    if (!disjoint || body->indirect_required_sgpr_count < required || !encoded ||
        expected_entry.size() != kLdsCheckTrapIndirectJumpWords ||
        expected_return.size() != kLdsCheckTrapIndirectJumpWords ||
        !matches(island.trampoline_offset, expected_entry) ||
        !matches(*body->indirect_return_offset, expected_return)) {
      errors.emplace_back(
          "ConSan indirect-island proof found corrupted SCC preservation or scalar resources");
    }
  }
  return errors;
}

[[nodiscard]] std::vector<std::string>
validate_sc_branch_only_continuation_semantics(const AmdGpuCodeObject &replacement,
                                               const ConSanResult &result) {
  std::vector<std::string> errors;
  const bool has_branch_only =
      std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.sc_branch_only_continuation;
      });
  if (!has_branch_only)
    return errors;
  if (replacement.text_sections().size() != 1) {
    errors.emplace_back("ConSan branch-only continuation proof requires one executable section");
    return errors;
  }
  const auto text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(replacement.text_sections().front()->data()),
      replacement.text_sections().front()->size());
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  const auto read_word = [&](uint64_t offset) -> std::optional<uint32_t> {
    if (offset > text.size() || sizeof(uint32_t) > text.size() - offset)
      return std::nullopt;
    uint32_t word = 0;
    std::memcpy(&word, text.data() + offset, sizeof(word));
    return word;
  };

  std::unordered_set<uint64_t> owned_relay_slots;
  for (const ConSanPatchInfo &patch : result.patches) {
    if ((patch.kind == ConSanPatchKind::LocalCaveLdsLoadCheckTrap ||
         patch.kind == ConSanPatchKind::LocalCaveLdsStoreCheckTrap) &&
        patch.original_size == 2u * sizeof(uint32_t))
      owned_relay_slots.insert(patch.anchor_offset + sizeof(uint32_t));
    if (patch.kind == ConSanPatchKind::TrampolineScBranchRelayDonor) {
      if (patch.original_size == 2u * sizeof(uint32_t))
        owned_relay_slots.insert(patch.anchor_offset + sizeof(uint32_t));
      if (patch.original_size >= 6u * sizeof(uint32_t)) {
        for (uint64_t offset = patch.anchor_offset + 4u * sizeof(uint32_t);
             offset < patch.anchor_offset + patch.original_size; offset += sizeof(uint32_t))
          owned_relay_slots.insert(offset);
      }
    }
    if (patch.kind == ConSanPatchKind::TrampolineScRelayReservoir) {
      if (patch.original_size == 0) {
        for (uint64_t offset = patch.trampoline_offset;
             offset < patch.trampoline_offset + patch.trampoline_size; offset += sizeof(uint32_t))
          owned_relay_slots.insert(offset);
      } else {
        const auto slots =
            sc_relay_reservoir_all_slot_offsets(patch.anchor_offset, patch.original_size);
        owned_relay_slots.insert(slots.begin(), slots.end());
      }
    }
  }
  std::unordered_set<uint64_t> used_relay_slots;
  const auto validate_hop = [&](uint64_t source, uint64_t target) {
    const auto delta = compute_sopp_branch_simm16(source, target);
    const auto word = read_word(source);
    return delta && word && *word == build_s_branch(*delta, arch);
  };

  for (const ConSanPatchInfo &patch : result.patches) {
    const bool has_route_metadata = !patch.sc_branch_only_entry_relay_offsets.empty() ||
                                    !patch.sc_branch_only_return_relay_offsets.empty();
    if (!patch.sc_branch_only_continuation) {
      if (has_route_metadata) {
        errors.emplace_back(
            "ConSan branch-only continuation proof found route metadata on another patch");
        return errors;
      }
      continue;
    }
    if ((patch.kind != ConSanPatchKind::LocalCaveLdsLoadCheckTrap &&
         patch.kind != ConSanPatchKind::LocalCaveLdsStoreCheckTrap) ||
        patch.original_size != 2u * sizeof(uint32_t) || patch.trampoline_size < sizeof(uint32_t) ||
        patch.trampoline_offset > text.size() ||
        patch.trampoline_size > text.size() - patch.trampoline_offset) {
      errors.emplace_back("ConSan branch-only continuation proof found invalid geometry");
      return errors;
    }

    const auto validate_route = [&](uint64_t source, std::span<const uint64_t> relays,
                                    uint64_t target) {
      uint64_t cursor = source;
      for (uint64_t relay : relays) {
        if (!owned_relay_slots.contains(relay) || !used_relay_slots.insert(relay).second ||
            !validate_hop(cursor, relay))
          return false;
        cursor = relay;
      }
      return validate_hop(cursor, target);
    };
    const uint64_t return_source =
        patch.trampoline_offset + patch.trampoline_size - sizeof(uint32_t);
    if (!validate_route(patch.anchor_offset, patch.sc_branch_only_entry_relay_offsets,
                        patch.trampoline_offset) ||
        !validate_route(return_source, patch.sc_branch_only_return_relay_offsets,
                        patch.anchor_offset + patch.original_size)) {
      errors.emplace_back(
          "ConSan branch-only continuation proof found a stale, shared, or corrupted route");
      return errors;
    }
  }
  return errors;
}

[[nodiscard]] std::vector<std::string>
validate_sc_dense_call_semantics(const AmdGpuCodeObject &original,
                                 const AmdGpuCodeObject &replacement, const ConSanResult &result) {
  std::vector<std::string> errors;
  const bool has_dense = std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::TrampolineScDenseCallDispatcher ||
           patch.kind == ConSanPatchKind::TrampolineScDenseEntryHost;
  });
  if (!has_dense)
    return errors;
  if (original.text_sections().size() != 1 || replacement.text_sections().size() != 1) {
    errors.emplace_back("ConSan dense-call proof requires one executable text section");
    return errors;
  }
  const auto original_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(original.text_sections().front()->data()),
      original.text_sections().front()->size());
  const auto replacement_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(replacement.text_sections().front()->data()),
      replacement.text_sections().front()->size());
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  const uint32_t nop = build_s_nop(0, arch);
  const auto matches = [&](uint64_t offset, std::span<const uint32_t> words) {
    const uint64_t size = words.size() * sizeof(uint32_t);
    return offset <= replacement_text.size() && size <= replacement_text.size() - offset &&
           std::memcmp(replacement_text.data() + offset, words.data(), size) == 0;
  };

  for (const ConSanPatchInfo &dispatcher : result.patches) {
    if (dispatcher.kind != ConSanPatchKind::TrampolineScDenseCallDispatcher)
      continue;
    if (!dispatcher.sc_dense_dispatcher_offset ||
        *dispatcher.sc_dense_dispatcher_offset != dispatcher.trampoline_offset ||
        dispatcher.sc_dense_call_return_sgpr.has_value() ==
            dispatcher.sc_dense_explicit_key_sgpr.has_value() ||
        !dispatcher.indirect_pc_sgpr || !dispatcher.indirect_saved_scc_sgpr) {
      errors.emplace_back("ConSan dense-call proof found incomplete dispatcher metadata");
      continue;
    }
    const auto host = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &patch) {
      return patch.kind == ConSanPatchKind::TrampolineScDenseEntryHost &&
             patch.sc_dense_dispatcher_offset == dispatcher.sc_dense_dispatcher_offset;
    });
    if (host == result.patches.end() || host->original_size != 8u * sizeof(uint32_t) ||
        host->trampoline_size != 15u * sizeof(uint32_t) ||
        host->sc_dense_call_return_sgpr != dispatcher.sc_dense_call_return_sgpr ||
        host->sc_dense_explicit_key_sgpr != dispatcher.sc_dense_explicit_key_sgpr ||
        host->indirect_pc_sgpr != dispatcher.indirect_pc_sgpr ||
        host->indirect_saved_scc_sgpr != dispatcher.indirect_saved_scc_sgpr ||
        host->anchor_offset > original_text.size() ||
        host->original_size > original_text.size() - host->anchor_offset ||
        host->trampoline_offset > replacement_text.size() ||
        host->trampoline_size > replacement_text.size() - host->trampoline_offset) {
      errors.emplace_back("ConSan dense-call proof found an invalid entry-host pairing");
      continue;
    }
    const uint16_t jump_pc = *dispatcher.indirect_pc_sgpr;
    const uint16_t saved_scc = *dispatcher.indirect_saved_scc_sgpr;
    const bool explicit_key = dispatcher.sc_dense_explicit_key_sgpr.has_value();
    const uint16_t dispatch_key = explicit_key ? *dispatcher.sc_dense_explicit_key_sgpr
                                               : *dispatcher.sc_dense_call_return_sgpr;
    const uint16_t dispatch_key_width = explicit_key ? 1u : 2u;
    const bool resources_disjoint = dispatch_key + dispatch_key_width - 1u < jump_pc ||
                                    dispatch_key > static_cast<uint16_t>(jump_pc + 1u);
    if (!resources_disjoint || saved_scc == dispatch_key ||
        (!explicit_key && saved_scc == dispatch_key + 1u) || saved_scc == jump_pc ||
        saved_scc == jump_pc + 1u ||
        dispatcher.indirect_required_sgpr_count < dispatch_key + dispatch_key_width) {
      errors.emplace_back("ConSan dense-call proof found overlapping scalar resources");
      continue;
    }

    std::vector<const ConSanPatchInfo *> bodies;
    for (const ConSanPatchInfo &patch : result.patches) {
      if (patch.sc_dense_dispatcher_offset != dispatcher.sc_dense_dispatcher_offset)
        continue;
      if (patch.kind == ConSanPatchKind::LocalCaveLdsLoadCheckTrap ||
          patch.kind == ConSanPatchKind::LocalCaveLdsStoreCheckTrap)
        bodies.push_back(&patch);
    }
    std::ranges::sort(bodies, {}, &ConSanPatchInfo::anchor_offset);
    if (bodies.empty()) {
      errors.emplace_back("ConSan dense-call proof found an empty dispatcher");
      continue;
    }

    const uint64_t island_offset = host->anchor_offset + sizeof(uint32_t);
    const std::optional<uint32_t> host_entry =
        explicit_key ? std::optional<uint32_t>(
                           build_s_mov_b32(dispatch_key, scalar_positive_inline_u32(0), arch))
                     : instrumentation::build_s_call_i64(dispatch_key, 0, arch);
    std::vector<uint32_t> island_words;
    if (!host_entry ||
        !append_lds_check_trap_indirect_jump(
            island_words, island_offset, dispatcher.trampoline_offset, jump_pc, saved_scc, arch) ||
        island_words.size() != kLdsCheckTrapIndirectJumpWords) {
      errors.emplace_back("ConSan dense-call proof could not reconstruct its entry host");
      continue;
    }
    std::vector<uint32_t> expected_host_anchor = {*host_entry};
    expected_host_anchor.insert(expected_host_anchor.end(), island_words.begin(),
                                island_words.end());
    if (!matches(host->anchor_offset, expected_host_anchor)) {
      errors.emplace_back("ConSan dense-call proof found a corrupted entry host");
      continue;
    }

    std::vector<uint32_t> expected_host_body(host->original_size / sizeof(uint32_t));
    std::memcpy(expected_host_body.data(), original_text.data() + host->anchor_offset,
                host->original_size);
    if (!append_lds_check_trap_indirect_jump(expected_host_body, host->trampoline_offset,
                                             host->anchor_offset + host->original_size, jump_pc,
                                             saved_scc, arch) ||
        !matches(host->trampoline_offset, expected_host_body)) {
      errors.emplace_back("ConSan dense-call proof found a corrupted relocated entry host");
      continue;
    }

    std::vector<uint32_t> dispatcher_words;
    if (!explicit_key) {
      const auto subtract_pc =
          instrumentation::build_s_sub_u32(dispatch_key, dispatch_key, jump_pc, arch);
      if (!subtract_pc) {
        errors.emplace_back("ConSan dense-call proof could not reconstruct its dispatch key");
        continue;
      }
      dispatcher_words.push_back(*subtract_pc);
      dispatcher_words.push_back(build_s_add_u32(dispatch_key, dispatch_key, 255u, arch));
      dispatcher_words.push_back(
          static_cast<uint32_t>(dispatcher.trampoline_offset - island_offset));
    }
    const auto append_target = [&](uint32_t key, uint64_t target) {
      const auto compare = instrumentation::build_s_cmp_eq_u32(
          dispatch_key, explicit_key || key == 0u ? scalar_positive_inline_u32(key) : 255u, arch);
      if (!compare)
        return false;
      dispatcher_words.push_back(*compare);
      if (!explicit_key && key != 0u)
        dispatcher_words.push_back(key);
      std::vector<uint32_t> matched;
      const uint64_t matched_offset =
          dispatcher.trampoline_offset + (dispatcher_words.size() + 1u) * sizeof(uint32_t);
      if (!append_lds_check_trap_indirect_return(matched, matched_offset, target, jump_pc,
                                                 saved_scc, arch))
        return false;
      const auto skip =
          instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(matched.size()), arch);
      if (!skip)
        return false;
      dispatcher_words.push_back(*skip);
      dispatcher_words.insert(dispatcher_words.end(), matched.begin(), matched.end());
      return true;
    };
    bool valid = append_target(0u, host->trampoline_offset);
    for (size_t body_index = 0; body_index < bodies.size(); ++body_index) {
      const ConSanPatchInfo *body = bodies[body_index];
      const auto delta = compute_sopp_branch_simm16(
          body->anchor_offset + (explicit_key ? sizeof(uint32_t) : 0u), island_offset);
      std::optional<std::array<uint32_t, 2>> anchor;
      if (explicit_key && delta && body_index < 64u &&
          body->sc_dense_explicit_key_sgpr == dispatcher.sc_dense_explicit_key_sgpr &&
          body->indirect_saved_vcc_sgpr == dispatcher.sc_dense_explicit_key_sgpr) {
        anchor = std::array<uint32_t, 2>{
            build_s_mov_b32(dispatch_key,
                            scalar_positive_inline_u32(static_cast<uint16_t>(body_index + 1u)),
                            arch),
            build_s_branch(*delta, arch)};
      } else if (!explicit_key && delta &&
                 body->sc_dense_call_return_sgpr == dispatcher.sc_dense_call_return_sgpr) {
        const auto call = instrumentation::build_s_call_i64(dispatch_key, *delta, arch);
        if (call)
          anchor = std::array<uint32_t, 2>{*call, nop};
      }
      std::vector<uint32_t> expected_return;
      valid &= anchor.has_value() && matches(body->anchor_offset, *anchor) &&
               body->indirect_return_offset &&
               append_lds_check_trap_indirect_return(expected_return, *body->indirect_return_offset,
                                                     body->anchor_offset + body->original_size,
                                                     jump_pc, saved_scc, arch) &&
               matches(*body->indirect_return_offset, expected_return) &&
               append_target(explicit_key ? static_cast<uint32_t>(body_index + 1u)
                                          : static_cast<uint32_t>(body->anchor_offset +
                                                                  sizeof(uint32_t) - island_offset),
                             body->trampoline_offset);
    }
    dispatcher_words.push_back(build_s_endpgm(arch));
    valid &= dispatcher.trampoline_size == dispatcher_words.size() * sizeof(uint32_t) &&
             matches(dispatcher.trampoline_offset, dispatcher_words);
    if (!valid)
      errors.emplace_back("ConSan dense-call proof found corrupted call or dispatch geometry");
  }
  return errors;
}

[[nodiscard]] std::vector<std::string>
validate_sc_branch_relay_donor_semantics(const AmdGpuCodeObject &original,
                                         const AmdGpuCodeObject &replacement,
                                         const ConSanResult &result) {
  std::vector<std::string> errors;
  std::vector<const ConSanPatchInfo *> anchors;
  std::vector<const ConSanPatchInfo *> hosts;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineScBranchRelayDonor)
      continue;
    if (patch.phase != ConSanPatchPhase::Instrumentation) {
      errors.emplace_back("ConSan relay donor proof found a non-instrumentation donor");
      return errors;
    }
    if (patch.original_size == 2u * sizeof(uint32_t) &&
        patch.trampoline_size == 3u * sizeof(uint32_t)) {
      anchors.push_back(&patch);
    } else if (patch.original_size >= 6u * sizeof(uint32_t) &&
               patch.original_size % sizeof(uint32_t) == 0 &&
               patch.trampoline_size == patch.original_size + sizeof(uint32_t)) {
      hosts.push_back(&patch);
    } else {
      errors.emplace_back("ConSan relay donor proof found an unknown donor geometry");
      return errors;
    }
  }
  if (anchors.empty() && hosts.empty())
    return errors;
  if (original.text_sections().size() != 1 || replacement.text_sections().size() != 1 ||
      anchors.size() != hosts.size()) {
    errors.emplace_back("ConSan relay donor proof found an incomplete wide-donor pairing");
    return errors;
  }
  const auto original_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(original.text_sections().front()->data()),
      original.text_sections().front()->size());
  const auto replacement_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(replacement.text_sections().front()->data()),
      replacement.text_sections().front()->size());
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    errors.emplace_back("ConSan relay donor proof could not create a pristine decoder");
    return errors;
  }
  const uint32_t nop = build_s_nop(0, arch);
  const uint32_t branch_opcode = build_s_branch(0, arch) & 0xffff0000u;
  const auto read_word = [](std::span<const uint8_t> bytes,
                            uint64_t offset) -> std::optional<uint32_t> {
    if (offset > bytes.size() || sizeof(uint32_t) > bytes.size() - offset)
      return std::nullopt;
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    return word;
  };

  std::unordered_set<uint64_t> relay_vertices;
  std::unordered_set<uint64_t> island_vertices;
  for (const ConSanPatchInfo &patch : result.patches) {
    if ((patch.kind == ConSanPatchKind::LocalCaveLdsLoadCheckTrap ||
         patch.kind == ConSanPatchKind::LocalCaveLdsStoreCheckTrap) &&
        patch.original_size == 2u * sizeof(uint32_t))
      relay_vertices.insert(patch.anchor_offset + sizeof(uint32_t));
    if (patch.kind == ConSanPatchKind::TrampolineScIndirectBranchIsland)
      island_vertices.insert(patch.trampoline_offset);
  }
  for (const ConSanPatchInfo *anchor : anchors)
    relay_vertices.insert(anchor->anchor_offset + sizeof(uint32_t));
  for (const ConSanPatchInfo *host : hosts) {
    for (uint64_t offset = host->anchor_offset + 4u * sizeof(uint32_t);
         offset < host->anchor_offset + host->original_size; offset += sizeof(uint32_t))
      relay_vertices.insert(offset);
  }
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineScRelayReservoir)
      continue;
    if (patch.original_size == 0) {
      for (uint64_t offset = patch.trampoline_offset;
           offset < patch.trampoline_offset + patch.trampoline_size; offset += sizeof(uint32_t))
        relay_vertices.insert(offset);
      continue;
    }
    for (uint64_t offset :
         sc_relay_reservoir_slot_offsets(patch.anchor_offset, patch.original_size))
      relay_vertices.insert(offset);
  }
  const auto valid_relay_payload = [&](uint64_t offset) {
    const auto word = read_word(replacement_text, offset);
    if (!word)
      return false;
    if (*word == nop)
      return true;
    if ((*word & 0xffff0000u) != branch_opcode)
      return false;
    const int64_t target =
        static_cast<int64_t>(offset) + sizeof(uint32_t) +
        static_cast<int64_t>(static_cast<int16_t>(*word & 0xffffu)) * sizeof(uint32_t);
    if (target <= static_cast<int64_t>(offset) || target < 0)
      return false;
    const uint64_t target_offset = static_cast<uint64_t>(target);
    return relay_vertices.contains(target_offset) || island_vertices.contains(target_offset);
  };

  std::unordered_set<const ConSanPatchInfo *> paired_hosts;
  for (const ConSanPatchInfo *anchor : anchors) {
    const auto host_it = std::ranges::find_if(hosts, [&](const ConSanPatchInfo *host) {
      return anchor->trampoline_offset == host->anchor_offset + sizeof(uint32_t);
    });
    if (host_it == hosts.end() || !paired_hosts.insert(*host_it).second) {
      errors.emplace_back("ConSan relay donor proof found an ambiguous wide-donor pairing");
      return errors;
    }
    const ConSanPatchInfo &host = **host_it;
    if (anchor->anchor_offset > original_text.size() ||
        anchor->original_size > original_text.size() - anchor->anchor_offset ||
        host.anchor_offset > original_text.size() ||
        host.original_size > original_text.size() - host.anchor_offset ||
        host.trampoline_offset > replacement_text.size() ||
        host.trampoline_size > replacement_text.size() - host.trampoline_offset) {
      errors.emplace_back("ConSan relay donor proof found a stale donor range");
      return errors;
    }

    std::array<uint32_t, 4> anchor_words{};
    std::unique_ptr<Instruction> anchor_instruction;
    try {
      std::memcpy(anchor_words.data(), original_text.data() + anchor->anchor_offset,
                  anchor->original_size);
      anchor_instruction.reset(decoder->decode(anchor_words.data(), anchor->anchor_offset));
    } catch (...) {
      anchor_instruction.reset();
    }
    if (!anchor_instruction ||
        anchor_instruction->size() != static_cast<int>(anchor->original_size) ||
        std::string_view(anchor_instruction->mnemonic()).starts_with("ds_") ||
        !is_relocatable_anchor(*anchor_instruction, anchor->anchor_offset, original_text, arch)) {
      errors.emplace_back("ConSan relay donor proof rejected its pristine anchor semantics");
      return errors;
    }
    uint32_t host_cursor = 0;
    while (host_cursor < host.original_size) {
      std::array<uint32_t, 4> words{};
      std::unique_ptr<Instruction> instruction;
      try {
        const uint32_t remaining = host.original_size - host_cursor;
        std::memcpy(words.data(), original_text.data() + host.anchor_offset + host_cursor,
                    std::min<uint32_t>(remaining, words.size() * sizeof(uint32_t)));
        instruction.reset(decoder->decode(words.data(), host.anchor_offset + host_cursor));
      } catch (...) {
        instruction.reset();
      }
      if (!instruction || instruction->size() <= 0 ||
          static_cast<uint32_t>(instruction->size()) > host.original_size - host_cursor) {
        errors.emplace_back("ConSan relay donor proof could not decode its pristine host");
        return errors;
      }
      const std::string_view mnemonic = instruction->mnemonic();
      const bool valid =
          host_cursor == 0
              ? instruction->size() == 3 * static_cast<int>(sizeof(uint32_t)) &&
                    (mnemonic.starts_with("flat_load") || mnemonic.starts_with("flat_store"))
              : is_sc_relay_reservoir_instruction(*instruction, host.anchor_offset + host_cursor,
                                                  original_text, arch);
      if (!valid) {
        errors.emplace_back("ConSan relay donor proof rejected its pristine host semantics");
        return errors;
      }
      host_cursor += static_cast<uint32_t>(instruction->size());
    }

    std::vector<uint32_t> pristine_host(host.original_size / sizeof(uint32_t));
    std::vector<uint32_t> rewritten_host(host.original_size / sizeof(uint32_t));
    std::vector<uint32_t> relocated_host(host.trampoline_size / sizeof(uint32_t));
    std::array<uint32_t, 2> pristine_anchor{};
    std::array<uint32_t, 2> rewritten_anchor{};
    std::memcpy(pristine_host.data(), original_text.data() + host.anchor_offset,
                pristine_host.size() * sizeof(uint32_t));
    std::memcpy(rewritten_host.data(), replacement_text.data() + host.anchor_offset,
                rewritten_host.size() * sizeof(uint32_t));
    std::memcpy(relocated_host.data(), replacement_text.data() + host.trampoline_offset,
                relocated_host.size() * sizeof(uint32_t));
    std::memcpy(pristine_anchor.data(), original_text.data() + anchor->anchor_offset,
                sizeof(pristine_anchor));
    std::memcpy(rewritten_anchor.data(), replacement_text.data() + anchor->anchor_offset,
                sizeof(rewritten_anchor));
    if (!std::ranges::equal(
            pristine_host, std::span<const uint32_t>(relocated_host).first(pristine_host.size()))) {
      errors.emplace_back("ConSan relay donor proof found a corrupted displaced host sequence");
      return errors;
    }
    if (!std::ranges::equal(pristine_anchor,
                            std::span<const uint32_t>(rewritten_host).subspan(1u, 2u))) {
      errors.emplace_back(
          "ConSan relay donor proof found a corrupted displaced anchor sequence at " +
          std::to_string(anchor->anchor_offset) + " via host " +
          std::to_string(host.anchor_offset) + "; pristine=" + std::to_string(pristine_anchor[0]) +
          "," + std::to_string(pristine_anchor[1]) + " relocated=" +
          std::to_string(rewritten_host[1]) + "," + std::to_string(rewritten_host[2]));
      return errors;
    }
    const auto anchor_to_host =
        compute_sopp_branch_simm16(anchor->anchor_offset, anchor->trampoline_offset);
    const auto host_to_body =
        compute_sopp_branch_simm16(host.anchor_offset, host.trampoline_offset);
    const auto anchor_return = compute_sopp_branch_simm16(
        host.anchor_offset + 3u * sizeof(uint32_t), anchor->anchor_offset + anchor->original_size);
    const auto host_return = compute_sopp_branch_simm16(host.trampoline_offset + host.original_size,
                                                        host.anchor_offset + host.original_size);
    if (!anchor_to_host || !host_to_body || !anchor_return || !host_return ||
        rewritten_anchor[0] != build_s_branch(*anchor_to_host, arch) ||
        rewritten_host[0] != build_s_branch(*host_to_body, arch) ||
        rewritten_host[3] != build_s_branch(*anchor_return, arch) ||
        relocated_host.back() != build_s_branch(*host_return, arch)) {
      errors.emplace_back("ConSan relay donor proof found a corrupted normal execution path");
      return errors;
    }
    std::vector<uint64_t> payload_offsets = {anchor->anchor_offset + sizeof(uint32_t)};
    for (uint64_t offset = host.anchor_offset + 4u * sizeof(uint32_t);
         offset < host.anchor_offset + host.original_size; offset += sizeof(uint32_t))
      payload_offsets.push_back(offset);
    bool has_used_payload = false;
    for (uint64_t payload_offset : payload_offsets) {
      const auto payload_word = read_word(replacement_text, payload_offset);
      if (!payload_word || !valid_relay_payload(payload_offset)) {
        errors.emplace_back("ConSan relay donor proof found an invalid relay payload");
        return errors;
      }
      has_used_payload |= *payload_word != nop;
    }
    if (!has_used_payload) {
      errors.emplace_back("ConSan relay donor proof found an unused wide donor");
      return errors;
    }
  }
  return errors;
}

[[nodiscard]] std::vector<std::string>
validate_sc_relay_reservoir_semantics(const AmdGpuCodeObject &original,
                                      const AmdGpuCodeObject &replacement,
                                      const ConSanResult &result) {
  std::vector<std::string> errors;
  std::vector<const ConSanPatchInfo *> reservoirs;
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind == ConSanPatchKind::TrampolineScRelayReservoir)
      reservoirs.push_back(&patch);
  }
  if (reservoirs.empty())
    return errors;
  if (original.text_sections().size() != 1 || replacement.text_sections().size() != 1) {
    errors.emplace_back("ConSan relay reservoir proof requires one executable text section");
    return errors;
  }
  const auto original_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(original.text_sections().front()->data()),
      original.text_sections().front()->size());
  const auto replacement_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(replacement.text_sections().front()->data()),
      replacement.text_sections().front()->size());
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    errors.emplace_back("ConSan relay reservoir proof could not create a pristine decoder");
    return errors;
  }

  std::vector<uint64_t> extra_leaders = kernel_entry_offsets(result);
  for (const ConSanFunctionInfo &function : result.functions)
    extra_leaders.push_back(function.entry_text_offset);
  std::ranges::sort(extra_leaders);
  extra_leaders.erase(std::ranges::unique(extra_leaders).begin(), extra_leaders.end());
  std::vector<BasicBlock::CodeRange> code_ranges;
  for (const AmdGpuFunctionInfo &function : original.functions()) {
    if (function.code_size != 0)
      code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
  }
  auto blocks = BasicBlock::build(original, *decoder, arch, extra_leaders, code_ranges);
  std::vector<BasicBlock *> blocks_by_offset;
  blocks_by_offset.reserve(blocks.size());
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    if (!block->instructions().empty())
      blocks_by_offset.push_back(block.get());
  }
  std::ranges::sort(blocks_by_offset, [](BasicBlock *lhs, BasicBlock *rhs) {
    return (*lhs->instructions().begin()).src_loc() < (*rhs->instructions().begin()).src_loc();
  });
  const uint32_t nop = build_s_nop(0, arch);
  const uint32_t branch_opcode = build_s_branch(0, arch) & 0xffff0000u;
  const auto read_word = [](std::span<const uint8_t> bytes,
                            uint64_t offset) -> std::optional<uint32_t> {
    if (offset > bytes.size() || sizeof(uint32_t) > bytes.size() - offset)
      return std::nullopt;
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + offset, sizeof(word));
    return word;
  };

  std::unordered_set<uint64_t> relay_vertices;
  std::unordered_set<uint64_t> island_vertices;
  std::unordered_map<uint64_t, uint64_t> branch_only_relay_targets;
  for (const ConSanPatchInfo &patch : result.patches) {
    for (uint64_t offset : patch.branch_relay_offsets) {
      if (offset < patch.trampoline_offset ||
          offset - patch.trampoline_offset >= patch.trampoline_size ||
          offset % sizeof(uint32_t) != 0) {
        errors.emplace_back("ConSan relay reservoir proof found stale branch-relay provenance");
        return errors;
      }
      relay_vertices.insert(offset);
    }
    if (patch.kind == ConSanPatchKind::TrampolineScIndirectBranchIsland)
      island_vertices.insert(patch.trampoline_offset);
    if ((patch.kind == ConSanPatchKind::LocalCaveLdsLoadCheckTrap ||
         patch.kind == ConSanPatchKind::LocalCaveLdsStoreCheckTrap) &&
        patch.original_size == 2u * sizeof(uint32_t))
      relay_vertices.insert(patch.anchor_offset + sizeof(uint32_t));
    if (patch.kind == ConSanPatchKind::TrampolineScBranchRelayDonor) {
      if (patch.original_size == 2u * sizeof(uint32_t))
        relay_vertices.insert(patch.anchor_offset + sizeof(uint32_t));
      if (patch.original_size >= 6u * sizeof(uint32_t))
        for (uint64_t offset = patch.anchor_offset + 4u * sizeof(uint32_t);
             offset < patch.anchor_offset + patch.original_size; offset += sizeof(uint32_t))
          relay_vertices.insert(offset);
    }
    if (patch.kind == ConSanPatchKind::TrampolineScRelayReservoir) {
      if (patch.original_size == 0) {
        for (uint64_t offset = patch.trampoline_offset;
             offset < patch.trampoline_offset + patch.trampoline_size; offset += sizeof(uint32_t))
          relay_vertices.insert(offset);
      } else {
        for (uint64_t offset :
             sc_relay_reservoir_all_slot_offsets(patch.anchor_offset, patch.original_size))
          relay_vertices.insert(offset);
      }
    }
    if (patch.sc_branch_only_continuation) {
      for (size_t index = 0; index < patch.sc_branch_only_entry_relay_offsets.size(); ++index) {
        const uint64_t source = patch.sc_branch_only_entry_relay_offsets[index];
        const uint64_t target = index + 1u < patch.sc_branch_only_entry_relay_offsets.size()
                                    ? patch.sc_branch_only_entry_relay_offsets[index + 1u]
                                    : patch.trampoline_offset;
        branch_only_relay_targets.emplace(source, target);
      }
      for (size_t index = 0; index < patch.sc_branch_only_return_relay_offsets.size(); ++index) {
        const uint64_t source = patch.sc_branch_only_return_relay_offsets[index];
        const uint64_t target = index + 1u < patch.sc_branch_only_return_relay_offsets.size()
                                    ? patch.sc_branch_only_return_relay_offsets[index + 1u]
                                    : patch.anchor_offset + patch.original_size;
        branch_only_relay_targets.emplace(source, target);
      }
    }
  }

  const auto valid_relay_target = [&](uint64_t source, int64_t target) {
    if (target < 0)
      return false;
    if (const auto branch_only = branch_only_relay_targets.find(source);
        branch_only != branch_only_relay_targets.end())
      return static_cast<uint64_t>(target) == branch_only->second;
    return target > static_cast<int64_t>(source) &&
           (relay_vertices.contains(static_cast<uint64_t>(target)) ||
            island_vertices.contains(static_cast<uint64_t>(target)));
  };

  for (const ConSanPatchInfo *patch : reservoirs) {
    if (patch->original_size == 0) {
      if (patch->phase != ConSanPatchPhase::Instrumentation || patch->trampoline_size == 0 ||
          patch->trampoline_size % sizeof(uint32_t) != 0 ||
          patch->trampoline_offset > replacement_text.size() ||
          patch->trampoline_size > replacement_text.size() - patch->trampoline_offset) {
        errors.emplace_back("ConSan relay reservoir proof found invalid generated-bank geometry");
        return errors;
      }
      bool has_used_payload = false;
      for (uint64_t offset = patch->trampoline_offset;
           offset < patch->trampoline_offset + patch->trampoline_size; offset += sizeof(uint32_t)) {
        const auto word = read_word(replacement_text, offset);
        if (!word) {
          errors.emplace_back("ConSan relay reservoir proof found a missing payload word");
          return errors;
        }
        if (*word == nop)
          continue;
        if ((*word & 0xffff0000u) != branch_opcode) {
          errors.emplace_back("ConSan relay reservoir proof found an invalid relay payload");
          return errors;
        }
        const int64_t target =
            static_cast<int64_t>(offset) + sizeof(uint32_t) +
            static_cast<int64_t>(static_cast<int16_t>(*word & 0xffffu)) * sizeof(uint32_t);
        if (!valid_relay_target(offset, target)) {
          errors.emplace_back("ConSan relay reservoir proof found an invalid relay target from " +
                              std::to_string(offset) + " to " + std::to_string(target));
          return errors;
        }
        has_used_payload = true;
      }
      if (!has_used_payload) {
        errors.emplace_back(
            "ConSan relay reservoir proof found an unused reservoir (generated) at " +
            std::to_string(patch->trampoline_offset) + " with " +
            std::to_string(patch->trampoline_size) + " bytes");
        return errors;
      }
      continue;
    }
    if (patch->phase != ConSanPatchPhase::Instrumentation ||
        patch->original_size <
            (kScRelayEntryWords + kScRelayTailRestoreWords + 1u) * sizeof(uint32_t) ||
        patch->original_size > 512u * sizeof(uint32_t) ||
        patch->original_size % sizeof(uint32_t) != 0 ||
        patch->trampoline_size !=
            patch->original_size + kScRelayAppendedOverheadWords * sizeof(uint32_t) ||
        !patch->indirect_pc_sgpr || !patch->indirect_saved_vcc_sgpr ||
        !patch->indirect_return_pc_sgpr || !patch->indirect_return_saved_vcc_sgpr) {
      errors.emplace_back("ConSan relay reservoir proof found invalid geometry or phase");
      return errors;
    }
    if (patch->anchor_offset > original_text.size() ||
        patch->original_size > original_text.size() - patch->anchor_offset ||
        patch->anchor_offset > replacement_text.size() ||
        patch->original_size > replacement_text.size() - patch->anchor_offset ||
        patch->trampoline_offset > replacement_text.size() ||
        patch->trampoline_size > replacement_text.size() - patch->trampoline_offset) {
      errors.emplace_back("ConSan relay reservoir proof found a stale range");
      return errors;
    }

    bool found_pristine_block = false;
    const auto owning_block =
        std::ranges::upper_bound(blocks_by_offset, patch->anchor_offset, {}, [](BasicBlock *block) {
          return (*block->instructions().begin()).src_loc();
        });
    if (owning_block != blocks_by_offset.begin()) {
      BasicBlock *block = *std::prev(owning_block);
      uint64_t cursor = patch->anchor_offset;
      uint32_t clause_remaining = 0;
      bool inside = false;
      bool valid = true;
      for (const Instruction &instruction : block->instructions()) {
        if (instruction.src_loc() < patch->anchor_offset) {
          if (clause_remaining > 0)
            --clause_remaining;
          if (is_s_clause(instruction))
            clause_remaining = s_clause_following_instruction_count(instruction);
          continue;
        }
        if (instruction.src_loc() >= patch->anchor_offset + patch->original_size)
          break;
        inside = true;
        if (instruction.src_loc() != cursor || clause_remaining != 0 ||
            !is_sc_relay_reservoir_instruction(instruction, instruction.src_loc(), original_text,
                                               arch)) {
          valid = false;
          break;
        }
        cursor += static_cast<uint64_t>(instruction.size());
        if (is_s_clause(instruction))
          clause_remaining = s_clause_following_instruction_count(instruction);
        else if (clause_remaining > 0)
          --clause_remaining;
      }
      if (inside && valid && cursor == patch->anchor_offset + patch->original_size) {
        found_pristine_block = true;
      }
    }
    if (!found_pristine_block) {
      errors.emplace_back(
          "ConSan relay reservoir proof rejected its pristine straight-line sequence");
      return errors;
    }
    if (!std::ranges::equal(original_text.subspan(patch->anchor_offset, patch->original_size),
                            replacement_text.subspan(patch->trampoline_offset + sizeof(uint32_t),
                                                     patch->original_size))) {
      errors.emplace_back("ConSan relay reservoir proof found a corrupted displaced sequence");
      return errors;
    }
    std::vector<uint32_t> expected_entry;
    std::vector<uint32_t> expected_return;
    expected_entry.push_back(build_s_mov_b32(*patch->indirect_saved_vcc_sgpr, kRdna4VccLo, arch));
    expected_return.push_back(
        build_s_mov_b32(*patch->indirect_return_saved_vcc_sgpr, kRdna4VccLo, arch));
    if (!append_lds_check_trap_indirect_jump(expected_entry, patch->anchor_offset,
                                             patch->trampoline_offset, *patch->indirect_pc_sgpr,
                                             kRdna4ExecHi, arch) ||
        !append_lds_check_trap_indirect_jump(
            expected_return, patch->trampoline_offset + sizeof(uint32_t) + patch->original_size,
            patch->anchor_offset + patch->original_size -
                kScRelayTailRestoreWords * sizeof(uint32_t),
            *patch->indirect_return_pc_sgpr, kRdna4ExecHi, arch) ||
        std::memcmp(expected_entry.data(), replacement_text.data() + patch->anchor_offset,
                    expected_entry.size() * sizeof(uint32_t)) != 0 ||
        read_word(replacement_text, patch->trampoline_offset) !=
            build_s_mov_b32(kRdna4VccLo, *patch->indirect_saved_vcc_sgpr, arch) ||
        std::memcmp(expected_return.data(),
                    replacement_text.data() + patch->trampoline_offset + sizeof(uint32_t) +
                        patch->original_size,
                    expected_return.size() * sizeof(uint32_t)) != 0 ||
        read_word(replacement_text,
                  patch->anchor_offset + patch->original_size - sizeof(uint32_t)) !=
            build_s_mov_b32(kRdna4VccLo, *patch->indirect_return_saved_vcc_sgpr, arch)) {
      errors.emplace_back("ConSan relay reservoir proof found a corrupted normal execution path");
      return errors;
    }
    bool has_used_payload = false;
    for (uint64_t offset = patch->anchor_offset + kScRelayEntryWords * sizeof(uint32_t);
         offset <
         patch->anchor_offset + patch->original_size - kScRelayTailRestoreWords * sizeof(uint32_t);
         offset += sizeof(uint32_t)) {
      const auto word = read_word(replacement_text, offset);
      if (!word) {
        errors.emplace_back("ConSan relay reservoir proof found a missing payload word");
        return errors;
      }
      if (*word == nop)
        continue;
      if ((*word & 0xffff0000u) != branch_opcode) {
        errors.emplace_back("ConSan relay reservoir proof found an invalid relay payload");
        return errors;
      }
      const int64_t target =
          static_cast<int64_t>(offset) + sizeof(uint32_t) +
          static_cast<int64_t>(static_cast<int16_t>(*word & 0xffffu)) * sizeof(uint32_t);
      if (!valid_relay_target(offset, target)) {
        errors.emplace_back("ConSan relay reservoir proof found an invalid relay target from " +
                            std::to_string(offset) + " to " + std::to_string(target));
        return errors;
      }
      has_used_payload = true;
    }
    if (!has_used_payload) {
      errors.emplace_back("ConSan relay reservoir proof found an unused reservoir (displaced) at " +
                          std::to_string(patch->anchor_offset) + " with " +
                          std::to_string(patch->original_size) + " bytes");
      return errors;
    }
  }
  return errors;
}

[[nodiscard]] bool decodes_as_barrier(std::span<const uint8_t> text, uint64_t offset,
                                      Decoder &decoder) {
  if (offset > text.size() || sizeof(uint32_t) > text.size() - offset)
    return false;
  std::array<uint32_t, 4> words{};
  std::memcpy(words.data(), text.data() + offset,
              std::min(words.size() * sizeof(uint32_t), text.size() - offset));
  std::unique_ptr<Instruction> instruction;
  try {
    instruction.reset(decoder.decode(words.data(), offset));
  } catch (...) {
    return false;
  }
  return instruction != nullptr && instruction->size() == static_cast<int>(sizeof(uint32_t)) &&
         is_barrier_instruction(*instruction);
}

[[nodiscard]] std::vector<std::string>
validate_mutation_semantics(const AmdGpuCodeObject &original, const AmdGpuCodeObject &replacement,
                            const ConSanResult &result) {
  std::vector<std::string> errors;
  // Instrumentation-only transforms have no mutation semantics to rederive.
  // Avoid rebuilding the entire pristine synchronization/CFG inventory here:
  // final validation calls this function once internally and callers may
  // independently validate the same result again. On large code objects that
  // otherwise turned one transform into three identical whole-image analyses.
  if (std::ranges::none_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.phase == ConSanPatchPhase::Mutation;
      })) {
    return errors;
  }
  if (original.text_sections().size() != 1 || replacement.text_sections().size() != 1)
    return errors;
  const auto original_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(original.text_sections().front()->data()),
      original.text_sections().front()->size());
  const auto replacement_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(replacement.text_sections().front()->data()),
      replacement.text_sections().front()->size());
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    errors.emplace_back("ConSan mutation proof could not create a replacement decoder");
    return errors;
  }
  const uint32_t nop = build_s_nop(0, arch);
  ConSanOptions pristine_options;
  pristine_options.flavor = ConSanFlavor::SuperCollider;
  pristine_options.fault_drop_barrier = true;
  // This pass only re-derives pristine barrier identities. Applying the fault
  // here would enter staged mutation and recursively final-validate another
  // mutation while already inside final validation.
  pristine_options.fault_dry_run = true;
  const auto original_image = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(original.image_data()), original.image_size());
  const ConSanResult pristine_inventory = try_patch_consan_impl(original_image, pristine_options);
  std::vector<const ConSanPatchInfo *> markerless_sources;
  const ConSanPatchInfo *markerless_target = nullptr;
  std::vector<ConSanBarrierSite> barrier_id_scope_originals;
  std::vector<ConSanBarrierSite> barrier_id_scope_rewrites;
  std::vector<const ConSanPatchInfo *> barrier_id_scope_patches;
  std::vector<const ConSanPatchInfo *> exact_barrier_drop_patches;
  bool barrier_id_scope_reinstrumented = false;
  const auto composed_atomic_bytes =
      [&](const ConSanPatchInfo &mutation) -> std::optional<std::span<const uint8_t>> {
    if (!result.staged_composition_validated)
      return std::nullopt;
    const ConSanPatchInfo *match = nullptr;
    for (const ConSanPatchInfo &candidate : result.patches) {
      if (candidate.phase != ConSanPatchPhase::Instrumentation ||
          candidate.anchor_offset != mutation.anchor_offset ||
          candidate.original_size != mutation.original_size)
        continue;
      const bool sc_relocation = candidate.kind == ConSanPatchKind::TrampolineScPerturbation &&
                                 candidate.perturbation_composite_atomic_overlap &&
                                 candidate.perturbation_edge;
      const bool moi_atomic_relocation =
          (candidate.kind == ConSanPatchKind::TrampolineMoiAtomicRecord ||
           candidate.kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata) &&
          candidate.relocated_guest_instruction_offset;
      const bool inline_shadow_relocation =
          candidate.kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering &&
          candidate.relocated_guest_instruction_offset;
      if (!sc_relocation && !moi_atomic_relocation && !inline_shadow_relocation)
        continue;
      if (match != nullptr)
        return std::nullopt;
      match = &candidate;
    }
    if (match == nullptr)
      return std::nullopt;
    uint64_t body_offset = 0;
    if (match->kind == ConSanPatchKind::TrampolineMoiAtomicRecord ||
        match->kind == ConSanPatchKind::TrampolineMoiSampledSyncMetadata ||
        match->kind == ConSanPatchKind::TrampolineMoiInlineAtomicOrdering) {
      body_offset = *match->relocated_guest_instruction_offset;
      if (match->trampoline_offset > replacement_text.size() ||
          match->trampoline_size > replacement_text.size() - match->trampoline_offset)
        return std::nullopt;
      const uint64_t trampoline_end = match->trampoline_offset + match->trampoline_size;
      if (body_offset < match->trampoline_offset || body_offset > trampoline_end ||
          mutation.original_size > trampoline_end - body_offset)
        return std::nullopt;
    } else {
      body_offset =
          match->trampoline_offset +
          (*match->perturbation_edge == ConSanPerturbationEdge::Release ? sizeof(uint32_t) : 0u);
    }
    if (body_offset > replacement_text.size() ||
        mutation.original_size > replacement_text.size() - body_offset)
      return std::nullopt;
    return replacement_text.subspan(body_offset, mutation.original_size);
  };
  const auto composed_inline_barrier = [&](const ConSanPatchInfo &mutation) {
    const ConSanPatchInfo *match = nullptr;
    for (const ConSanPatchInfo &candidate : result.patches) {
      if (candidate.phase != ConSanPatchPhase::Instrumentation ||
          candidate.kind != ConSanPatchKind::TrampolineMoiInlineEpochBarrier ||
          candidate.anchor_offset != mutation.anchor_offset ||
          candidate.original_size != mutation.original_size ||
          !candidate.relocated_guest_instruction_offset)
        continue;
      if (match != nullptr)
        return false;
      match = &candidate;
    }
    if (match == nullptr || match->trampoline_offset > replacement_text.size() ||
        match->trampoline_size > replacement_text.size() - match->trampoline_offset)
      return false;
    const uint64_t trampoline_end = match->trampoline_offset + match->trampoline_size;
    const uint64_t relocated = *match->relocated_guest_instruction_offset;
    return relocated >= match->trampoline_offset && relocated < trampoline_end &&
           decodes_as_barrier(replacement_text, relocated, *decoder);
  };

  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.phase != ConSanPatchPhase::Mutation)
      continue;
    if (patch.anchor_offset > original_text.size() ||
        patch.original_size > original_text.size() - patch.anchor_offset ||
        patch.anchor_offset > replacement_text.size() ||
        patch.original_size > replacement_text.size() - patch.anchor_offset) {
      errors.emplace_back("ConSan mutation proof found an out-of-range mutation anchor");
      continue;
    }

    if (patch.kind == ConSanPatchKind::InlineBarrierNopRewrite ||
        patch.kind == ConSanPatchKind::InlineBarrierMoveSourceRewrite) {
      if (patch.kind == ConSanPatchKind::InlineBarrierNopRewrite &&
          !patch.fault_sequence_identity.empty())
        exact_barrier_drop_patches.push_back(&patch);
      if (patch.kind == ConSanPatchKind::InlineBarrierMoveSourceRewrite &&
          patch.barrier_move_direction)
        markerless_sources.push_back(&patch);
      uint32_t replacement_word = 0;
      std::memcpy(&replacement_word, replacement_text.data() + patch.anchor_offset,
                  sizeof(replacement_word));
      if (!decodes_as_barrier(original_text, patch.anchor_offset, *decoder) ||
          replacement_word != nop) {
        errors.emplace_back(
            "ConSan mutation proof did not replace the selected barrier with s_nop 0 at text "
            "offset " +
            std::to_string(patch.anchor_offset) +
            " (replacement=" + std::to_string(replacement_word) + ")");
      }
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineBarrierMoveTargetRewrite) {
      if (patch.trampoline_size != 0) {
        if (markerless_target != nullptr) {
          errors.emplace_back(
              "ConSan mutation proof found multiple markerless barrier relocation targets");
        } else {
          markerless_target = &patch;
        }
        continue;
      }
      if (!decodes_as_barrier(replacement_text, patch.anchor_offset, *decoder) &&
          !composed_inline_barrier(patch)) {
        errors.emplace_back(
            "ConSan mutation proof did not place a barrier at the relocation target");
      }
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineOrdinaryAddressRewrite ||
        patch.kind == ConSanPatchKind::InlineOrdinaryOrderRewrite ||
        patch.kind == ConSanPatchKind::InlineOrdinaryScopeRewrite) {
      ConSanOptions selection;
      selection.fault_site_identity = patch.fault_primary_identity;
      const auto target = select_ordinary_acquire_mutation_target(pristine_inventory, selection);
      const bool identities_match =
          target && !patch.fault_primary_identity.empty() &&
          patch.fault_sequence_identity == target->sequence->identity &&
          result.applied_fault_logical_identity == patch.fault_sequence_identity &&
          patch.owner_descriptor_file_offsets.size() == target->site->execution_owners.size() &&
          std::ranges::equal(patch.owner_descriptor_file_offsets, target->site->execution_owners,
                             {}, std::identity{}, &ConSanExecutionOwner::descriptor_file_offset);
      if (!identities_match) {
        errors.emplace_back(
            "ConSan mutation proof could not rederive the exact ordinary acquire identity and "
            "owners");
        continue;
      }
      if (patch.kind == ConSanPatchKind::InlineOrdinaryAddressRewrite) {
        if (!patch.fault_companion_identity.empty() ||
            patch.anchor_offset != target->site->text_offset ||
            patch.original_size != sizeof(rdna4::VglobalMachineInst)) {
          errors.emplace_back(
              "ConSan mutation proof found stale ordinary acquire address metadata");
          continue;
        }
        std::array<uint32_t, 3> before{};
        std::array<uint32_t, 3> after{};
        std::memcpy(before.data(), original_text.data() + patch.anchor_offset, patch.original_size);
        std::memcpy(after.data(), replacement_text.data() + patch.anchor_offset,
                    patch.original_size);
        const int32_t before_offset = sign_extend_24(before[2] >> 8u);
        const int32_t after_offset = sign_extend_24(after[2] >> 8u);
        const int64_t delta = static_cast<int64_t>(after_offset) - before_offset;
        if (before[0] != after[0] || before[1] != after[1] ||
            (before[2] & 0xffu) != (after[2] & 0xffu) || delta <= 0 || delta > 0x7fffff ||
            delta % static_cast<int64_t>(sizeof(uint32_t)) != 0) {
          errors.emplace_back(
              "ConSan mutation proof changed fields other than the ordinary load ioffset");
        }
        continue;
      }
      if (patch.kind == ConSanPatchKind::InlineOrdinaryOrderRewrite) {
        if (patch.fault_companion_identity != target->cache->identity ||
            patch.anchor_offset != target->cache->text_offset ||
            patch.original_size != target->cache->size) {
          errors.emplace_back(
              "ConSan mutation proof did not target the exact ordinary acquire cache boundary");
          continue;
        }
        bool all_nops = true;
        for (uint32_t offset = 0; offset < patch.original_size; offset += sizeof(uint32_t)) {
          uint32_t word = 0;
          std::memcpy(&word, replacement_text.data() + patch.anchor_offset + offset, sizeof(word));
          all_nops &= word == nop;
        }
        if (!all_nops) {
          errors.emplace_back(
              "ConSan mutation proof did not remove only the ordinary acquire global_inv");
        }
        continue;
      }

      if (!patch.fault_companion_identity.empty() ||
          patch.anchor_offset != target->site->text_offset ||
          patch.original_size != sizeof(rdna4::VglobalMachineInst)) {
        errors.emplace_back("ConSan mutation proof found stale ordinary acquire scope metadata");
        continue;
      }
      std::array<uint32_t, 3> before{};
      std::array<uint32_t, 3> after{};
      std::memcpy(before.data(), original_text.data() + patch.anchor_offset, patch.original_size);
      std::memcpy(after.data(), replacement_text.data() + patch.anchor_offset, patch.original_size);
      const uint32_t original_scope = (before[1] >> 18u) & 0x3u;
      const uint32_t expected_scope_word = before[1] & ~(0x3u << 18u);
      if ((original_scope != 2u && original_scope != 3u) || before[0] != after[0] ||
          after[1] != expected_scope_word || before[2] != after[2]) {
        errors.emplace_back(
            "ConSan mutation proof changed fields other than the exact ordinary load scope");
      }
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineBarrierParticipantCountRewrite) {
      if (patch.original_size != 2u * sizeof(uint32_t) || !patch.original_participant_count ||
          !patch.target_participant_count ||
          *patch.original_participant_count == *patch.target_participant_count ||
          *patch.original_participant_count == 0u || *patch.original_participant_count > 63u ||
          *patch.target_participant_count == 0u || *patch.target_participant_count > 63u) {
        errors.emplace_back("ConSan mutation proof found invalid participant-count patch metadata");
        continue;
      }
      uint32_t before_setup = 0;
      uint32_t after_setup = 0;
      uint32_t before_literal = 0;
      uint32_t after_literal = 0;
      std::memcpy(&before_setup, original_text.data() + patch.anchor_offset, sizeof(before_setup));
      std::memcpy(&after_setup, replacement_text.data() + patch.anchor_offset, sizeof(after_setup));
      std::memcpy(&before_literal, original_text.data() + patch.anchor_offset + sizeof(uint32_t),
                  sizeof(before_literal));
      std::memcpy(&after_literal, replacement_text.data() + patch.anchor_offset + sizeof(uint32_t),
                  sizeof(after_literal));
      constexpr uint32_t kBarrierIdMask = 0x3fu;
      constexpr uint32_t kParticipantCountMask = 0x3fu << 16u;
      constexpr uint32_t kAllowedMask = kBarrierIdMask | kParticipantCountMask;
      const uint32_t barrier_id = before_literal & kBarrierIdMask;
      const bool exact_setup =
          before_setup == after_setup && barrier_id >= 1u && barrier_id <= 16u &&
          (before_literal & ~kAllowedMask) == 0u && (after_literal & ~kAllowedMask) == 0u &&
          (after_literal & kBarrierIdMask) == barrier_id &&
          ((before_literal >> 16u) & 0x3fu) == *patch.original_participant_count &&
          ((after_literal >> 16u) & 0x3fu) == *patch.target_participant_count &&
          (before_literal & ~kParticipantCountMask) == (after_literal & ~kParticipantCountMask);
      const uint64_t init_offset = patch.anchor_offset + patch.original_size;
      std::unique_ptr<Instruction> setup_instruction;
      std::unique_ptr<Instruction> init_instruction;
      try {
        std::array<uint32_t, 2> setup_words{before_setup, before_literal};
        setup_instruction.reset(decoder->decode(setup_words.data(), patch.anchor_offset));
        if (init_offset + sizeof(uint32_t) <= original_text.size()) {
          uint32_t init_word = 0;
          std::memcpy(&init_word, original_text.data() + init_offset, sizeof(init_word));
          init_instruction.reset(decoder->decode(&init_word, init_offset));
        }
      } catch (...) {
        errors.emplace_back("ConSan mutation proof could not decode participant-count setup");
        continue;
      }
      if (!exact_setup || !setup_instruction || setup_instruction->mnemonic() != "s_mov_b32" ||
          setup_instruction->size() != static_cast<int>(patch.original_size) || !init_instruction ||
          init_instruction->mnemonic() != "s_barrier_init") {
        errors.emplace_back(
            "ConSan mutation proof changed fields outside a proven literal M0 participant count");
      }
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineBarrierIdScopeRewrite) {
      std::array<uint32_t, 4> before_words{};
      std::array<uint32_t, 4> after_words{};
      if (patch.original_size == 0 || patch.original_size > sizeof(before_words) ||
          patch.original_size % sizeof(uint32_t) != 0) {
        errors.emplace_back("ConSan mutation proof found an invalid barrier ID/scope member size");
        continue;
      }
      std::memcpy(before_words.data(), original_text.data() + patch.anchor_offset,
                  patch.original_size);
      std::memcpy(after_words.data(), replacement_text.data() + patch.anchor_offset,
                  patch.original_size);
      std::unique_ptr<Instruction> before_instruction;
      std::unique_ptr<Instruction> after_instruction;
      try {
        before_instruction.reset(decoder->decode(before_words.data(), patch.anchor_offset));
      } catch (...) {
        errors.emplace_back("ConSan mutation proof could not decode a barrier ID/scope rewrite");
        continue;
      }
      const bool reinstrumented =
          result.staged_composition_validated &&
          std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &candidate) {
            return candidate.phase == ConSanPatchPhase::Instrumentation &&
                   candidate.anchor_offset <= patch.anchor_offset &&
                   patch.anchor_offset + patch.original_size <=
                       candidate.anchor_offset + candidate.original_size;
          });
      if (reinstrumented) {
        if (!before_instruction ||
            before_instruction->size() != static_cast<int>(patch.original_size)) {
          errors.emplace_back(
              "ConSan mutation proof changed the pristine shape of a composed barrier member");
          continue;
        }
        ConSanBarrierSite before_site;
        before_site.mnemonic = std::string(before_instruction->mnemonic());
        decode_barrier_operand(
            *before_instruction,
            original_text.subspan(patch.anchor_offset, static_cast<size_t>(patch.original_size)),
            before_site);
        if (!before_site.barrier_id || before_site.scope == ConSanBarrierSite::Scope::Unknown) {
          errors.emplace_back(
              "ConSan mutation proof found an unsupported pristine composed barrier member");
          continue;
        }
        barrier_id_scope_originals.push_back(std::move(before_site));
        barrier_id_scope_patches.push_back(&patch);
        barrier_id_scope_reinstrumented = true;
        continue;
      }
      try {
        after_instruction.reset(decoder->decode(after_words.data(), patch.anchor_offset));
      } catch (...) {
        errors.emplace_back("ConSan mutation proof could not decode a barrier ID/scope rewrite");
        continue;
      }
      if (!before_instruction || !after_instruction ||
          before_instruction->mnemonic() != after_instruction->mnemonic() ||
          before_instruction->size() != static_cast<int>(patch.original_size) ||
          after_instruction->size() != static_cast<int>(patch.original_size)) {
        errors.emplace_back("ConSan mutation proof changed the shape of a barrier ID/scope member");
        continue;
      }
      ConSanBarrierSite before_site;
      ConSanBarrierSite after_site;
      before_site.mnemonic = std::string(before_instruction->mnemonic());
      after_site.mnemonic = std::string(after_instruction->mnemonic());
      decode_barrier_operand(
          *before_instruction,
          original_text.subspan(patch.anchor_offset, static_cast<size_t>(patch.original_size)),
          before_site);
      decode_barrier_operand(
          *after_instruction,
          replacement_text.subspan(patch.anchor_offset, static_cast<size_t>(patch.original_size)),
          after_site);
      bool preserved = before_site.barrier_id && after_site.barrier_id &&
                       before_site.barrier_id != after_site.barrier_id &&
                       before_site.operand_source == after_site.operand_source &&
                       after_site.scope != ConSanBarrierSite::Scope::Unknown;
      if (before_site.mnemonic == "s_barrier_init" || before_site.mnemonic == "s_barrier_join" ||
          before_site.mnemonic == "s_barrier_signal" ||
          before_site.mnemonic == "s_barrier_signal_isfirst") {
        if (before_site.operand_source == ConSanBarrierSite::OperandSource::Immediate) {
          preserved &= (before_words[0] & ~0xffu) == (after_words[0] & ~0xffu);
        } else if (before_site.operand_source == ConSanBarrierSite::OperandSource::Literal32) {
          preserved &= before_words[0] == after_words[0];
        } else {
          preserved = false;
        }
      } else if (before_site.mnemonic == "s_barrier_wait") {
        preserved &= patch.original_size == sizeof(uint32_t) &&
                     (before_words[0] & 0xffff0000u) == (after_words[0] & 0xffff0000u);
      } else {
        preserved = false;
      }
      if (!preserved) {
        errors.emplace_back(
            "ConSan mutation proof changed fields outside a barrier ID/scope operand");
        continue;
      }
      barrier_id_scope_originals.push_back(std::move(before_site));
      barrier_id_scope_rewrites.push_back(std::move(after_site));
      barrier_id_scope_patches.push_back(&patch);
      continue;
    }
    if (patch.kind != ConSanPatchKind::InlineAtomicAddressRewrite &&
        patch.kind != ConSanPatchKind::InlineAtomicOrderRewrite &&
        patch.kind != ConSanPatchKind::InlineAtomicScopeRewrite) {
      continue;
    }
    if (patch.kind == ConSanPatchKind::InlineAtomicOrderRewrite) {
      if (patch.original_size == 0 || patch.original_size % sizeof(uint32_t) != 0) {
        errors.emplace_back(
            "ConSan mutation proof found an unexpected atomic cache-operation size");
        continue;
      }
      bool all_nops = true;
      const auto staged = composed_atomic_bytes(patch);
      const std::span<const uint8_t> after_bytes =
          staged.value_or(replacement_text.subspan(patch.anchor_offset, patch.original_size));
      for (uint32_t offset = 0; offset < patch.original_size; offset += sizeof(uint32_t)) {
        uint32_t after_word = 0;
        std::memcpy(&after_word, after_bytes.data() + offset, sizeof(after_word));
        all_nops &= after_word == nop;
      }
      if (!all_nops) {
        errors.emplace_back(
            "ConSan mutation proof did not remove the selected atomic cache operation");
      }
      continue;
    }

    if (patch.original_size != sizeof(rdna4::VdsMachineInst) &&
        patch.original_size != sizeof(rdna4::VflatMachineInst)) {
      errors.emplace_back("ConSan mutation proof found an unexpected atomic instruction size");
      continue;
    }
    std::array<uint32_t, 3> before{};
    std::array<uint32_t, 3> after{};
    std::memcpy(before.data(), original_text.data() + patch.anchor_offset, patch.original_size);
    const auto staged = composed_atomic_bytes(patch);
    const std::span<const uint8_t> after_bytes =
        staged.value_or(replacement_text.subspan(patch.anchor_offset, patch.original_size));
    std::memcpy(after.data(), after_bytes.data(), patch.original_size);
    if (patch.kind == ConSanPatchKind::InlineAtomicAddressRewrite) {
      bool valid_displacement = false;
      if (patch.original_size == sizeof(rdna4::VdsMachineInst)) {
        const int64_t delta = static_cast<int64_t>(after[0] & 0xffu) - (before[0] & 0xffu);
        valid_displacement = (after[0] & ~0xffu) == (before[0] & ~0xffu) && after[1] == before[1] &&
                             delta > 0 && delta <= 0xff &&
                             delta % static_cast<int64_t>(sizeof(uint32_t)) == 0 &&
                             (after[0] & 0xffu) % sizeof(uint32_t) == 0;
      } else {
        const int32_t before_offset = sign_extend_24(before[2] >> 8u);
        const int32_t after_offset = sign_extend_24(after[2] >> 8u);
        const int64_t delta = static_cast<int64_t>(after_offset) - before_offset;
        valid_displacement = before[0] == after[0] && before[1] == after[1] &&
                             (after[2] & 0xffu) == (before[2] & 0xffu) && delta > 0 &&
                             delta <= 0x7fffff &&
                             delta % static_cast<int64_t>(sizeof(uint32_t)) == 0;
      }
      if (!valid_displacement) {
        errors.emplace_back("ConSan mutation proof found the wrong atomic address displacement");
      }
    } else {
      if (patch.original_size != sizeof(rdna4::VflatMachineInst)) {
        errors.emplace_back(
            "ConSan mutation proof found scope mutation on an unsupported atomic encoding");
        continue;
      }
      const uint32_t expected_scope_word = before[1] & ~(0x3u << 18u);
      if (after[0] != before[0] || after[1] != expected_scope_word || after[2] != before[2]) {
        errors.emplace_back(
            "ConSan mutation proof changed fields other than the selected atomic scope");
      }
    }
  }

  if (!barrier_id_scope_originals.empty()) {
    const bool consistent_original_id_and_scope =
        std::ranges::all_of(barrier_id_scope_originals, [&](const ConSanBarrierSite &site) {
          return barrier_id_scope_originals.front().barrier_id && site.barrier_id &&
                 site.barrier_id == barrier_id_scope_originals.front().barrier_id &&
                 site.scope == barrier_id_scope_originals.front().scope;
        });
    const bool consistent_target_id_and_scope =
        std::ranges::all_of(barrier_id_scope_rewrites, [&](const ConSanBarrierSite &site) {
          return barrier_id_scope_rewrites.front().barrier_id && site.barrier_id &&
                 site.barrier_id == barrier_id_scope_rewrites.front().barrier_id &&
                 site.scope == barrier_id_scope_rewrites.front().scope &&
                 site.scope == barrier_scope_for_id(*site.barrier_id);
        });
    const bool pair_shape =
        barrier_id_scope_originals.size() == 2u &&
        barrier_id_scope_originals[0].mnemonic.find("signal") != std::string::npos &&
        barrier_id_scope_originals[1].mnemonic == "s_barrier_wait";
    bool lifecycle_shape =
        barrier_id_scope_originals.size() >= 4u &&
        barrier_id_scope_originals.front().mnemonic == "s_barrier_init" &&
        barrier_id_scope_originals[barrier_id_scope_originals.size() - 2u].mnemonic.find(
            "signal") != std::string::npos &&
        barrier_id_scope_originals.back().mnemonic == "s_barrier_wait";
    for (size_t i = 1; lifecycle_shape && i + 2u < barrier_id_scope_originals.size(); ++i)
      lifecycle_shape &= barrier_id_scope_originals[i].mnemonic == "s_barrier_join";
    if ((!pair_shape && !lifecycle_shape) || !consistent_original_id_and_scope ||
        (!barrier_id_scope_reinstrumented && !consistent_target_id_and_scope)) {
      errors.emplace_back(
          "ConSan mutation proof found an incomplete or inconsistent barrier ID/scope group");
    } else {
      // Re-derive the exact selected group from pristine executable bytes and
      // patch anchors. Do not use result.sync_sequences or lifecycle groups:
      // they are mutable transform output, not validation evidence.
      for (size_t i = 1; i < barrier_id_scope_patches.size(); ++i) {
        const ConSanPatchInfo &previous = *barrier_id_scope_patches[i - 1u];
        const ConSanPatchInfo &current = *barrier_id_scope_patches[i];
        const uint64_t gap_begin = previous.anchor_offset + previous.original_size;
        if (gap_begin > current.anchor_offset) {
          errors.emplace_back(
              "ConSan mutation proof found overlapping or out-of-order barrier group members");
          break;
        }
        const uint64_t gap_size = current.anchor_offset - gap_begin;
        if (lifecycle_shape && gap_size > 0u) {
          errors.emplace_back(
              "ConSan mutation proof found an omitted or noncontiguous barrier lifecycle member");
          break;
        }
        if (pair_shape && gap_size > 0u &&
            !std::equal(original_text.begin() + gap_begin,
                        original_text.begin() + current.anchor_offset,
                        replacement_text.begin() + gap_begin)) {
          errors.emplace_back(
              "ConSan mutation proof found changed bytes between barrier group members");
          break;
        }
      }
      if (lifecycle_shape) {
        const ConSanPatchInfo &wait_patch = *barrier_id_scope_patches.back();
        const uint64_t leave_offset = wait_patch.anchor_offset + wait_patch.original_size;
        if (leave_offset > original_text.size() ||
            sizeof(uint32_t) > original_text.size() - leave_offset ||
            leave_offset > replacement_text.size() ||
            sizeof(uint32_t) > replacement_text.size() - leave_offset) {
          errors.emplace_back(
              "ConSan mutation proof could not derive the lifecycle leave from pristine text");
        } else {
          std::array<uint32_t, 4> leave_words{};
          std::memcpy(leave_words.data(), original_text.data() + leave_offset, sizeof(uint32_t));
          std::unique_ptr<Instruction> leave_instruction;
          try {
            leave_instruction.reset(decoder->decode(leave_words.data(), leave_offset));
          } catch (...) {
          }
          uint32_t original_leave = 0;
          uint32_t replacement_leave = 0;
          std::memcpy(&original_leave, original_text.data() + leave_offset, sizeof(original_leave));
          std::memcpy(&replacement_leave, replacement_text.data() + leave_offset,
                      sizeof(replacement_leave));
          const bool leave_reinstrumented =
              result.staged_composition_validated &&
              std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &candidate) {
                return candidate.phase == ConSanPatchPhase::Instrumentation &&
                       candidate.anchor_offset <= leave_offset &&
                       leave_offset + sizeof(uint32_t) <=
                           candidate.anchor_offset + candidate.original_size;
              });
          if (!leave_instruction || leave_instruction->mnemonic() != "s_barrier_leave" ||
              leave_instruction->size() != static_cast<int>(sizeof(uint32_t)) ||
              static_cast<uint16_t>(original_leave) != 0u ||
              (!leave_reinstrumented && replacement_leave != original_leave)) {
            errors.emplace_back(
                "ConSan mutation proof did not preserve the derived fixed-zero lifecycle leave");
          }
          if (!barrier_id_scope_reinstrumented &&
              (*barrier_id_scope_rewrites.front().barrier_id < 1 ||
               *barrier_id_scope_rewrites.front().barrier_id > 16 ||
               barrier_id_scope_rewrites.front().scope != ConSanBarrierSite::Scope::Workgroup)) {
            errors.emplace_back(
                "ConSan mutation proof found invalid lifecycle target ID/scope metadata");
          }
        }
      }
    }
    if (!result.applied_fault_logical_identity)
      errors.emplace_back("ConSan mutation proof lost the retained logical fault identity");
  }

  if (!exact_barrier_drop_patches.empty()) {
    std::vector<std::string> sequence_identities;
    for (const ConSanPatchInfo *patch : exact_barrier_drop_patches) {
      if (std::ranges::find(sequence_identities, patch->fault_sequence_identity) ==
          sequence_identities.end())
        sequence_identities.push_back(patch->fault_sequence_identity);
    }
    std::ranges::sort(sequence_identities, [&](const std::string &left, const std::string &right) {
      const auto first_anchor = [&](const std::string &identity) {
        uint64_t value = std::numeric_limits<uint64_t>::max();
        for (const ConSanPatchInfo *patch : exact_barrier_drop_patches)
          if (patch->fault_sequence_identity == identity)
            value = std::min(value, patch->anchor_offset);
        return value;
      };
      return first_anchor(left) < first_anchor(right);
    });
    std::vector<ExactBarrierDropPair> pairs;
    bool metadata_matches = sequence_identities.size() == 1u || sequence_identities.size() == 2u;
    std::string reason;
    for (const std::string &sequence_identity : sequence_identities) {
      const auto member = std::ranges::find_if(exact_barrier_drop_patches, [&](const auto *patch) {
        return patch->fault_sequence_identity == sequence_identity;
      });
      ConSanOptions selection;
      selection.fault_site_identity = (*member)->fault_primary_identity;
      selection.fault_barrier_sequence_identity = sequence_identity;
      const auto pair = resolve_exact_barrier_drop_pair(pristine_inventory, selection, &reason);
      const size_t member_count = static_cast<size_t>(
          std::ranges::count_if(exact_barrier_drop_patches, [&](const auto *patch) {
            return patch->fault_sequence_identity == sequence_identity;
          }));
      metadata_matches &= pair.has_value() && member_count == 2u;
      if (!pair)
        continue;
      metadata_matches &= std::ranges::all_of(exact_barrier_drop_patches, [&](const auto *patch) {
        return patch->fault_sequence_identity != sequence_identity ||
               (patch->fault_primary_identity == pair->primary->identity &&
                patch->fault_companion_identity == pair->companion->identity &&
                patch->original_size == sizeof(uint32_t));
      });
      metadata_matches &= std::ranges::any_of(exact_barrier_drop_patches, [&](const auto *patch) {
        return patch->fault_sequence_identity == sequence_identity &&
               patch->anchor_offset == pair->primary->text_offset;
      });
      metadata_matches &= std::ranges::any_of(exact_barrier_drop_patches, [&](const auto *patch) {
        return patch->fault_sequence_identity == sequence_identity &&
               patch->anchor_offset == pair->companion->text_offset;
      });
      pairs.push_back(*pair);
    }
    if (pairs.size() == 1u) {
      metadata_matches &= exact_barrier_drop_patches.size() == 2u &&
                          result.applied_fault_logical_identity == pairs.front().sequence->identity;
    } else if (pairs.size() == 2u) {
      ConSanOptions selection;
      selection.fault_site_identity = pairs[0].primary->identity;
      selection.fault_barrier_sequence_identity = pairs[0].sequence->identity;
      selection.fault_barrier_companion_site_identity = pairs[1].primary->identity;
      selection.fault_barrier_companion_sequence_identity = pairs[1].sequence->identity;
      const auto group = resolve_exact_barrier_drop_group(pristine_inventory, selection, &reason);
      metadata_matches &=
          exact_barrier_drop_patches.size() == 4u && group &&
          result.applied_fault_logical_identity == exact_barrier_drop_group_identity(*group);
    } else {
      metadata_matches = false;
    }
    if (!metadata_matches) {
      errors.emplace_back(
          "ConSan mutation proof could not rederive the complete exact whole-barrier drop group");
    }
  }

  if (markerless_target != nullptr || !markerless_sources.empty()) {
    if (markerless_target == nullptr || markerless_sources.size() != 2u ||
        !markerless_target->barrier_move_direction ||
        (*markerless_target->barrier_move_direction != ConSanBarrierMoveDirection::Earlier &&
         *markerless_target->barrier_move_direction != ConSanBarrierMoveDirection::Later)) {
      errors.emplace_back(
          "ConSan mutation proof found an incomplete markerless barrier relocation record");
      return errors;
    }
    std::ranges::sort(markerless_sources, {}, &ConSanPatchInfo::anchor_offset);
    const ConSanPatchInfo &first = *markerless_sources[0];
    const ConSanPatchInfo &second = *markerless_sources[1];
    const ConSanPatchInfo &target = *markerless_target;
    if (first.original_size != sizeof(uint32_t) || second.original_size != sizeof(uint32_t) ||
        first.anchor_offset + sizeof(uint32_t) != second.anchor_offset) {
      errors.emplace_back(
          "ConSan mutation proof did not identify an adjacent ordered barrier pair");
      return errors;
    }
    if (target.original_size == 0 || target.original_size % sizeof(uint32_t) != 0) {
      errors.emplace_back("ConSan mutation proof found an unsupported displaced destination size");
      return errors;
    }
    const uint32_t expected_trampoline_size =
        2u * sizeof(uint32_t) + target.original_size + sizeof(uint32_t);
    if (target.trampoline_size != expected_trampoline_size ||
        target.trampoline_offset > replacement_text.size() ||
        target.trampoline_size > replacement_text.size() - target.trampoline_offset) {
      errors.emplace_back(
          "ConSan mutation proof found incorrect whole-pair trampoline byte accounting");
      return errors;
    }

    const ConSanBarrierMoveDirection direction = *target.barrier_move_direction;
    if ((direction == ConSanBarrierMoveDirection::Earlier &&
         target.anchor_offset + target.original_size > first.anchor_offset) ||
        (direction == ConSanBarrierMoveDirection::Later &&
         target.anchor_offset < second.anchor_offset + second.original_size)) {
      errors.emplace_back(
          "ConSan mutation proof found the relocation target on the wrong side of the pair");
      return errors;
    }
    const bool has_structured_proof =
        target.barrier_move_cfg_contract != ConSanBarrierMoveCfgContract::SameBlock ||
        target.structured_guard_block_index.has_value() ||
        target.structured_destination_block_index.has_value() ||
        target.structured_source_block_index.has_value() ||
        target.structured_guard_offset.has_value() ||
        target.structured_destination_offset.has_value() ||
        target.structured_source_offset.has_value();
    auto original_blocks = build_original_proof_basic_blocks(original, *decoder, arch);
    const auto containing_block_index = [&](uint64_t offset) -> std::optional<uint32_t> {
      for (size_t i = 0; i < original_blocks.size(); ++i) {
        const BasicBlock *block = original_blocks[i].get();
        if (block != nullptr && offset >= block->start_offset() && offset < block->end_offset())
          return static_cast<uint32_t>(i);
      }
      return std::nullopt;
    };
    const auto original_source_block = containing_block_index(first.anchor_offset);
    const auto original_destination_block = containing_block_index(target.anchor_offset);
    if (!original_source_block || !original_destination_block) {
      errors.emplace_back("ConSan mutation proof could not locate barrier relocation CFG blocks");
      return errors;
    }
    if (*original_source_block != *original_destination_block && !has_structured_proof) {
      errors.emplace_back(
          "ConSan mutation proof found a cross-block move without a structured CFG contract and "
          "metadata");
      return errors;
    }
    if (has_structured_proof) {
      if (!target.structured_guard_block_index || !target.structured_destination_block_index ||
          !target.structured_source_block_index || !target.structured_guard_offset ||
          !target.structured_destination_offset || !target.structured_source_offset ||
          direction != ConSanBarrierMoveDirection::Earlier ||
          *target.structured_destination_offset != target.anchor_offset ||
          *target.structured_source_block_index == *target.structured_destination_block_index) {
        errors.emplace_back("ConSan mutation proof found incomplete structured CFG metadata");
        return errors;
      }
      if (*original_source_block != *target.structured_source_block_index ||
          *original_destination_block != *target.structured_destination_block_index) {
        errors.emplace_back(
            "ConSan mutation proof found structured CFG block metadata that does not "
            "contain the selected relocation instructions");
        return errors;
      }
      std::optional<StructuredExecDiamondProof> rederived;
      if (target.barrier_move_cfg_contract ==
          ConSanBarrierMoveCfgContract::CompletingStructuredDiamond) {
        rederived = prove_completing_structured_diamond(
            original_blocks, *target.structured_source_block_index,
            *target.structured_destination_block_index, first.anchor_offset, target.anchor_offset);
      } else if (target.barrier_move_cfg_contract ==
                 ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond) {
        rederived = prove_structured_exec_diamond(
            original_blocks, *target.structured_source_block_index,
            *target.structured_destination_block_index, first.anchor_offset);
      }
      if (!rederived || rederived->guard_block_index != *target.structured_guard_block_index ||
          rederived->source_block_index != *target.structured_source_block_index ||
          rederived->destination_block_index != *target.structured_destination_block_index ||
          rederived->guard_offset != *target.structured_guard_offset ||
          rederived->source_offset != *target.structured_source_offset) {
        errors.emplace_back("ConSan mutation proof could not rederive its structured CFG contract");
        return errors;
      }
    }
    const auto forward = compute_sopp_branch_simm16(target.anchor_offset, target.trampoline_offset);
    uint32_t replacement_anchor = 0;
    std::memcpy(&replacement_anchor, replacement_text.data() + target.anchor_offset,
                sizeof(replacement_anchor));
    if (!forward || replacement_anchor != build_s_branch(*forward, arch)) {
      errors.emplace_back(
          "ConSan mutation proof found an invalid branch to the whole-pair trampoline");
      return errors;
    }
    for (uint32_t offset = sizeof(uint32_t); offset < target.original_size;
         offset += sizeof(uint32_t)) {
      uint32_t anchor_tail = 0;
      std::memcpy(&anchor_tail, replacement_text.data() + target.anchor_offset + offset,
                  sizeof(anchor_tail));
      if (anchor_tail != nop) {
        errors.emplace_back(
            "ConSan mutation proof did not fully replace the displaced destination anchor");
        return errors;
      }
    }

    std::array<uint32_t, 2> original_barriers{};
    std::memcpy(&original_barriers[0], original_text.data() + first.anchor_offset,
                sizeof(uint32_t));
    std::memcpy(&original_barriers[1], original_text.data() + second.anchor_offset,
                sizeof(uint32_t));
    const ByteRange mutation_body{target.trampoline_offset,
                                  target.trampoline_offset + target.trampoline_size};
    const bool body_reinstrumented =
        std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
          if (patch.phase != ConSanPatchPhase::Instrumentation || patch.original_size == 0)
            return false;
          const ByteRange anchor{patch.anchor_offset, patch.anchor_offset + patch.original_size};
          return ranges_overlap(mutation_body, anchor);
        });
    // A staged composition validates pristine -> mutation and mutation ->
    // instrumentation independently before combining their inventories. If
    // the second validated stage replaced an instruction inside this body, the
    // final image cannot also retain the intermediate raw bytes. Cross-phase
    // nesting is checked by byte accounting; retain the source-NOP and
    // entry-branch checks above and rely on both recorded stage proofs for the
    // nested rewrite.
    if (body_reinstrumented && result.staged_composition_validated)
      return errors;
    const uint64_t barrier_body_offset =
        direction == ConSanBarrierMoveDirection::Earlier ? 0u : target.original_size;
    const uint64_t destination_body_offset =
        direction == ConSanBarrierMoveDirection::Earlier ? 2u * sizeof(uint32_t) : 0u;
    const uint8_t *body = replacement_text.data() + target.trampoline_offset;
    if (std::memcmp(body + barrier_body_offset, original_barriers.data(),
                    sizeof(original_barriers)) != 0) {
      errors.emplace_back(
          "ConSan mutation proof did not preserve an adjacent byte-identical ordered barrier "
          "pair");
      return errors;
    }
    if (std::memcmp(body + destination_body_offset, original_text.data() + target.anchor_offset,
                    target.original_size) != 0) {
      errors.emplace_back(
          "ConSan mutation proof did not preserve the displaced destination semantics");
      return errors;
    }

    // The body is exactly pair + destination + return, and the two
    // non-overlapping structural slots above exhaust its payload. That proves
    // exact-once execution without rejecting incidental byte-pattern matches.
    const uint64_t payload_size = target.trampoline_size - sizeof(uint32_t);
    const uint64_t return_pc = target.trampoline_offset + payload_size;
    const auto ret =
        compute_sopp_branch_simm16(return_pc, target.anchor_offset + target.original_size);
    uint32_t return_word = 0;
    std::memcpy(&return_word, replacement_text.data() + return_pc, sizeof(return_word));
    if (!ret || return_word != build_s_branch(*ret, arch)) {
      errors.emplace_back("ConSan mutation proof found an invalid whole-pair trampoline return");
    }
  }
  return errors;
}

[[nodiscard]] std::vector<std::string> validate_sc_perturbation_semantics(
    std::span<const uint8_t> original_image, const AmdGpuCodeObject &original,
    const AmdGpuCodeObject &replacement, const ConSanResult &result) {
  std::vector<std::string> errors;
  const size_t perturbation_patch_count = static_cast<size_t>(std::ranges::count(
      result.patches, ConSanPatchKind::TrampolineScPerturbation, &ConSanPatchInfo::kind));
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineScPerturbation &&
        (patch.perturbation_edge || !patch.perturbation_sequence_identity.empty() ||
         !patch.perturbation_source_candidate_identity.empty() ||
         !patch.perturbation_source_sequence_identity.empty() ||
         !patch.perturbation_source_container_name.empty() ||
         !patch.perturbation_source_anchor_identity.empty() ||
         patch.perturbation_source_anchor_size != 0u ||
         patch.perturbation_source_owner_descriptor_file_offset ||
         patch.perturbation_composite_atomic_overlap ||
         patch.perturbation_composite_removed_boundary)) {
      errors.emplace_back(
          "ConSan SC perturbation proof found semantic metadata on another patch kind");
    }
  }
  if (result.applied_perturbations != perturbation_patch_count) {
    errors.emplace_back("ConSan SC perturbation proof found inconsistent applied patch count");
  }
  if (perturbation_patch_count == 0u)
    return errors;
  if (original.text_sections().size() != 1u || replacement.text_sections().size() != 1u) {
    errors.emplace_back(
        "ConSan SC perturbation proof requires one original and replacement text section");
    return errors;
  }

  ConSanOptions inventory_options;
  inventory_options.flavor = ConSanFlavor::SuperCollider;
  inventory_options.fault_dry_run = true;
  const ConSanResult pristine = try_patch_consan_impl(original_image, inventory_options);
  const bool carried_stage =
      result.carried_composite_instrumentation_stage && !result.staged_composition_validated;
  if (!carried_stage && !pristine.errors.empty()) {
    errors.emplace_back(
        "ConSan SC perturbation proof could not rederive pristine synchronization semantics");
    return errors;
  }

  const auto original_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(original.text_sections().front()->data()),
      original.text_sections().front()->size());
  const auto replacement_text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(replacement.text_sections().front()->data()),
      replacement.text_sections().front()->size());
  const rj_code_arch_t arch = arch_for_target(original.target_id());
  const uint32_t nop = build_s_nop(0, arch);
  std::unordered_set<std::string> proven_edges;

  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.kind != ConSanPatchKind::TrampolineScPerturbation)
      continue;
    if (patch.phase != ConSanPatchPhase::Instrumentation) {
      errors.emplace_back("ConSan SC perturbation proof found a non-instrumentation patch phase");
      continue;
    }
    if (!patch.perturbation_edge || patch.perturbation_sequence_identity.empty()) {
      errors.emplace_back("ConSan SC perturbation proof found incomplete semantic identity");
      continue;
    }
    const bool composite = result.staged_composition_validated;
    if (composite && (patch.perturbation_source_candidate_identity.empty() ||
                      patch.perturbation_source_sequence_identity.empty() ||
                      patch.perturbation_source_container_name.empty() ||
                      patch.perturbation_source_anchor_identity.empty() ||
                      patch.perturbation_source_anchor_size == 0u ||
                      (patch.perturbation_source_in_kernel &&
                       !patch.perturbation_source_owner_descriptor_file_offset))) {
      errors.emplace_back(
          "ConSan composite perturbation proof found incomplete pristine identity and owner");
      continue;
    }
    const std::string &proof_sequence_identity = composite
                                                     ? patch.perturbation_source_sequence_identity
                                                     : patch.perturbation_sequence_identity;
    const uint64_t proof_anchor_offset =
        composite ? patch.perturbation_source_anchor_offset : patch.anchor_offset;
    const uint32_t proof_anchor_size =
        composite ? patch.perturbation_source_anchor_size : patch.original_size;
    const std::string proven_edge =
        proof_sequence_identity +
        (*patch.perturbation_edge == ConSanPerturbationEdge::Release ? "|release" : "|acquire");
    if (!proven_edges.insert(proven_edge).second) {
      errors.emplace_back("ConSan SC perturbation proof found a duplicate sequence edge");
      continue;
    }
    if (patch.original_size != sizeof(uint32_t) && patch.original_size != 2u * sizeof(uint32_t) &&
        patch.original_size != 3u * sizeof(uint32_t)) {
      errors.emplace_back("ConSan SC perturbation proof found an unsupported boundary size");
      continue;
    }
    const uint32_t expected_trampoline_size = patch.original_size + 2u * sizeof(uint32_t);
    if (patch.trampoline_size != expected_trampoline_size) {
      errors.emplace_back("ConSan SC perturbation proof found a non-exact trampoline size");
      continue;
    }
    if (proof_anchor_offset > original_text.size() ||
        proof_anchor_size > original_text.size() - proof_anchor_offset ||
        patch.anchor_offset > replacement_text.size() ||
        patch.original_size > replacement_text.size() - patch.anchor_offset ||
        patch.trampoline_offset > replacement_text.size() ||
        patch.trampoline_size > replacement_text.size() - patch.trampoline_offset) {
      errors.emplace_back("ConSan SC perturbation proof found an out-of-range patch");
      continue;
    }

    const ConSanResult &semantic_inventory = carried_stage ? result : pristine;
    const auto candidate_matches = [&](const ConSanPerturbationCandidate &candidate) {
      return candidate.eligible && candidate.sequence_identity == proof_sequence_identity &&
             candidate.edge == *patch.perturbation_edge &&
             (!composite || (candidate.identity == patch.perturbation_source_candidate_identity &&
                             candidate.container_name == patch.perturbation_source_container_name &&
                             candidate.in_kernel == patch.perturbation_source_in_kernel));
    };
    const size_t candidate_count = static_cast<size_t>(
        std::ranges::count_if(semantic_inventory.perturbation_candidates, candidate_matches));
    const auto candidate =
        std::ranges::find_if(semantic_inventory.perturbation_candidates, candidate_matches);
    if (candidate_count != 1u || candidate == semantic_inventory.perturbation_candidates.end()) {
      errors.emplace_back(
          "ConSan SC perturbation proof could not match one pristine admitted sequence edge");
      continue;
    }
    if (composite && candidate->in_kernel) {
      const auto source_owner =
          std::ranges::find(pristine.kernels, candidate->container_name, &ConSanKernelInfo::name);
      if (source_owner == pristine.kernels.end() ||
          patch.perturbation_source_owner_descriptor_file_offset !=
              source_owner->descriptor_file_offset) {
        errors.emplace_back(
            "ConSan composite perturbation proof did not retain its pristine kernel owner");
        continue;
      }
    } else if (candidate->in_kernel) {
      const auto owner = std::ranges::find(semantic_inventory.kernels, candidate->container_name,
                                           &ConSanKernelInfo::name);
      if (owner == semantic_inventory.kernels.end() ||
          patch.owner_descriptor_file_offsets.size() != 1u ||
          patch.owner_descriptor_file_offsets.front() != owner->descriptor_file_offset) {
        errors.emplace_back("ConSan SC perturbation proof did not retain its exact kernel owner");
        continue;
      }
    } else if (!patch.owner_descriptor_file_offsets.empty()) {
      errors.emplace_back(
          "ConSan SC perturbation proof attached a kernel owner to a local function");
      continue;
    }
    if (!carried_stage) {
      const auto sequence = std::ranges::find(pristine.sync_sequences, proof_sequence_identity,
                                              &ConSanSyncSequence::identity);
      if (sequence == pristine.sync_sequences.end() || !sequence->basic_block_index ||
          !consan_sync_confidence_meets(sequence->confidence,
                                        ConSanSemanticConfidence::Conservative) ||
          !perturbation_rejection_reason(pristine, *sequence, candidate->kind,
                                         *patch.perturbation_edge)
               .empty() ||
          candidate->container_name != sequence->container_name ||
          candidate->in_kernel != sequence->in_kernel ||
          candidate->basic_block_index != *sequence->basic_block_index ||
          candidate->ordered_member_identities != sequence->member_event_identities) {
        errors.emplace_back("ConSan SC perturbation proof could not rederive sequence role, "
                            "confidence, container, and block");
        continue;
      }
      const std::string &outer_member = *patch.perturbation_edge == ConSanPerturbationEdge::Release
                                            ? sequence->member_event_identities.front()
                                            : sequence->member_event_identities.back();
      const ConSanSyncEvent *anchor_event = find_sync_event(pristine, outer_member);
      if (candidate->anchor_event_identity != outer_member || anchor_event == nullptr ||
          (composite && patch.perturbation_source_anchor_identity != outer_member) ||
          anchor_event->container_name != sequence->container_name ||
          anchor_event->in_kernel != sequence->in_kernel ||
          candidate->anchor_text_offset != anchor_event->text_offset ||
          candidate->anchor_size != anchor_event->size ||
          proof_anchor_offset != anchor_event->text_offset ||
          proof_anchor_size != anchor_event->size || patch.original_size != proof_anchor_size) {
        errors.emplace_back(
            "ConSan SC perturbation proof did not target the rederived outer sequence member");
        continue;
      }
    }
    const auto forward = compute_sopp_branch_simm16(patch.anchor_offset, patch.trampoline_offset);
    uint32_t anchor_word = 0;
    std::memcpy(&anchor_word, replacement_text.data() + patch.anchor_offset, sizeof(anchor_word));
    if (!forward || anchor_word != build_s_branch(*forward, arch))
      errors.emplace_back("ConSan SC perturbation proof found an invalid anchor branch");
    for (uint32_t offset = sizeof(uint32_t); offset < patch.original_size;
         offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, replacement_text.data() + patch.anchor_offset + offset, sizeof(word));
      if (word != nop) {
        errors.emplace_back("ConSan SC perturbation proof found a non-NOP anchor tail");
        break;
      }
    }

    const uint32_t original_body_offset =
        *patch.perturbation_edge == ConSanPerturbationEdge::Release ? sizeof(uint32_t) : 0u;
    const uint8_t *body = replacement_text.data() + patch.trampoline_offset + original_body_offset;
    if (patch.perturbation_composite_atomic_overlap && composite) {
      const auto mutation = std::ranges::find_if(result.patches, [&](const ConSanPatchInfo &item) {
        return item.phase == ConSanPatchPhase::Mutation &&
               item.anchor_offset == patch.anchor_offset &&
               item.original_size == patch.original_size &&
               (item.kind == ConSanPatchKind::InlineAtomicAddressRewrite ||
                item.kind == ConSanPatchKind::InlineAtomicOrderRewrite ||
                item.kind == ConSanPatchKind::InlineAtomicScopeRewrite);
      });
      if (mutation == result.patches.end() ||
          patch.perturbation_composite_removed_boundary !=
              (mutation->kind == ConSanPatchKind::InlineAtomicOrderRewrite)) {
        errors.emplace_back(
            "ConSan SC perturbation proof could not match its exact overlapping mutation");
      } else if (patch.perturbation_composite_removed_boundary) {
        bool all_nops = true;
        for (uint32_t offset = 0; offset < patch.original_size; offset += sizeof(uint32_t)) {
          uint32_t word = 0;
          std::memcpy(&word, body + offset, sizeof(word));
          all_nops &= word == nop;
        }
        if (!all_nops)
          errors.emplace_back("ConSan SC perturbation proof resurrected a removed cache operation");
      }
    } else if (std::memcmp(original_text.data() + proof_anchor_offset, body, patch.original_size) !=
               0) {
      errors.emplace_back(
          "ConSan SC perturbation proof did not preserve the exact original boundary bytes");
    }
    const uint32_t sleep_body_offset =
        *patch.perturbation_edge == ConSanPerturbationEdge::Release ? 0u : patch.original_size;
    uint32_t sleep_word = 0;
    std::memcpy(&sleep_word, replacement_text.data() + patch.trampoline_offset + sleep_body_offset,
                sizeof(sleep_word));
    bool bounded_sleep = false;
    for (uint16_t immediate = 1; immediate <= 15; ++immediate)
      bounded_sleep |= sleep_word == build_s_sleep(immediate, arch);
    if (!bounded_sleep) {
      errors.emplace_back(
          "ConSan SC perturbation proof found no single bounded sleep on the declared side");
    }

    const uint64_t return_offset =
        patch.trampoline_offset + patch.trampoline_size - sizeof(uint32_t);
    const auto ret =
        compute_sopp_branch_simm16(return_offset, patch.anchor_offset + patch.original_size);
    uint32_t return_word = 0;
    std::memcpy(&return_word, replacement_text.data() + return_offset, sizeof(return_word));
    if (!ret || return_word != build_s_branch(*ret, arch)) {
      errors.emplace_back("ConSan SC perturbation proof found an invalid exact boundary return");
    }
  }
  return errors;
}

struct DescriptorRequirement {
  bool allow_resource_delta = false;
  bool allow_entry_delta = false;
  bool has_dispatch_id_capture = false;
  bool dispatch_id_preload_inserted = false;
  uint16_t dispatch_id_original_user_sgpr_count = 0;
  uint16_t dispatch_id_expanded_user_sgpr_count = 0;
  uint16_t dispatch_id_original_kernarg_preload_length = 0;
  uint16_t dispatch_id_replacement_kernarg_preload_length = 0;
  uint16_t required_vgpr_count = 0;
  uint16_t required_sgpr_count = 0;
  uint32_t required_private_bytes = 0;
  uint32_t required_group_bytes = 0;
};

[[nodiscard]] std::vector<std::string>
validate_dispatch_id_prologue_semantics(const AmdGpuCodeObject &original,
                                        const AmdGpuCodeObject &replacement,
                                        const ConSanResult &result) {
  std::vector<std::string> errors;
  if (original.text_sections().size() != 1 || replacement.text_sections().size() != 1)
    return errors;
  const auto *replacement_section = replacement.text_sections().front();
  const std::span<const uint8_t> replacement_text(
      reinterpret_cast<const uint8_t *>(replacement_section->data()), replacement_section->size());
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  const auto validate_entry = [&](const ConSanPatchInfo &patch, uint64_t entry_offset,
                                  std::string_view kernel_name) {
    const uint32_t guest_input_count =
        static_cast<uint32_t>(patch.dispatch_id_original_user_sgpr_count) +
        patch.dispatch_id_system_sgpr_count;
    if (patch.dispatch_id_preload_inserted && patch.dispatch_id_source_sgpr > guest_input_count) {
      errors.emplace_back(
          "ConSan final validation found invalid dispatch-ID restore bounds for kernel '" +
          std::string(kernel_name) + "'");
      return;
    }
    const uint32_t restore_count =
        patch.dispatch_id_preload_inserted
            ? (patch.dispatch_id_kernarg_reload_count != 0
                   ? patch.dispatch_id_kernarg_reload_sgpr - patch.dispatch_id_source_sgpr
                   : guest_input_count - patch.dispatch_id_source_sgpr)
            : 0u;
    const uint64_t reload_words = patch.dispatch_id_kernarg_reload_count != 0
                                      ? 2u * patch.dispatch_id_kernarg_reload_count + 1u
                                      : 0u;
    const uint64_t required_words =
        4u + 2u * (restore_count + patch.dispatch_id_shifted_system_sgpr_count) + reload_words;
    const uint64_t required_bytes = required_words * sizeof(uint32_t);
    if (entry_offset > replacement_text.size() ||
        required_bytes > replacement_text.size() - entry_offset) {
      errors.emplace_back(
          "ConSan final validation found a truncated dispatch-ID entry prologue for kernel '" +
          std::string(kernel_name) + "'");
      return;
    }
    const auto read_word = [&](uint64_t word_index) {
      uint32_t word = 0;
      std::memcpy(&word, replacement_text.data() + entry_offset + word_index * sizeof(uint32_t),
                  sizeof(word));
      return word;
    };
    uint64_t cursor = 0;
    const uint16_t persistent = *patch.dispatch_id_capture_sgpr;
    const auto dependency_delay = instrumentation::build_salu_dependency_delay(arch);
    if (!dependency_delay) {
      errors.emplace_back("ConSan final validation cannot encode the target SALU dependency delay");
      return;
    }
    const auto expect_write = [&](uint32_t expected, std::string_view what) {
      if (read_word(cursor++) != expected || read_word(cursor++) != *dependency_delay) {
        errors.emplace_back("ConSan final validation found an invalid dispatch-ID " +
                            std::string(what) + " in kernel '" + std::string(kernel_name) + "'");
      }
    };
    expect_write(build_s_mov_b32(persistent, patch.dispatch_id_source_sgpr, arch),
                 "low-word capture");
    expect_write(build_s_mov_b32(static_cast<uint16_t>(persistent + 1u),
                                 static_cast<uint16_t>(patch.dispatch_id_source_sgpr + 1u), arch),
                 "high-word capture");
    expect_write(build_s_add_u32(persistent, persistent, scalar_positive_inline_u32(1), arch),
                 "low-word sentinel encoding");
    expect_write(build_s_addc_u32(static_cast<uint16_t>(persistent + 1u),
                                  static_cast<uint16_t>(persistent + 1u),
                                  scalar_positive_inline_u32(0), arch),
                 "high-word sentinel encoding");
    for (uint32_t i = 0; i < restore_count; ++i) {
      const uint16_t destination = static_cast<uint16_t>(patch.dispatch_id_source_sgpr + i);
      expect_write(build_s_mov_b32(destination, static_cast<uint16_t>(destination + 2u), arch),
                   "ascending guest restore");
    }
    for (uint16_t i = 0; i < patch.dispatch_id_shifted_system_sgpr_count; ++i) {
      const uint16_t destination =
          static_cast<uint16_t>(patch.dispatch_id_original_user_sgpr_count + i);
      expect_write(
          build_s_mov_b32(destination,
                          static_cast<uint16_t>(destination + patch.dispatch_id_system_sgpr_shift),
                          arch),
          "system-SGPR restore");
    }
    if (patch.dispatch_id_kernarg_reload_count != 0) {
      for (uint16_t i = 0; i < patch.dispatch_id_kernarg_reload_count; ++i) {
        const auto load = instrumentation::build_s_load_dword(
            static_cast<uint16_t>(patch.dispatch_id_kernarg_reload_sgpr + i),
            patch.dispatch_id_kernarg_reload_base_sgpr,
            static_cast<uint32_t>(patch.dispatch_id_kernarg_reload_offset_dwords + i) *
                sizeof(uint32_t),
            arch);
        if (!load || read_word(cursor++) != (*load)[0] || read_word(cursor++) != (*load)[1]) {
          errors.emplace_back("ConSan final validation found an invalid dispatch-ID kernarg "
                              "reload in kernel '" +
                              std::string(kernel_name) + "'");
        }
      }
      const auto wait = instrumentation::build_s_wait_scalar_load0(arch);
      if (!wait || read_word(cursor++) != *wait) {
        errors.emplace_back("ConSan final validation found an invalid dispatch-ID kernarg wait "
                            "in kernel '" +
                            std::string(kernel_name) + "'");
      }
    }
  };

  for (const ConSanPatchInfo &patch : result.patches) {
    if (!patch.dispatch_id_capture_sgpr)
      continue;
    const auto owner =
        std::ranges::find_if(original.kernels(), [&](const AmdGpuKernelInfo &kernel) {
          return kernel.entry_text_offset == patch.anchor_offset;
        });
    if (owner == original.kernels().end()) {
      errors.emplace_back("ConSan final validation could not resolve a dispatch-ID prologue owner");
      continue;
    }
    validate_entry(patch,
                   patch.dispatch_id_primary_prologue_offset.value_or(patch.trampoline_offset),
                   owner->name);
    KD descriptor{};
    std::memcpy(&descriptor, original.image_data() + owner->descriptor_file_offset,
                sizeof(descriptor));
    const bool has_kernarg_preload =
        AMDHSA_BITS_GET(descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH) != 0;
    if (has_kernarg_preload) {
      validate_entry(
          patch,
          patch.dispatch_id_secondary_prologue_offset.value_or(patch.trampoline_offset + 256u),
          owner->name);
    }
  }
  return errors;
}

[[nodiscard]] std::vector<std::string>
validate_inline_exact_shadow_semantics(const AmdGpuCodeObject &replacement,
                                       const ConSanResult &result) {
  std::vector<std::string> errors;
  const auto is_exact_patch = [](const ConSanPatchInfo &patch) {
    return patch.kind == ConSanPatchKind::InlineMoiExactShadowStore ||
           patch.kind == ConSanPatchKind::TrampolineMoiExactShadowStore;
  };
  if (!std::ranges::any_of(result.patches, is_exact_patch))
    return errors;
  const bool has_global_exact_patch =
      std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
        return is_exact_patch(patch) && patch.workgroup_shadow_size == 0u;
      });
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  if (replacement.text_sections().size() != 1 ||
      (has_global_exact_patch && !result.resolved_moi_dispatch_id_sgpr &&
       !is_rdna4_family_arch(arch))) {
    errors.emplace_back(
        "ConSan final validation found exact-shadow publication without dispatch-ID state");
    return errors;
  }

  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    errors.emplace_back(
        "ConSan final validation could not create an exact-shadow semantic decoder");
    return errors;
  }
  const Section *section = replacement.text_sections().front();
  const std::span<const uint8_t> text(reinterpret_cast<const uint8_t *>(section->data()),
                                      section->size());

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
      std::unique_ptr<Instruction> instruction;
      try {
        instruction.reset(decoder->decode(words.data(), body_offset + cursor));
      } catch (...) {
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
    if (patch.workgroup_shadow_size != 0u) {
      const auto candidate = std::ranges::find_if(result.moi_candidates, [&](const auto &item) {
        return item.text_offset == patch.anchor_offset &&
               (item.kind == ConSanLdsAccessKind::Read || item.kind == ConSanLdsAccessKind::Write ||
                item.kind == ConSanLdsAccessKind::Atomic);
      });
      const uint32_t minimum_cell_count =
          candidate == result.moi_candidates.end()
              ? 0u
              : (candidate->width_bits + consan_moi_exact_shadow::granule_bytes * 8u - 1u) /
                    (consan_moi_exact_shadow::granule_bytes * 8u);
      const size_t decoded_local_exchange_cell_count =
          patch.workgroup_shadow_compact
              ? decoded_local_exchange_b32_count + 2u * decoded_local_exchange_b64_count
              : decoded_local_exchange_b64_count;
      const bool needs_compact_loop = minimum_cell_count > decoded_local_exchange_cell_count;
      const bool cdna = arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
      const uint16_t old_value_offset = cdna ? 6u : 5u;
      const auto expected_local_exchange_b32 = instrumentation::build_ds_storexchg_rtn_b32(
          static_cast<uint16_t>(scratch + old_value_offset), scratch,
          static_cast<uint16_t>(scratch + 2u), 0u, arch);
      const auto expected_local_exchange_b64 = instrumentation::build_ds_storexchg_rtn_b64(
          static_cast<uint16_t>(scratch + old_value_offset), scratch,
          static_cast<uint16_t>(scratch + 2u), 0u, arch);
      const std::array<uint32_t, 1> duplicate_compact_metadata = {
          build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 3u),
                              vector_source_vgpr(static_cast<uint16_t>(scratch + 2u)), arch)};
      const uint16_t diagnostic_claim_offset = cdna ? 8u : 3u;
      const auto expected_first_diagnostic_claim = instrumentation::build_flat_atomic_cmpswap_b32(
          scratch, static_cast<uint16_t>(scratch + diagnostic_claim_offset),
          static_cast<uint16_t>(scratch + diagnostic_claim_offset),
          /*return_old_value=*/true,
          /*scope=*/2, arch);
      if ((patch.workgroup_shadow_compact && !expected_local_exchange_b32) ||
          !expected_local_exchange_b64 || !expected_first_diagnostic_claim || decoded_legacy_swap ||
          decoded_cas_count != decoded_local_exchange_cell_count ||
          decoded_local_exchange_count == 0u || candidate == result.moi_candidates.end() ||
          (needs_compact_loop && decoded_backward_local_loop_count == 0u) ||
          decoded_nonbackward_local_loop ||
          (patch.workgroup_shadow_compact
               ? count_pattern(*expected_local_exchange_b32) != decoded_local_exchange_b32_count ||
                     count_pattern(*expected_local_exchange_b64) !=
                         decoded_local_exchange_b64_count ||
                     count_pattern(duplicate_compact_metadata) < decoded_local_exchange_b64_count
               : decoded_local_exchange_b32_count != 0u ||
                     count_pattern(*expected_local_exchange_b64) !=
                         decoded_local_exchange_b64_count) ||
          count_pattern(*expected_first_diagnostic_claim) != decoded_cas_count) {
        errors.emplace_back(
            "ConSan final validation found invalid workgroup-local exact-shadow publication "
            "semantics: decoded_cas=" +
            std::to_string(decoded_cas_count) +
            " decoded_local_exchange=" + std::to_string(decoded_local_exchange_count) +
            " decoded_local_cells=" + std::to_string(decoded_local_exchange_cell_count) +
            " minimum_cells=" + std::to_string(minimum_cell_count) +
            " backward_loops=" + std::to_string(decoded_backward_local_loop_count) +
            " nonbackward_loop=" + std::to_string(decoded_nonbackward_local_loop) +
            " legacy_swap=" + std::to_string(decoded_legacy_swap) + " expected_exchange_b32=" +
            std::to_string(expected_local_exchange_b32 ? count_pattern(*expected_local_exchange_b32)
                                                       : 0u) +
            " expected_exchange_b64=" +
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
        scratch, static_cast<uint16_t>(scratch + 14u), static_cast<uint16_t>(scratch + 14u),
        /*return_old_value=*/true, /*scope=*/2, arch);
    const auto expected_dispatch_low_store = instrumentation::build_flat_store_b32(
        scratch, static_cast<uint16_t>(scratch + 2u), arch,
        offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id));
    const auto expected_dispatch_high_store = instrumentation::build_flat_store_b32(
        scratch, static_cast<uint16_t>(scratch + 3u), arch,
        offsetof(ConSanMoiInlineExactShadowSlot, dispatch_id) + sizeof(uint32_t));
    if (!expected_cas || !expected_dispatch_low_store || !expected_dispatch_high_store) {
      errors.emplace_back(
          "ConSan final validation could not encode exact-shadow semantic expectations");
      continue;
    }

    const size_t exact_cas_count = count_pattern(*expected_cas);
    const size_t transaction_count = exact_cas_count / 2u;
    std::vector<uint32_t> capture_low;
    std::vector<uint32_t> capture_high;
    if (result.resolved_moi_dispatch_id_sgpr) {
      capture_low.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(scratch + 2u),
                                                *result.resolved_moi_dispatch_id_sgpr, arch));
      capture_high.push_back(build_v_mov_b32_e32(
          static_cast<uint16_t>(scratch + 3u),
          static_cast<uint16_t>(*result.resolved_moi_dispatch_id_sgpr + 1u), arch));
    } else {
      const auto low = instrumentation::build_v_mov_b32_literal(
          static_cast<uint16_t>(scratch + 2u), static_cast<uint32_t>(result.moi_report_dispatch_id),
          arch);
      const auto high = instrumentation::build_v_mov_b32_literal(
          static_cast<uint16_t>(scratch + 3u),
          static_cast<uint32_t>(result.moi_report_dispatch_id >> 32u), arch);
      if (!low || !high) {
        errors.emplace_back(
            "ConSan final validation could not encode literal dispatch-ID expectations");
        continue;
      }
      capture_low = *low;
      capture_high = *high;
    }
    if (decoded_legacy_swap || decoded_cas_count < 4u || decoded_cas_count % 4u != 0u ||
        exact_cas_count != decoded_cas_count ||
        count_pattern(*expected_dispatch_low_store) != transaction_count ||
        count_pattern(*expected_dispatch_high_store) != transaction_count ||
        count_pattern(capture_low) < transaction_count ||
        count_pattern(capture_high) < transaction_count) {
      errors.emplace_back(
          "ConSan final validation found invalid versioned exact-shadow publication semantics: "
          "decoded_cas=" +
          std::to_string(decoded_cas_count) + " exact_cas=" + std::to_string(exact_cas_count) +
          " transactions=" + std::to_string(transaction_count) +
          " dispatch_low_stores=" + std::to_string(count_pattern(*expected_dispatch_low_store)) +
          " dispatch_high_stores=" + std::to_string(count_pattern(*expected_dispatch_high_store)) +
          " capture_low=" + std::to_string(count_pattern(capture_low)) +
          " capture_high=" + std::to_string(count_pattern(capture_high)) +
          " legacy_swap=" + std::to_string(decoded_legacy_swap));
    }
  }
  return errors;
}

[[nodiscard]] std::vector<std::string>
validate_inline_release_transaction_semantics(const AmdGpuCodeObject &replacement,
                                              const ConSanResult &result) {
  constexpr uint16_t kVccLo = 106u;
  std::vector<std::string> errors;
  const auto is_release_transaction = [&](const ConSanPatchInfo &patch) {
    if (patch.kind != ConSanPatchKind::TrampolineMoiInlineAtomicOrdering)
      return false;
    const auto event = std::ranges::find_if(result.sync_events, [&](const ConSanSyncEvent &item) {
      return item.kind == ConSanSyncEventKind::Atomic && item.text_offset == patch.anchor_offset;
    });
    if (event == result.sync_events.end())
      return false;
    const ConSanSyncSequence *sequence =
        find_consan_sync_sequence_for_event(result, event->identity);
    return sequence != nullptr &&
           (sequence->memory_role == ConSanSyncMemoryRole::Release ||
            sequence->memory_role == ConSanSyncMemoryRole::AcquireRelease ||
            sequence->memory_role == ConSanSyncMemoryRole::SequentiallyConsistent);
  };
  if (!std::ranges::any_of(result.patches, is_release_transaction))
    return errors;
  if (replacement.text_sections().size() != 1 ||
      (!result.resolved_moi_owner_vgpr && !result.resolved_moi_persistent_owner_sgpr) ||
      !result.resolved_moi_exec_save_sgpr) {
    errors.emplace_back(
        "ConSan final validation found release transaction without persistent state");
    return errors;
  }

  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  std::unique_ptr<Decoder> decoder = Decoder::create(arch);
  if (!decoder) {
    errors.emplace_back(
        "ConSan final validation could not create a release-transaction semantic decoder");
    return errors;
  }
  const Section *section = replacement.text_sections().front();
  const std::span<const uint8_t> text(reinterpret_cast<const uint8_t *>(section->data()),
                                      section->size());

  const auto atomic_site_for = [&](uint64_t text_offset) -> const ConSanAtomicSite * {
    for (const ConSanKernelInfo &kernel : result.kernels) {
      const auto site =
          std::ranges::find(kernel.atomic_sites, text_offset, &ConSanAtomicSite::text_offset);
      if (site != kernel.atomic_sites.end())
        return &*site;
    }
    for (const ConSanFunctionInfo &function : result.functions) {
      const auto site =
          std::ranges::find(function.atomic_sites, text_offset, &ConSanAtomicSite::text_offset);
      if (site != function.atomic_sites.end())
        return &*site;
    }
    return nullptr;
  };

  for (const ConSanPatchInfo &patch : result.patches) {
    if (!is_release_transaction(patch))
      continue;
    if (!patch.scratch_vgpr || patch.trampoline_offset > text.size() ||
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
      std::unique_ptr<Instruction> instruction;
      try {
        instruction.reset(decoder->decode(words.data(), patch.trampoline_offset + cursor));
      } catch (...) {
        decode_failed = true;
        break;
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
    const bool cdna = arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
    const uint16_t snapshot_address = static_cast<uint16_t>(scratch + (cdna ? 6u : 5u));
    const uint16_t temporary = static_cast<uint16_t>(scratch + 20u);
    const uint16_t version_cas_value = static_cast<uint16_t>(scratch + (cdna ? 4u : 5u));
    const auto version_cas = instrumentation::build_flat_atomic_cmpswap_b32(
        scratch, version_cas_value, version_cas_value, /*return_old_value=*/true, /*scope=*/2,
        arch);
    const uint16_t owner_vgpr = result.resolved_moi_owner_vgpr
                                    ? *result.resolved_moi_owner_vgpr
                                    : static_cast<uint16_t>(scratch + 24u);
    const auto owner_store = instrumentation::build_flat_store_b32(
        scratch, owner_vgpr, arch, offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id));
    const auto wait_store = instrumentation::build_s_wait_flat_store0(arch);
    if (!version_cas || !owner_store || !wait_store) {
      errors.emplace_back(
          "ConSan final validation could not encode release-transaction expectations");
      continue;
    }
    const std::vector<size_t> cas_positions = positions(*version_cas);
    const std::vector<size_t> owner_positions = positions(*owner_store);
    const size_t owner_between =
        cas_positions.size() == 2u
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
    const bool claimed_acquire_release_arch = arch == ROCJITSU_CODE_ARCH_CDNA3 ||
                                              arch == ROCJITSU_CODE_ARCH_CDNA4 ||
                                              is_rdna4_family_arch(arch);
    bool valid = cas_positions.size() == 2u &&
                 (owner_between == 1u || (claimed_acquire_release_arch && owner_between == 6u));

    constexpr std::array<size_t, 11> kTokenPayloadOffsets = {
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) + sizeof(uint32_t),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address) + sizeof(uint32_t),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version),
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reserved),
    };
    const auto version_before = instrumentation::build_flat_load_b32(
        snapshot_address, static_cast<uint16_t>(scratch + 17u), arch,
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version));
    const auto version_after = instrumentation::build_flat_load_b32(
        snapshot_address, temporary, arch,
        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version));
    bool token_loads_valid = version_before && version_after &&
                             !positions(*version_before).empty() &&
                             !positions(*version_after).empty();
    for (size_t offset : kTokenPayloadOffsets) {
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
             cas_positions.front() < store_positions.front() &&
             std::ranges::any_of(owner_positions, [&](size_t owner_position) {
               return store_positions.front() < owner_position &&
                      owner_position < cas_positions.back();
             });
    };
    bool snapshot_stores_valid =
        require_snapshot_store(offsetof(ConSanMoiInlineCausalSnapshot, entry_count),
                               static_cast<uint16_t>(scratch + (cdna ? 5u : 7u)));
    snapshot_stores_valid &= require_snapshot_store(offsetof(ConSanMoiInlineCausalSnapshot, flags),
                                                    static_cast<uint16_t>(scratch + 8u));
    for (uint16_t i = 0; i < 4u; ++i) {
      const size_t entry_offset = offsetof(ConSanMoiInlineCausalSnapshot, entries) +
                                  i * sizeof(ConSanMoiInlineCausalSnapshotEntry);
      snapshot_stores_valid &=
          require_snapshot_store(entry_offset, static_cast<uint16_t>(scratch + 9u + i));
      snapshot_stores_valid &= require_snapshot_store(entry_offset + sizeof(uint32_t),
                                                      static_cast<uint16_t>(scratch + 13u + i));
    }
    valid &= snapshot_stores_valid;
    const size_t wait_store_count = positions(std::array<uint32_t, 1>{*wait_store}).size();
    valid &= wait_store_count >= 2u;

    const ConSanAtomicSite *site = atomic_site_for(patch.anchor_offset);
    const bool compare_exchange = site && consan_atomic_is_compare_exchange(*site);
    if (compare_exchange) {
      if (!site->data_vgpr || !site->dst_vgpr) {
        valid = false;
      } else {
        const auto success = instrumentation::build_v_cmp_eq_u32_vcc(
            vector_source_vgpr(static_cast<uint16_t>(*site->data_vgpr + 1u)), *site->dst_vgpr,
            arch);
        const auto narrow = instrumentation::build_s_and_saveexec_b64(
            *result.resolved_moi_exec_save_sgpr, kVccLo, arch);
        if (!success || !narrow) {
          valid = false;
        } else {
          const std::array<uint32_t, 2> success_sequence = {*success, *narrow};
          const std::vector<size_t> success_positions = positions(success_sequence);
          valid &= success_positions.size() == 1u && !cas_positions.empty() &&
                   success_positions.front() < cas_positions.front();
        }
      }
    }
    if (!valid) {
      errors.emplace_back("ConSan final validation found invalid versioned release transaction "
                          "semantics (cas=" +
                          std::to_string(cas_positions.size()) +
                          " owner_between=" + std::to_string(owner_between) +
                          " token_loads=" + (token_loads_valid ? "yes" : "no") +
                          " snapshot_stores=" + (snapshot_stores_valid ? "yes" : "no") +
                          " waits=" + std::to_string(wait_store_count) + ")");
    }
  }
  return errors;
}

[[nodiscard]] std::vector<std::string>
validate_resource_and_metadata_deltas(const AmdGpuCodeObject &original,
                                      const AmdGpuCodeObject &replacement,
                                      const ConSanResult &result) {
  using KernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;
  std::vector<std::string> errors;
  if (original.text_sections().size() != 1 || replacement.text_sections().size() != 1)
    return errors;
  std::unordered_map<uint64_t, std::string> owner_names;
  for (const AmdGpuKernelInfo &kernel : original.kernels())
    owner_names.emplace(kernel.descriptor_file_offset, kernel.name);
  for (const AmdGpuKernelInfo &kernel : replacement.kernels())
    owner_names.emplace(kernel.descriptor_file_offset, kernel.name);
  std::vector<uint64_t> referenced_owner_offsets;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans)
    referenced_owner_offsets.insert(referenced_owner_offsets.end(),
                                    plan.owner_descriptor_file_offsets.begin(),
                                    plan.owner_descriptor_file_offsets.end());
  for (const ConSanPatchInfo &patch : result.patches)
    referenced_owner_offsets.insert(referenced_owner_offsets.end(),
                                    patch.owner_descriptor_file_offsets.begin(),
                                    patch.owner_descriptor_file_offsets.end());
  const uint64_t text_growth =
      replacement.text_sections().front()->size() >= original.text_sections().front()->size()
          ? replacement.text_sections().front()->size() - original.text_sections().front()->size()
          : 0;
  std::unordered_map<uint64_t, size_t> relocation_frequency;
  for (uint64_t owner : referenced_owner_offsets) {
    if (owner_names.contains(owner))
      continue;
    for (const AmdGpuKernelInfo &kernel : original.kernels()) {
      if (owner >= kernel.descriptor_file_offset &&
          owner - kernel.descriptor_file_offset <= text_growth) {
        ++relocation_frequency[owner - kernel.descriptor_file_offset];
      }
    }
  }
  for (uint64_t owner : referenced_owner_offsets) {
    if (owner_names.contains(owner))
      continue;
    const AmdGpuKernelInfo *best = nullptr;
    size_t best_frequency = 0;
    bool ambiguous = false;
    for (const AmdGpuKernelInfo &kernel : original.kernels()) {
      if (owner < kernel.descriptor_file_offset ||
          owner - kernel.descriptor_file_offset > text_growth)
        continue;
      const size_t frequency = relocation_frequency[owner - kernel.descriptor_file_offset];
      if (frequency > best_frequency) {
        best = &kernel;
        best_frequency = frequency;
        ambiguous = false;
      } else if (frequency == best_frequency && best && best->name != kernel.name) {
        ambiguous = true;
      }
    }
    if (best && !ambiguous)
      owner_names.emplace(owner, best->name);
  }
  std::unordered_map<std::string, DescriptorRequirement> requirements;
  const auto apply_to_owner = [&](uint64_t descriptor_offset, const auto &apply) {
    const auto owner = owner_names.find(descriptor_offset);
    if (owner == owner_names.end()) {
      bool saw_relocation_candidate = false;
      for (const AmdGpuKernelInfo &kernel : original.kernels()) {
        if (descriptor_offset >= kernel.descriptor_file_offset &&
            descriptor_offset - kernel.descriptor_file_offset <= text_growth) {
          saw_relocation_candidate = true;
        }
      }
      if (!saw_relocation_candidate) {
        errors.emplace_back("ConSan final validation found unknown descriptor owner offset " +
                            std::to_string(descriptor_offset));
      }
      return;
    }
    apply(requirements[owner->second]);
  };
  const auto plan_has_emitted_patch = [&](const ConSanCandidateResourcePlan &plan) {
    return std::ranges::any_of(result.patches, [&](const ConSanPatchInfo &patch) {
      return patch.phase == ConSanPatchPhase::Instrumentation &&
             patch.anchor_offset == plan.text_offset && patch.scratch_vgpr.has_value();
    });
  };
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (!plan_has_emitted_patch(plan))
      continue;
    for (uint64_t owner : plan.owner_descriptor_file_offsets) {
      apply_to_owner(owner, [&](DescriptorRequirement &requirement) {
        requirement.allow_resource_delta = true;
        requirement.required_vgpr_count =
            std::max(requirement.required_vgpr_count, plan.required_vgpr_count);
      });
    }
  }
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.dynamic_private_segment_addend != 0) {
      if (patch.required_private_segment_size < patch.dynamic_private_segment_addend) {
        errors.emplace_back(
            "ConSan final validation found a dynamic private addend larger than its absolute "
            "private requirement");
      }
      if (patch.owner_descriptor_file_offsets.empty()) {
        errors.emplace_back(
            "ConSan final validation found an unowned dynamic private-segment requirement");
      }
      for (uint64_t owner : patch.owner_descriptor_file_offsets) {
        const auto owner_name = owner_names.find(owner);
        const auto owner_kernel =
            owner_name == owner_names.end()
                ? original.kernels().end()
                : std::ranges::find(original.kernels(), owner_name->second,
                                    &AmdGpuKernelInfo::name);
        if (owner_kernel == original.kernels().end() ||
            !owner_kernel->uses_dynamic_stack.value_or(false)) {
          errors.emplace_back(
              "ConSan final validation found a dynamic private-segment requirement on a "
              "non-dynamic owner");
        }
      }
    }
    for (uint64_t owner : patch.owner_descriptor_file_offsets) {
      apply_to_owner(owner, [&](DescriptorRequirement &requirement) {
        requirement.allow_resource_delta = true;
        requirement.required_private_bytes =
            std::max(requirement.required_private_bytes, patch.required_private_segment_size);
        requirement.required_group_bytes =
            std::max(requirement.required_group_bytes, patch.required_group_segment_size);
        requirement.required_sgpr_count =
            std::max({requirement.required_sgpr_count, patch.required_sgpr_count,
                      patch.indirect_required_sgpr_count});
        requirement.allow_entry_delta |=
            patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue ||
            patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
        if (patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue) {
          const auto owner_vgpr = patch.persistent_owner_vgpr ? patch.persistent_owner_vgpr
                                                              : result.resolved_moi_owner_vgpr;
          const auto epoch_vgpr = patch.persistent_epoch_vgpr ? patch.persistent_epoch_vgpr
                                                              : result.resolved_moi_epoch_vgpr;
          const auto workgroup_key_vgpr = patch.persistent_workgroup_key_vgpr
                                              ? patch.persistent_workgroup_key_vgpr
                                              : result.resolved_moi_workgroup_key_vgpr;
          if (owner_vgpr) {
            requirement.required_vgpr_count = std::max<uint16_t>(
                requirement.required_vgpr_count, static_cast<uint16_t>(*owner_vgpr + 1u));
          }
          if (epoch_vgpr) {
            requirement.required_vgpr_count = std::max<uint16_t>(
                requirement.required_vgpr_count, static_cast<uint16_t>(*epoch_vgpr + 1u));
          }
          if (workgroup_key_vgpr) {
            requirement.required_vgpr_count = std::max<uint16_t>(
                requirement.required_vgpr_count, static_cast<uint16_t>(*workgroup_key_vgpr + 1u));
          }
          if (result.resolved_moi_persistent_owner_sgpr) {
            requirement.required_sgpr_count = std::max<uint16_t>(
                requirement.required_sgpr_count,
                static_cast<uint16_t>(*result.resolved_moi_persistent_owner_sgpr + 1u));
          }
          if (result.resolved_moi_persistent_epoch_sgpr) {
            requirement.required_sgpr_count = std::max<uint16_t>(
                requirement.required_sgpr_count,
                static_cast<uint16_t>(*result.resolved_moi_persistent_epoch_sgpr + 1u));
          }
          if (result.resolved_moi_persistent_workgroup_key_sgpr) {
            requirement.required_sgpr_count = std::max<uint16_t>(
                requirement.required_sgpr_count,
                static_cast<uint16_t>(*result.resolved_moi_persistent_workgroup_key_sgpr + 1u));
          }
        }
        if (patch.dispatch_id_capture_sgpr) {
          requirement.has_dispatch_id_capture = true;
          requirement.dispatch_id_preload_inserted = patch.dispatch_id_preload_inserted;
          requirement.dispatch_id_original_user_sgpr_count =
              patch.dispatch_id_original_user_sgpr_count;
          requirement.dispatch_id_expanded_user_sgpr_count =
              patch.dispatch_id_expanded_user_sgpr_count;
          requirement.dispatch_id_original_kernarg_preload_length =
              patch.dispatch_id_original_kernarg_preload_length;
          requirement.dispatch_id_replacement_kernarg_preload_length =
              patch.dispatch_id_replacement_kernarg_preload_length;
          requirement.required_sgpr_count =
              std::max(requirement.required_sgpr_count, patch.dispatch_id_required_sgpr_count);
        }
      });
    }
  }
  if (!errors.empty())
    return errors;

  uint16_t unscoped_required_vgprs = 0;
  uint32_t unscoped_required_private_bytes = 0;
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty() && plan_has_emitted_patch(plan))
      unscoped_required_vgprs = std::max(unscoped_required_vgprs, plan.required_vgpr_count);
  }
  for (const ConSanPatchInfo &patch : result.patches) {
    if (patch.owner_descriptor_file_offsets.empty()) {
      unscoped_required_private_bytes =
          std::max(unscoped_required_private_bytes, patch.required_private_segment_size);
    }
  }

  const bool text_relocated =
      original.text_sections().front()->size() != replacement.text_sections().front()->size();
  const rj_code_arch_t arch = arch_for_target(replacement.target_id());
  for (const AmdGpuKernelInfo &original_kernel : original.kernels()) {
    const auto replacement_it =
        std::ranges::find_if(replacement.kernels(), [&](const AmdGpuKernelInfo &kernel) {
          return kernel.name == original_kernel.name;
        });
    if (replacement_it == replacement.kernels().end())
      continue;
    KernelDescriptor original_descriptor{};
    KernelDescriptor replacement_descriptor{};
    std::memcpy(&original_descriptor,
                original.image_data() + original_kernel.descriptor_file_offset,
                sizeof(original_descriptor));
    std::memcpy(&replacement_descriptor,
                replacement.image_data() + replacement_it->descriptor_file_offset,
                sizeof(replacement_descriptor));
    DescriptorRequirement requirement = requirements[original_kernel.name];
    const bool descriptor_resources_changed =
        replacement_descriptor.compute_pgm_rsrc1 != original_descriptor.compute_pgm_rsrc1 ||
        replacement_descriptor.compute_pgm_rsrc2 != original_descriptor.compute_pgm_rsrc2 ||
        replacement_descriptor.private_segment_fixed_size !=
            original_descriptor.private_segment_fixed_size ||
        replacement_descriptor.group_segment_fixed_size !=
            original_descriptor.group_segment_fixed_size;
    if (descriptor_resources_changed) {
      requirement.required_vgpr_count =
          std::max(requirement.required_vgpr_count, unscoped_required_vgprs);
      requirement.required_private_bytes =
          std::max(requirement.required_private_bytes, unscoped_required_private_bytes);
    }
    const bool has_unscoped_resource_growth =
        std::ranges::any_of(
            result.resource_plans,
            [&](const ConSanCandidateResourcePlan &plan) {
              return plan.owner_descriptor_file_offsets.empty() && plan_has_emitted_patch(plan) &&
                     (plan.source == ConSanRegisterAllocationSource::DescriptorGrowth ||
                      plan.source == ConSanRegisterAllocationSource::SpillRequired);
            }) ||
        std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
          return patch.owner_descriptor_file_offsets.empty() && patch.scratch_vgpr.has_value();
        });
    KernelDescriptor normalized = replacement_descriptor;
    if (requirement.has_dispatch_id_capture) {
      const uint32_t original_user_count = AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc2,
                                                           kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
      const uint32_t replacement_user_count = AMDHSA_BITS_GET(
          replacement_descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
      const bool original_dispatch =
          AMDHSA_BITS_GET(original_descriptor.kernel_code_properties,
                          kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) != 0;
      const bool replacement_dispatch =
          AMDHSA_BITS_GET(replacement_descriptor.kernel_code_properties,
                          kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) != 0;
      const bool expected_original_dispatch = !requirement.dispatch_id_preload_inserted;
      if (original_user_count != requirement.dispatch_id_original_user_sgpr_count ||
          replacement_user_count != requirement.dispatch_id_expanded_user_sgpr_count ||
          original_dispatch != expected_original_dispatch || !replacement_dispatch) {
        errors.emplace_back(
            "ConSan final validation found an invalid dispatch-ID preload descriptor delta for "
            "kernel '" +
            original_kernel.name + "'");
      }
      const uint32_t original_preload_length =
          AMDHSA_BITS_GET(original_descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH);
      const uint32_t replacement_preload_length =
          AMDHSA_BITS_GET(replacement_descriptor.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH);
      if (original_preload_length != requirement.dispatch_id_original_kernarg_preload_length ||
          replacement_preload_length !=
              requirement.dispatch_id_replacement_kernarg_preload_length) {
        errors.emplace_back(
            "ConSan final validation found an invalid dispatch-ID kernarg-preload delta for "
            "kernel '" +
            original_kernel.name + "'");
      }

      uint32_t normalized_rsrc2 = replacement_descriptor.compute_pgm_rsrc2;
      AMDHSA_BITS_SET(normalized_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, original_user_count);
      if (requirement.required_private_bytes != 0) {
        AMDHSA_BITS_SET(normalized_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT,
                        AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc2,
                                        kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT));
      }
      if (normalized_rsrc2 != original_descriptor.compute_pgm_rsrc2) {
        errors.emplace_back(
            "ConSan final validation found an extra RSRC2 delta beside dispatch-ID insertion "
            "for kernel '" +
            original_kernel.name + "'");
      }

      uint32_t normalized_rsrc1 = replacement_descriptor.compute_pgm_rsrc1;
      AMDHSA_BITS_SET(normalized_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT,
                      AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc1,
                                      kd::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT));
      AMDHSA_BITS_SET(normalized_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                      AMDHSA_BITS_GET(original_descriptor.compute_pgm_rsrc1,
                                      kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT));
      if (normalized_rsrc1 != original_descriptor.compute_pgm_rsrc1) {
        errors.emplace_back(
            "ConSan final validation found an extra RSRC1 delta beside register allocation for "
            "kernel '" +
            original_kernel.name + "'");
      }

      AMDHSA_BITS_SET(normalized.kernel_code_properties,
                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID,
                      AMDHSA_BITS_GET(original_descriptor.kernel_code_properties,
                                      kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID));
      if (requirement.dispatch_id_original_kernarg_preload_length != 0) {
        AMDHSA_BITS_SET(normalized.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH,
                        requirement.dispatch_id_original_kernarg_preload_length);
      }
    }
    if (text_relocated || requirement.allow_entry_delta)
      normalized.kernel_code_entry_byte_offset = original_descriptor.kernel_code_entry_byte_offset;
    if (requirement.allow_resource_delta || has_unscoped_resource_growth) {
      normalized.compute_pgm_rsrc1 = original_descriptor.compute_pgm_rsrc1;
      normalized.compute_pgm_rsrc2 = original_descriptor.compute_pgm_rsrc2;
      normalized.private_segment_fixed_size = original_descriptor.private_segment_fixed_size;
      normalized.group_segment_fixed_size = original_descriptor.group_segment_fixed_size;
    }
    if (std::memcmp(&normalized, &original_descriptor, sizeof(normalized)) != 0) {
      errors.emplace_back(
          "ConSan final validation found an unplanned descriptor delta for kernel '" +
          original_kernel.name + "' (original offset " +
          std::to_string(original_kernel.descriptor_file_offset) + ", replacement offset " +
          std::to_string(replacement_it->descriptor_file_offset) + ")");
      continue;
    }
    if (requirement.required_vgpr_count != 0) {
      const std::optional<uint16_t> actual_vgprs = descriptor_vgpr_allocation_count(
          result.elf_bytes, replacement_it->descriptor_file_offset, arch);
      if (!actual_vgprs || *actual_vgprs < requirement.required_vgpr_count) {
        errors.emplace_back(
            "ConSan final validation found insufficient descriptor VGPRs for kernel '" +
            original_kernel.name + "'");
      }
    }
    const uint16_t replacement_sgpr_count =
        descriptor_sgpr_allocation_count(replacement_descriptor);
    if (requirement.required_sgpr_count != 0 &&
        replacement_sgpr_count < requirement.required_sgpr_count) {
      errors.emplace_back(
          "ConSan final validation found insufficient descriptor SGPRs for kernel '" +
          original_kernel.name + "': required=" + std::to_string(requirement.required_sgpr_count) +
          " allocated=" + std::to_string(replacement_sgpr_count));
    }
    if (requirement.required_group_bytes != 0 &&
        replacement_descriptor.group_segment_fixed_size != requirement.required_group_bytes) {
      errors.emplace_back(
          "ConSan final validation found inconsistent fixed-LDS reservation for kernel '" +
          original_kernel.name + "'");
    }
    if (requirement.required_private_bytes == 0)
      continue;
    std::vector<uint8_t> descriptor_probe(result.elf_bytes);
    const SpillDescriptorUpdate descriptor_update = update_kernel_descriptor_for_spills(
        descriptor_probe, replacement_it->descriptor_file_offset,
        requirement.required_private_bytes);
    if (descriptor_update != SpillDescriptorUpdate::Unchanged) {
      errors.emplace_back(
          "ConSan final validation found insufficient spill descriptor state for kernel '" +
          original_kernel.name + "' (required " +
          std::to_string(requirement.required_private_bytes) + ", found " +
          std::to_string(replacement_descriptor.private_segment_fixed_size) + ")");
    }
  }
  return errors;
}

[[nodiscard]] std::vector<std::string>
validate_final_consan_elf(std::span<const uint8_t> original_bytes, const ConSanResult &result) {
  std::vector<std::string> errors;
  AmdGpuCodeObject original(original_bytes.data(), original_bytes.size());
  AmdGpuCodeObject replacement(result.elf_bytes.data(), result.elf_bytes.size());
  if (!original.is_valid() || !replacement.is_valid()) {
    errors.emplace_back("ConSan final validation could not parse original and replacement ELFs");
    return errors;
  }

  const bool redirects_entry =
      std::ranges::any_of(result.patches, [](const ConSanPatchInfo &patch) {
        return patch.kind == ConSanPatchKind::KernelEntryMoiOwnerEpochPrologue ||
               patch.kind == ConSanPatchKind::KernelEntryMoiPrivateEpochPrologue;
      });
  errors = validate_consan_input_layout(replacement, redirects_entry);
  for (std::string &error : errors)
    error = "ConSan final validation: " + error;

  if (replacement.target_id() != original.target_id()) {
    errors.emplace_back("ConSan final validation changed the AMDGPU target identity");
    // Every remaining proof decodes or synthesizes instructions for the
    // replacement target. A corrupted target ID is already a conclusive
    // validation failure and must not escape as an instruction-builder
    // exception while validating an untrusted replacement image.
    return errors;
  }
  if (sorted_entity_names<AmdGpuKernelInfo>(replacement.kernels()) !=
      sorted_entity_names<AmdGpuKernelInfo>(original.kernels())) {
    errors.emplace_back("ConSan final validation changed the kernel symbol identity set");
  }
  if (sorted_entity_names<AmdGpuFunctionInfo>(replacement.functions()) !=
      sorted_entity_names<AmdGpuFunctionInfo>(original.functions())) {
    errors.emplace_back("ConSan final validation changed the function symbol identity set");
  }

  const std::vector<TextSectionIdentity> original_text = text_section_identities(original);
  const std::vector<TextSectionIdentity> replacement_text = text_section_identities(replacement);
  if (original_text.size() != replacement_text.size()) {
    errors.emplace_back("ConSan final validation changed the executable section count");
  } else {
    for (size_t i = 0; i < original_text.size(); ++i) {
      if (original_text[i].name != replacement_text[i].name ||
          original_text[i].virtual_address != replacement_text[i].virtual_address) {
        errors.emplace_back("ConSan final validation changed executable section identity");
        break;
      }
      if (replacement_text[i].size < original_text[i].size) {
        errors.emplace_back("ConSan final validation unexpectedly shrank executable text");
        break;
      }
    }
  }
  if (result.elf_bytes.size() == original_bytes.size() &&
      std::ranges::equal(result.elf_bytes, original_bytes)) {
    errors.emplace_back("ConSan final validation found no byte change in a modified result");
  }
  if (result.patches.empty())
    errors.emplace_back("ConSan final validation found no patch inventory for a modified result");
  std::vector<std::string> accounting_errors =
      validate_patch_byte_accounting(original, replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(accounting_errors.begin()),
                std::make_move_iterator(accounting_errors.end()));
  std::vector<std::string> decoding_errors = validate_patch_decoding(replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(decoding_errors.begin()),
                std::make_move_iterator(decoding_errors.end()));
  std::vector<std::string> indirect_island_errors =
      validate_sc_indirect_island_semantics(replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(indirect_island_errors.begin()),
                std::make_move_iterator(indirect_island_errors.end()));
  std::vector<std::string> branch_only_errors =
      validate_sc_branch_only_continuation_semantics(replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(branch_only_errors.begin()),
                std::make_move_iterator(branch_only_errors.end()));
  std::vector<std::string> dense_call_errors =
      validate_sc_dense_call_semantics(original, replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(dense_call_errors.begin()),
                std::make_move_iterator(dense_call_errors.end()));
  std::vector<std::string> relay_donor_errors =
      validate_sc_branch_relay_donor_semantics(original, replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(relay_donor_errors.begin()),
                std::make_move_iterator(relay_donor_errors.end()));
  std::vector<std::string> relay_reservoir_errors =
      validate_sc_relay_reservoir_semantics(original, replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(relay_reservoir_errors.begin()),
                std::make_move_iterator(relay_reservoir_errors.end()));
  std::vector<std::string> mutation_errors =
      validate_mutation_semantics(original, replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(mutation_errors.begin()),
                std::make_move_iterator(mutation_errors.end()));
  std::vector<std::string> perturbation_errors =
      validate_sc_perturbation_semantics(original_bytes, original, replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(perturbation_errors.begin()),
                std::make_move_iterator(perturbation_errors.end()));
  std::vector<std::string> resource_errors =
      validate_resource_and_metadata_deltas(original, replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(resource_errors.begin()),
                std::make_move_iterator(resource_errors.end()));
  std::vector<std::string> dispatch_errors =
      validate_dispatch_id_prologue_semantics(original, replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(dispatch_errors.begin()),
                std::make_move_iterator(dispatch_errors.end()));
  std::vector<std::string> exact_shadow_errors =
      validate_inline_exact_shadow_semantics(replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(exact_shadow_errors.begin()),
                std::make_move_iterator(exact_shadow_errors.end()));
  std::vector<std::string> release_transaction_errors =
      validate_inline_release_transaction_semantics(replacement, result);
  errors.insert(errors.end(), std::make_move_iterator(release_transaction_errors.begin()),
                std::make_move_iterator(release_transaction_errors.end()));
  return errors;
}

[[nodiscard]] ConSanResult finalize_consan_result(ConSanResult result,
                                                  std::span<const uint8_t> original_bytes) {
  if (has_rejected_kernel(result)) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.modified = false;
    result.elf_bytes.clear();
    result.patches.clear();
    result.applied_fault_logical_identity.reset();
    return result;
  }
  if (!result.errors.empty()) {
    result.outcome = ConSanTransformOutcome::Invalid;
    result.modified = false;
    result.elf_bytes.clear();
    result.patches.clear();
    result.applied_fault_logical_identity.reset();
    return result;
  }
  if (result.modified) {
    if (result.elf_bytes.empty()) {
      result.errors.emplace_back("ConSan modified result has no replacement ELF bytes");
      result.outcome = ConSanTransformOutcome::Invalid;
      result.modified = false;
      result.applied_fault_logical_identity.reset();
    } else {
      std::vector<std::string> validation_errors =
          validate_final_consan_elf(original_bytes, result);
      result.errors.insert(result.errors.end(), std::make_move_iterator(validation_errors.begin()),
                           std::make_move_iterator(validation_errors.end()));
      if (result.errors.empty()) {
        result.outcome = ConSanTransformOutcome::ModifiedValid;
        result.final_validation_passed = true;
      } else {
        result.outcome = ConSanTransformOutcome::Invalid;
        result.modified = false;
        result.applied_fault_logical_identity.reset();
        result.elf_bytes.clear();
        result.patches.clear();
      }
    }
    return result;
  }
  if (result.outcome == ConSanTransformOutcome::Unsupported ||
      result.resource_plan_summary.unsupported_plans != 0 || has_unsupported_preflight(result)) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.applied_fault_logical_identity.reset();
  } else {
    result.outcome = ConSanTransformOutcome::Unchanged;
  }
  return result;
}

} // namespace

std::vector<std::string> validate_consan_modified_elf(std::span<const uint8_t> original_bytes,
                                                      const ConSanResult &modified_result) {
  return validate_final_consan_elf(original_bytes, modified_result);
}

ConSanInstallAction consan_install_action(const ConSanResult &result, bool fail_closed) {
  switch (result.outcome) {
  case ConSanTransformOutcome::ModifiedValid:
    if (result.modified && result.final_validation_passed && !result.elf_bytes.empty())
      return ConSanInstallAction::LoadReplacement;
    return ConSanInstallAction::Reject;
  case ConSanTransformOutcome::Unchanged:
    return ConSanInstallAction::LoadOriginal;
  case ConSanTransformOutcome::Unsupported:
  case ConSanTransformOutcome::Invalid:
    return fail_closed ? ConSanInstallAction::Reject : ConSanInstallAction::LoadOriginal;
  }
  return ConSanInstallAction::Reject;
}
