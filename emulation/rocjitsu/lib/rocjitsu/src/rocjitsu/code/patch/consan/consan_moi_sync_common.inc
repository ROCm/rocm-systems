struct AtomicRecordCandidate {
  std::string container_name;
  ConSanAtomicSite site;
  std::optional<uint64_t> kernel_descriptor_file_offset;
  ConSanMoiAtomicEventKind event_kind = ConSanMoiAtomicEventKind::Release;
  bool uses_gfx1250_cluster_workgroup_id = false;
};

struct SampledAtomicSemantics {
  ConSanMoiSampledSyncRole role = ConSanMoiSampledSyncRole::None;
  ConSanMoiSampledSyncScope scope = ConSanMoiSampledSyncScope::None;
  ConSanMoiSampledSyncOutcome outcome = ConSanMoiSampledSyncOutcome::NotApplicable;
  uint32_t byte_count = 0;
  uint32_t descriptor = 0;
  std::optional<uint32_t> cas_failure_descriptor;
};

[[nodiscard]] std::optional<SampledAtomicSemantics>
sampled_atomic_semantics_for_site(const ConSanResult &result, std::string_view container_name,
                                  bool in_kernel, const ConSanAtomicSite &site,
                                  std::string &unsupported_reason) {
  const auto event = std::ranges::find_if(result.sync_events, [&](const ConSanSyncEvent &item) {
    return item.kind == ConSanSyncEventKind::Atomic && item.container_name == container_name &&
           item.in_kernel == in_kernel && item.text_offset == site.text_offset;
  });
  if (event == result.sync_events.end()) {
    unsupported_reason = "missing-shared-sync-event";
    return std::nullopt;
  }
  const ConSanSyncSequence *sequence = find_consan_sync_sequence_for_event(result, event->identity);
  if (sequence == nullptr ||
      !consan_sync_confidence_meets(sequence->confidence, ConSanSemanticConfidence::Conservative) ||
      !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                    ConSanSemanticConfidence::Conservative)) {
    unsupported_reason = "unqualified-shared-sync-sequence";
    return std::nullopt;
  }

  SampledAtomicSemantics semantics;
  switch (sequence->memory_role) {
  case ConSanSyncMemoryRole::Release:
    semantics.role = ConSanMoiSampledSyncRole::RmwRelease;
    break;
  case ConSanSyncMemoryRole::Acquire:
    semantics.role = ConSanMoiSampledSyncRole::RmwAcquire;
    break;
  case ConSanSyncMemoryRole::AcquireRelease:
    semantics.role = ConSanMoiSampledSyncRole::RmwAcquireRelease;
    break;
  case ConSanSyncMemoryRole::Unknown:
  case ConSanSyncMemoryRole::None:
  case ConSanSyncMemoryRole::SequentiallyConsistent:
    unsupported_reason = "unsupported-qualified-memory-role";
    return std::nullopt;
  }
  // LDS visibility is inherently workgroup-scoped. gfx1250 DS encodings do
  // not carry the raw scope field used by FLAT/GLOBAL atomics, so derive that
  // semantic only after the shared sequence analysis has qualified the DS
  // instruction as an ordering event.
  const std::optional<uint32_t> semantic_scope =
      sequence->raw_scope ? sequence->raw_scope
                          : (result.arch_name == "gfx1250" && site.mnemonic.starts_with("ds_")
                                 ? std::optional<uint32_t>(1u)
                                 : std::nullopt);
  if (!semantic_scope) {
    unsupported_reason = "missing-qualified-scope";
    return std::nullopt;
  }
  switch (*semantic_scope) {
  case 0:
    semantics.scope = ConSanMoiSampledSyncScope::Wavefront;
    break;
  case 1:
    semantics.scope = ConSanMoiSampledSyncScope::Workgroup;
    break;
  case 2:
    semantics.scope = ConSanMoiSampledSyncScope::Agent;
    break;
  case 3:
    semantics.scope = ConSanMoiSampledSyncScope::System;
    break;
  default:
    unsupported_reason = "unsupported-qualified-scope";
    return std::nullopt;
  }
  if (sequence->width_bits == 0 || sequence->width_bits % 8u != 0 ||
      sequence->width_bits / 8u > std::numeric_limits<uint32_t>::max()) {
    unsupported_reason = "unsupported-qualified-byte-range";
    return std::nullopt;
  }
  semantics.byte_count = sequence->width_bits / 8u;
  switch (sequence->rmw_outcome) {
  case ConSanSyncRmwOutcome::NoReturn:
    semantics.outcome = ConSanMoiSampledSyncOutcome::RmwNoReturn;
    break;
  case ConSanSyncRmwOutcome::ReturnsOldValue:
    semantics.outcome = ConSanMoiSampledSyncOutcome::RmwReturnsOld;
    break;
  case ConSanSyncRmwOutcome::CompareExchange:
    if (!site.returns_old_value.value_or(false) || !site.data_vgpr || !site.dst_vgpr) {
      unsupported_reason = "compare-exchange-dynamic-outcome-unavailable";
      return std::nullopt;
    }
    semantics.outcome = ConSanMoiSampledSyncOutcome::CasSuccess;
    break;
  case ConSanSyncRmwOutcome::NotApplicable:
  case ConSanSyncRmwOutcome::Unknown:
    unsupported_reason = "unsupported-qualified-rmw-outcome";
    return std::nullopt;
  }
  const ConSanMoiSampledSyncEncodeResult encoded = encode_consan_moi_sampled_sync_metadata({
      .address = 1,
      .byte_count = semantics.byte_count,
      .kind = ConSanMoiSampledSyncKind::Atomic,
      .role = semantics.role,
      .scope = semantics.scope,
      .outcome = semantics.outcome,
  });
  if (encoded.classification != ConSanMoiSampledSyncClassification::Valid) {
    unsupported_reason = "sampled-sync-abi-rejected-qualified-sequence";
    return std::nullopt;
  }
  semantics.descriptor = encoded.packed.descriptor;
  if (sequence->rmw_outcome == ConSanSyncRmwOutcome::CompareExchange) {
    ConSanMoiSampledSyncMetadata failure_metadata{
        .address = 1,
        .byte_count = semantics.byte_count,
        .kind = ConSanMoiSampledSyncKind::Atomic,
        .role = semantics.role,
        .scope = semantics.scope,
        .outcome = ConSanMoiSampledSyncOutcome::CasFailure,
    };
    const ConSanMoiSampledSyncEncodeResult failure =
        encode_consan_moi_sampled_sync_metadata(failure_metadata);
    if (failure.classification != ConSanMoiSampledSyncClassification::Valid) {
      unsupported_reason = "sampled-sync-abi-rejected-cas-failure";
      return std::nullopt;
    }
    semantics.cas_failure_descriptor = failure.packed.descriptor;
  }
  return semantics;
}

struct FenceRecordCandidate {
  ConSanMoiFenceCandidate fence;
  // Fence emission consumes the common address-plan input shape. Ordinary
  // buffer communication is normalized into that shape without reclassifying
  // the source event as an atomic.
  ConSanAtomicSite communication_site;
  std::optional<uint64_t> kernel_descriptor_file_offset;
  uint64_t container_entry_text_offset = 0;
  uint64_t text_file_offset = 0;
};

[[nodiscard]] ConSanAtomicSite
normalize_ordinary_fence_communication(const ConSanOrdinaryMemorySite &site) {
  ConSanAtomicSite normalized;
  normalized.text_offset = site.text_offset;
  normalized.file_offset = site.file_offset;
  normalized.size = site.size;
  normalized.width_bits = site.width_bits;
  normalized.dst_vgpr = site.destination_vgpr;
  normalized.addr_vgpr = site.address_vgpr;
  normalized.data_vgpr = site.operation == ConSanOrdinaryMemoryOperation::Load
                             ? site.destination_vgpr
                             : site.value_vgpr;
  normalized.saddr_sgpr = site.address_sgpr;
  normalized.raw_vaddr = site.raw_vaddr;
  normalized.raw_vdata = site.operation == ConSanOrdinaryMemoryOperation::Load
                             ? std::optional<uint32_t>(site.destination_vgpr)
                             : std::optional<uint32_t>(site.value_vgpr);
  normalized.raw_rsrc = site.raw_rsrc;
  normalized.raw_soffset = site.raw_soffset;
  normalized.raw_offen = site.raw_offen;
  normalized.raw_idxen = site.raw_idxen;
  normalized.raw_saddr = site.raw_saddr;
  normalized.raw_ioffset = site.raw_ioffset;
  normalized.raw_scope = site.raw_scope;
  normalized.raw_th = site.raw_th;
  normalized.returns_old_value = false;
  normalized.mnemonic = site.mnemonic;
  return normalized;
}

void append_fence_record_candidates(const ConSanResult &result,
                                    std::vector<FenceRecordCandidate> &candidates) {
  for (const ConSanMoiFenceCandidate &fence : result.moi_fence_candidates) {
    if (!fence.eligible)
      continue;
    const auto event = std::ranges::find(result.sync_events, fence.communication_event_identity,
                                         &ConSanSyncEvent::identity);
    if (event == result.sync_events.end() ||
        (event->kind != ConSanSyncEventKind::Atomic &&
         event->kind != ConSanSyncEventKind::OrdinaryMemory) ||
        event->container_name != fence.container_name || event->in_kernel != fence.in_kernel ||
        event->execution_owners.empty())
      continue;
    const auto append_from = [&](const auto &container,
                                 std::optional<uint64_t> descriptor_file_offset) {
      if (event->kind == ConSanSyncEventKind::Atomic) {
        const auto site = std::ranges::find(container.atomic_sites, event->text_offset,
                                            &ConSanAtomicSite::text_offset);
        if (site == container.atomic_sites.end() ||
            std::ranges::count(container.atomic_sites, event->text_offset,
                               &ConSanAtomicSite::text_offset) != 1)
          return;
        candidates.push_back({fence, *site, descriptor_file_offset, container.entry_text_offset,
                              container.text_file_offset});
        return;
      }
      const auto site = std::ranges::find(container.ordinary_memory_sites, event->text_offset,
                                          &ConSanOrdinaryMemorySite::text_offset);
      if (site == container.ordinary_memory_sites.end() ||
          std::ranges::count(container.ordinary_memory_sites, event->text_offset,
                             &ConSanOrdinaryMemorySite::text_offset) != 1 ||
          (site->support_reason != ConSanOrdinaryMemorySupportReason::Supported &&
           site->support_reason != ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly))
        return;
      candidates.push_back({fence, normalize_ordinary_fence_communication(*site),
                            descriptor_file_offset, container.entry_text_offset,
                            container.text_file_offset});
    };
    if (fence.in_kernel) {
      const auto kernel =
          std::ranges::find(result.kernels, fence.container_name, &ConSanKernelInfo::name);
      if (kernel != result.kernels.end() && !is_rocclr_runtime_kernel_name(kernel->name))
        append_from(*kernel, kernel->descriptor_file_offset);
    } else {
      const auto function =
          std::ranges::find(result.functions, fence.container_name, &ConSanFunctionInfo::name);
      if (function != result.functions.end())
        append_from(*function, std::nullopt);
    }
  }
}

[[nodiscard]] uint16_t inline_atomic_scratch_count(const ConSanAtomicSite &site,
                                                   rj_code_arch_t arch) {
  if (site.mnemonic.starts_with("buffer_") || site.mnemonic.starts_with("ds_"))
    return 5u;
  const bool returned_value_aliases_address =
      site.returns_old_value.value_or(false) && site.addr_vgpr && site.dst_vgpr &&
      *site.dst_vgpr >= *site.addr_vgpr && *site.dst_vgpr < *site.addr_vgpr + 2u;
  return returned_value_aliases_address ||
                 (site.mnemonic.starts_with("global_") &&
                  (!site.raw_saddr || !is_vector_flat_no_saddr(*site.raw_saddr, arch) ||
                   site.raw_ioffset.value_or(0) != 0))
             ? 5u
             : 3u;
}

[[nodiscard]] uint16_t atomic_record_scratch_count(const ConSanAtomicSite &site,
                                                   rj_code_arch_t arch) {
  // Dynamic Record/Replay publication needs seven VGPRs for the slot,
  // address, value, and preserved guest atomic address. Compare-exchange
  // temporarily retains both halves of its success mask before the common
  // emitter reuses its value temporary.
  return std::max<uint16_t>(inline_atomic_scratch_count(site, arch), 7u);
}

[[nodiscard]] uint16_t inline_shadow_atomic_scratch_count(const ConSanAtomicSite &,
                                                          ConSanMoiAtomicEventKind event_kind,
                                                          rj_code_arch_t arch) {
  (void)event_kind;
  // Acquire import now retains the release version across the guest, captures
  // its causal snapshot, and transactionally publishes up to five full ABI-v6
  // tokens. Release-capable events use the same window for frontier capture.
  // CDNA4 additionally retains token-transaction producer state and scalar
  // persistent owner/epoch materialization ahead of the final aligned address
  // pair. RDNA4 keeps its established 24-register transaction layout.
  return arch == ROCJITSU_CODE_ARCH_CDNA4 ? 28u : 24u;
}

[[nodiscard]] bool is_first_light_atomic_record_candidate(const ConSanAtomicSite &site,
                                                          bool allow_gfx1250_lds = false) {
  const bool is_gfx1250_lds = allow_gfx1250_lds && site.mnemonic.starts_with("ds_");
  if (!site.mnemonic.starts_with("flat_atomic") && !site.mnemonic.starts_with("global_atomic") &&
      !is_gfx1250_lds)
    return false;
  if (!site.addr_vgpr || !site.data_vgpr || *site.addr_vgpr >= 255)
    return false;
  // Wave-scope atomics cannot communicate between the distinct owners that
  // ConSan's MOI engines model. Keep them in the discovered inventory, but do
  // not count them as supported lowering candidates.
  if (!site.raw_scope || *site.raw_scope < 1u || *site.raw_scope > 3u)
    return false;
  const bool cas = site.mnemonic.find("cmpswap") != std::string::npos ||
                   site.mnemonic.find("cmpxchg") != std::string::npos;
  if (cas && !site.returns_old_value.value_or(false))
    return false;
  return true;
}

[[nodiscard]] ConSanAtomicSite normalize_atomic_record_site(const ConSanAtomicSite &site,
                                                            bool allow_gfx1250_lds) {
  ConSanAtomicSite normalized = site;
  // LDS visibility is inherently workgroup-scoped. Its instruction encoding
  // has no explicit scope field, so supply the semantic scope only after the
  // surrounding sequence has qualified this site as an ordering event.
  if (allow_gfx1250_lds && normalized.mnemonic.starts_with("ds_") && !normalized.raw_scope)
    normalized.raw_scope = 1u;
  return normalized;
}

[[nodiscard]] std::optional<ConSanMoiAtomicEventKind>
atomic_event_kind_for_site(const ConSanResult &result, std::string_view container_name,
                           bool in_kernel, const ConSanAtomicSite &site) {
  const auto event = std::ranges::find_if(result.sync_events, [&](const ConSanSyncEvent &item) {
    return item.kind == ConSanSyncEventKind::Atomic && item.container_name == container_name &&
           item.in_kernel == in_kernel && item.text_offset == site.text_offset;
  });
  if (event == result.sync_events.end())
    return std::nullopt;
  const ConSanSyncSequence *sequence = find_consan_sync_sequence_for_event(result, event->identity);
  if (sequence == nullptr ||
      !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                    ConSanSemanticConfidence::Conservative)) {
    return std::nullopt;
  }
  switch (sequence->memory_role) {
  case ConSanSyncMemoryRole::Release:
    return ConSanMoiAtomicEventKind::Release;
  case ConSanSyncMemoryRole::Acquire:
    return ConSanMoiAtomicEventKind::Acquire;
  case ConSanSyncMemoryRole::AcquireRelease:
    return ConSanMoiAtomicEventKind::AcquireRelease;
  case ConSanSyncMemoryRole::Unknown:
  case ConSanSyncMemoryRole::None:
  case ConSanSyncMemoryRole::SequentiallyConsistent:
    return std::nullopt;
  }
  return std::nullopt;
}

void append_atomic_record_candidates(const ConSanKernelInfo &kernel, const ConSanResult &result,
                                     std::vector<AtomicRecordCandidate> &candidates) {
  if (is_rocclr_runtime_kernel_name(kernel.name))
    return;
  const bool allow_gfx1250_lds = result.arch_name == "gfx1250";
  for (const ConSanAtomicSite &site : kernel.atomic_sites) {
    const auto event_kind = atomic_event_kind_for_site(result, kernel.name, true, site);
    const ConSanAtomicSite normalized = normalize_atomic_record_site(site, allow_gfx1250_lds);
    if (is_first_light_atomic_record_candidate(normalized, allow_gfx1250_lds) && event_kind)
      candidates.push_back({"kernel:" + kernel.name, normalized, kernel.descriptor_file_offset,
                            *event_kind, kernel.uses_gfx1250_cluster_workgroup_id});
  }
}

void append_atomic_record_candidates(const ConSanFunctionInfo &function, const ConSanResult &result,
                                     std::vector<AtomicRecordCandidate> &candidates) {
  const bool allow_gfx1250_lds = result.arch_name == "gfx1250";
  for (const ConSanAtomicSite &site : function.atomic_sites) {
    const auto event_kind = atomic_event_kind_for_site(result, function.name, false, site);
    const ConSanAtomicSite normalized = normalize_atomic_record_site(site, allow_gfx1250_lds);
    if (is_first_light_atomic_record_candidate(normalized, allow_gfx1250_lds) && event_kind)
      candidates.push_back({"function:" + function.name, normalized, std::nullopt, *event_kind});
  }
}

void append_inline_atomic_candidates(const ConSanKernelInfo &kernel, const ConSanResult &result,
                                     std::vector<AtomicRecordCandidate> &candidates) {
  if (is_rocclr_runtime_kernel_name(kernel.name))
    return;
  for (const ConSanAtomicSite &site : kernel.atomic_sites) {
    const auto event_kind = atomic_event_kind_for_site(result, kernel.name, true, site);
    if (event_kind) {
      candidates.push_back({"kernel:" + kernel.name, site, kernel.descriptor_file_offset,
                            *event_kind, kernel.uses_gfx1250_cluster_workgroup_id});
    }
  }
}

void append_inline_atomic_candidates(const ConSanFunctionInfo &function, const ConSanResult &result,
                                     std::vector<AtomicRecordCandidate> &candidates) {
  for (const ConSanAtomicSite &site : function.atomic_sites) {
    const auto event_kind = atomic_event_kind_for_site(result, function.name, false, site);
    if (event_kind)
      candidates.push_back({"function:" + function.name, site, std::nullopt, *event_kind});
  }
}

[[nodiscard]] bool reject_atomic_candidate_scratch_overlap(const ConSanAtomicSite &site,
                                                           uint16_t scratch_vgpr,
                                                           uint16_t scratch_vgpr_count,
                                                           std::vector<std::string> &errors) {
  if (!site.addr_vgpr) {
    errors.emplace_back("ConSan MOI atomic record patch requires an address VGPR");
    return true;
  }
  const uint16_t address_vgpr_count =
      site.mnemonic.starts_with("buffer_") || site.mnemonic.starts_with("ds_") ? 1u : 2u;
  if (range_overlaps(*site.addr_vgpr, address_vgpr_count, scratch_vgpr, scratch_vgpr_count)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch scratch VGPRs overlap the atomic address VGPRs");
    return true;
  }
  if (site.data_vgpr && range_overlaps(*site.data_vgpr, 1, scratch_vgpr, scratch_vgpr_count)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch scratch VGPRs overlap the atomic data VGPR");
    return true;
  }
  if (site.dst_vgpr && range_overlaps(*site.dst_vgpr, 1, scratch_vgpr, scratch_vgpr_count)) {
    errors.emplace_back(
        "ConSan MOI atomic record patch scratch VGPRs overlap the atomic destination VGPR");
    return true;
  }
  return false;
}

[[nodiscard]] bool validate_inline_atomic_exec_save_sgpr(const ConSanOptions &options,
                                                         std::vector<std::string> &errors) {
  if (!options.moi_exec_save_sgpr) {
    errors.emplace_back(
        "ConSan MOI inline atomic acquire patch requires RJ_CONSAN_MOI_EXEC_SAVE_SGPR");
    return false;
  }
  if (*options.moi_exec_save_sgpr > 86u || *options.moi_exec_save_sgpr % 2u != 0u) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch requires an even "
                        "RJ_CONSAN_MOI_EXEC_SAVE_SGPR in 0..86");
    return false;
  }
  return true;
}

[[nodiscard]] bool append_inline_atomic_slot_address(std::vector<uint32_t> &words,
                                                     uint64_t table_base, uint32_t table_capacity,
                                                     uint16_t atomic_address_vgpr,
                                                     uint16_t scratch_vgpr, rj_code_arch_t arch) {
  if (table_capacity == 0 || (table_capacity & (table_capacity - 1u)) != 0)
    return false;
  const uint16_t slot_address_vgpr = scratch_vgpr;
  const uint16_t hash_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  const auto shifted = instrumentation::build_v_lshrrev_b32(
      hash_vgpr, scalar_positive_inline_u32(2), atomic_address_vgpr, arch);
  const auto mixed = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(static_cast<uint16_t>(atomic_address_vgpr + 1u)), hash_vgpr,
      arch);
  const auto folded = instrumentation::build_v_lshrrev_b32(
      slot_address_vgpr, scalar_positive_inline_u32(16), hash_vgpr, arch);
  const auto mix_fold = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(slot_address_vgpr), hash_vgpr, arch);
  const auto folded_again = instrumentation::build_v_lshrrev_b32(
      slot_address_vgpr, scalar_positive_inline_u32(8), hash_vgpr, arch);
  const auto mix_fold_again = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(slot_address_vgpr), hash_vgpr, arch);
  const auto folded_final = instrumentation::build_v_lshrrev_b32(
      slot_address_vgpr, scalar_positive_inline_u32(4), hash_vgpr, arch);
  const auto mix_fold_final = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(slot_address_vgpr), hash_vgpr, arch);
  const auto multiply = instrumentation::build_v_mul_lo_u32_literal(hash_vgpr, slot_address_vgpr,
                                                                    0x85ebca6bu, hash_vgpr, arch);
  const auto masked =
      instrumentation::build_v_and_b32_literal(hash_vgpr, table_capacity - 1u, hash_vgpr, arch);
  const auto times_slot_size = instrumentation::build_v_lshlrev_b32(
      hash_vgpr, scalar_positive_inline_u32(5), hash_vgpr, arch);
  const auto base_lo = instrumentation::build_v_mov_b32_literal(
      slot_address_vgpr, static_cast<uint32_t>(table_base), arch);
  const auto base_hi =
      instrumentation::build_v_mov_b32_literal(static_cast<uint16_t>(slot_address_vgpr + 1u),
                                               static_cast<uint32_t>(table_base >> 32u), arch);
  const auto add_offset =
      instrumentation::build_v_add_u64_vgpr_offset(slot_address_vgpr, hash_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(shifted, mixed, folded, mix_fold, folded_again, mix_fold_again,
                           folded_final, mix_fold_final, multiply, masked, times_slot_size, base_lo,
                           base_hi, add_offset);
}

[[nodiscard]] bool
append_inline_atomic_snapshot_address(std::vector<uint32_t> &words, uint64_t table_base,
                                      uint32_t table_capacity, uint16_t atomic_address_vgpr,
                                      uint16_t snapshot_address_vgpr, uint16_t hash_vgpr,
                                      uint16_t temporary_vgpr, rj_code_arch_t arch) {
  if (table_capacity == 0 || (table_capacity & (table_capacity - 1u)) != 0)
    return false;
  const auto shifted = instrumentation::build_v_lshrrev_b32(
      hash_vgpr, scalar_positive_inline_u32(2), atomic_address_vgpr, arch);
  const auto mixed = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(static_cast<uint16_t>(atomic_address_vgpr + 1u)), hash_vgpr,
      arch);
  const auto folded = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(16), hash_vgpr, arch);
  const auto mix_fold = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
  const auto folded_again = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(8), hash_vgpr, arch);
  const auto mix_fold_again = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
  const auto folded_final = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(4), hash_vgpr, arch);
  const auto mix_fold_final = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
  const auto multiply = instrumentation::build_v_mul_lo_u32_literal(hash_vgpr, temporary_vgpr,
                                                                    0x85ebca6bu, hash_vgpr, arch);
  const auto masked =
      instrumentation::build_v_and_b32_literal(hash_vgpr, table_capacity - 1u, hash_vgpr, arch);
  const auto times_eight = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(3), hash_vgpr, arch);
  const auto times_thirty_two = instrumentation::build_v_lshlrev_b32(
      hash_vgpr, scalar_positive_inline_u32(5), hash_vgpr, arch);
  const auto times_forty = instrumentation::build_v_add_u32(
      hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
  const auto base_lo = instrumentation::build_v_mov_b32_literal(
      snapshot_address_vgpr, static_cast<uint32_t>(table_base), arch);
  const auto base_hi =
      instrumentation::build_v_mov_b32_literal(static_cast<uint16_t>(snapshot_address_vgpr + 1u),
                                               static_cast<uint32_t>(table_base >> 32u), arch);
  const auto add_offset =
      instrumentation::build_v_add_u64_vgpr_offset(snapshot_address_vgpr, hash_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(shifted, mixed, folded, mix_fold, folded_again, mix_fold_again,
                           folded_final, mix_fold_final, multiply, masked, times_eight,
                           times_thirty_two, times_forty, base_lo, base_hi, add_offset);
}

[[nodiscard]] bool append_inline_workgroup_source_vgpr(std::vector<uint32_t> &words,
                                                       uint16_t destination_vgpr,
                                                       const ConSanMoiWorkgroupSource &source,
                                                       rj_code_arch_t arch) {
  if (!source.scalar_src) {
    const auto zero = instrumentation::build_v_mov_b32_literal(destination_vgpr, 0u, arch);
    InstructionSequence sequence(words);
    return sequence.emit(zero);
  }
  words.push_back(build_v_mov_b32_e32(destination_vgpr, *source.scalar_src, arch));
  if (source.mask_low_16) {
    const auto left = instrumentation::build_v_lshlrev_b32(
        destination_vgpr, scalar_positive_inline_u32(16), destination_vgpr, arch);
    const auto right = instrumentation::build_v_lshrrev_b32(
        destination_vgpr, scalar_positive_inline_u32(16), destination_vgpr, arch);
    InstructionSequence sequence(words);
    if (!sequence.emit_all(left, right))
      return false;
  }
  if (source.shift_right_16) {
    const auto shift = instrumentation::build_v_lshrrev_b32(
        destination_vgpr, scalar_positive_inline_u32(16), destination_vgpr, arch);
    InstructionSequence sequence(words);
    if (!sequence.emit(shift))
      return false;
  }
  return true;
}

[[nodiscard]] bool append_inline_workgroup_key(std::vector<uint32_t> &words,
                                               const ConSanMoiWorkgroupSources &sources,
                                               const ConSanOptions &options, uint16_t key_vgpr,
                                               uint16_t coordinate_vgpr, uint16_t value_vgpr,
                                               uint16_t original_exec_save_offset,
                                               rj_code_arch_t arch) {
  if (!options.moi_exec_save_sgpr)
    return false;
  const uint16_t exec_base = *options.moi_exec_save_sgpr;
  const auto save_original = instrumentation::build_s_mov_b64(
      static_cast<uint16_t>(exec_base + original_exec_save_offset), kRdna4ExecLo, arch);
  if (!save_original)
    return false;
  words.push_back(*save_original);

  if (options.moi_workgroup_key_vgpr) {
    words.push_back(
        build_v_mov_b32_e32(key_vgpr, vector_source_vgpr(*options.moi_workgroup_key_vgpr), arch));
    const auto valid =
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), key_vgpr, arch);
    const auto narrow = instrumentation::build_s_and_saveexec_b64(exec_base, kRdna4VccLo, arch);
    InstructionSequence sequence(words);
    return sequence.emit_all(valid, narrow);
  }
  if (options.moi_persistent_workgroup_key_sgpr) {
    words.push_back(
        build_v_mov_b32_e32(key_vgpr, *options.moi_persistent_workgroup_key_sgpr, arch));
    const auto valid =
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), key_vgpr, arch);
    const auto narrow = instrumentation::build_s_and_saveexec_b64(exec_base, kRdna4VccLo, arch);
    InstructionSequence sequence(words);
    return sequence.emit_all(valid, narrow);
  }

  const auto append_coordinate = [&](const ConSanMoiWorkgroupSource &source, uint32_t bits,
                                     uint32_t shift, uint16_t exec_save_offset,
                                     bool initialize) -> bool {
    if (!append_inline_workgroup_source_vgpr(words, coordinate_vgpr, source, arch))
      return false;
    if (source.scalar_src) {
      const uint32_t high_mask = ~((uint32_t{1} << bits) - 1u);
      const auto high =
          instrumentation::build_v_and_b32_literal(value_vgpr, high_mask, coordinate_vgpr, arch);
      const auto fits =
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value_vgpr, arch);
      const auto narrow = instrumentation::build_s_and_saveexec_b64(
          static_cast<uint16_t>(exec_base + exec_save_offset), kRdna4VccLo, arch);
      InstructionSequence sequence(words);
      if (!sequence.emit_all(high, fits, narrow))
        return false;
    }
    if (shift != 0u) {
      const auto shifted = instrumentation::build_v_lshlrev_b32(
          coordinate_vgpr, scalar_positive_inline_u32(shift), coordinate_vgpr, arch);
      InstructionSequence sequence(words);
      if (!sequence.emit(shifted))
        return false;
    }
    if (initialize) {
      words.push_back(build_v_mov_b32_e32(key_vgpr, vector_source_vgpr(coordinate_vgpr), arch));
      return true;
    }
    const auto add = instrumentation::build_v_add_u32(key_vgpr, vector_source_vgpr(coordinate_vgpr),
                                                      key_vgpr, arch);
    InstructionSequence sequence(words);
    return sequence.emit(add);
  };

  const bool has_z = sources.z.scalar_src.has_value();
  const bool has_y = sources.y.scalar_src.has_value();
  const uint32_t x_bits = has_z ? 8u : has_y ? 10u : 20u;
  const uint32_t y_bits = has_z ? 6u : 10u;
  if (!append_coordinate(sources.x, x_bits, 0u, 0u, /*initialize=*/true) ||
      !append_coordinate(sources.y, y_bits, x_bits, 2u, /*initialize=*/false) ||
      !append_coordinate(sources.z, 6u, x_bits + y_bits, 4u, /*initialize=*/false))
    return false;

  const auto max_key = instrumentation::build_v_mov_b32_literal(
      value_vgpr, consan_moi_exact_shadow::max_generation, arch);
  const auto not_reserved =
      instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(key_vgpr), value_vgpr, arch);
  const auto narrow_reserved = instrumentation::build_s_and_saveexec_b64(
      static_cast<uint16_t>(exec_base + 6u), kRdna4VccLo, arch);
  const auto add_one =
      instrumentation::build_v_add_u32(key_vgpr, scalar_positive_inline_u32(1), key_vgpr, arch);
  InstructionSequence sequence(words);
  return sequence.emit_all(max_key, not_reserved, narrow_reserved, add_one);
}

[[nodiscard]] bool append_inline_acquired_token_slot_address(
    std::vector<uint32_t> &words, uint64_t table_base, uint32_t table_capacity,
    uint16_t workgroup_key_vgpr, uint16_t consumer_owner_vgpr, uint16_t producer_owner_vgpr,
    uint16_t slot_address_vgpr, uint16_t hash_vgpr, uint16_t temporary_vgpr, bool release_sequence,
    rj_code_arch_t arch) {
  if (table_capacity == 0 || (table_capacity & (table_capacity - 1u)) != 0)
    return false;
  const uint32_t initialize_hash =
      build_v_mov_b32_e32(hash_vgpr, vector_source_vgpr(workgroup_key_vgpr), arch);
  const auto key_shift = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(6), workgroup_key_vgpr, arch);
  const auto mix_key = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
  const uint32_t copy_consumer =
      build_v_mov_b32_e32(temporary_vgpr, vector_source_vgpr(consumer_owner_vgpr), arch);
  const auto mix_consumer = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
  const auto consumer_shift = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(6), consumer_owner_vgpr, arch);
  const auto mix_consumer_shift = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
  const auto producer_left = instrumentation::build_v_lshlrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(1), producer_owner_vgpr, arch);
  const auto mix_producer_left = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
  const auto producer_right = instrumentation::build_v_lshrrev_b32(
      temporary_vgpr, scalar_positive_inline_u32(5), producer_owner_vgpr, arch);
  const auto mix_producer_right = instrumentation::build_v_xor_b32(
      hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
  const auto masked =
      instrumentation::build_v_and_b32_literal(hash_vgpr, table_capacity - 1u, hash_vgpr, arch);
  const auto times_sixteen = instrumentation::build_v_lshlrev_b32(
      slot_address_vgpr, scalar_positive_inline_u32(4), hash_vgpr, arch);
  const auto times_thirty_two = instrumentation::build_v_lshlrev_b32(
      hash_vgpr, scalar_positive_inline_u32(5), hash_vgpr, arch);
  const auto times_slot_size = instrumentation::build_v_add_u32(
      hash_vgpr, vector_source_vgpr(slot_address_vgpr), hash_vgpr, arch);
  const auto base_lo = instrumentation::build_v_mov_b32_literal(
      slot_address_vgpr, static_cast<uint32_t>(table_base), arch);
  const auto base_hi =
      instrumentation::build_v_mov_b32_literal(static_cast<uint16_t>(slot_address_vgpr + 1u),
                                               static_cast<uint32_t>(table_base >> 32u), arch);
  const auto add_offset =
      instrumentation::build_v_add_u64_vgpr_offset(slot_address_vgpr, hash_vgpr, arch);
  InstructionSequence sequence(words);
  if (!sequence.emit_all(initialize_hash, key_shift, mix_key, copy_consumer, mix_consumer,
                         consumer_shift, mix_consumer_shift, producer_left, mix_producer_left,
                         producer_right, mix_producer_right))
    return false;
  if (release_sequence) {
    const auto salt = instrumentation::build_v_mov_b32_literal(temporary_vgpr, 0x9e3779b9u, arch);
    const auto mix_salt = instrumentation::build_v_xor_b32(
        hash_vgpr, vector_source_vgpr(temporary_vgpr), hash_vgpr, arch);
    if (!salt || !mix_salt)
      return false;
    words.insert(words.end(), salt->begin(), salt->end());
    words.push_back(*mix_salt);
  }
  return sequence.emit_all(masked, times_sixteen, times_thirty_two, times_slot_size, base_lo,
                           base_hi, add_offset);
}

// Publishes the direct acquire edge and every inherited snapshot edge as one
// reservation transaction. The reserved field is an odd-state rollback
// journal containing the prior even version; it is cleared before commit and
// can never authorize a reader while the slot remains odd.
[[nodiscard]] bool append_inline_acquired_token_transaction(
    std::vector<uint32_t> &words, const ConSanOptions &options, uint16_t source_address_vgpr,
    uint16_t workgroup_key_vgpr, uint16_t direct_owner_vgpr, uint16_t direct_epoch_vgpr,
    uint16_t source_version_vgpr, uint16_t snapshot_count_vgpr, uint16_t snapshot_owners_vgpr,
    uint16_t snapshot_epochs_vgpr, uint64_t token_table_base, uint32_t token_table_capacity,
    bool release_sequence, rj_code_arch_t arch) {
  const uint16_t base = *options.scratch_vgpr;
  // CDNA4 FLAT compare-swap encodes the new/expected operands as an aligned
  // VGPR pair. Preserve the RDNA4 allocation, whose three-dword encoding does
  // not have that constraint, and move the CDNA4 pair down by one register.
  const bool cdna4 = arch == ROCJITSU_CODE_ARCH_CDNA4;
  const uint16_t value = static_cast<uint16_t>(base + (cdna4 ? 18u : 19u));
  const uint16_t expected = static_cast<uint16_t>(base + (cdna4 ? 19u : 20u));
  const uint16_t hash = static_cast<uint16_t>(base + 21u);
  const uint16_t saved_direct_owner = static_cast<uint16_t>(base + (cdna4 ? 22u : 17u));
  const uint16_t saved_direct_epoch = static_cast<uint16_t>(base + (cdna4 ? 23u : 18u));
  const uint16_t saved_source_version = static_cast<uint16_t>(base + 8u);
  const uint16_t exec_base = *options.moi_exec_save_sgpr;
  const uint16_t winners = static_cast<uint16_t>(exec_base + 16u);
  const uint16_t needed = static_cast<uint16_t>(exec_base + 18u);
  // The enclosing acquire helper's +6 workgroup predicate and +14 owner
  // predicate are dead before this transaction begins. Reuse those pairs for
  // masks that must survive this transaction instead of extending the scalar
  // window to +23.
  const uint16_t skipped = static_cast<uint16_t>(exec_base + 14u);
  const uint16_t eligible = static_cast<uint16_t>(exec_base + 6u);

  const auto restore_exec = [&](uint16_t source) {
    const auto instruction = instrumentation::build_s_mov_b64(kRdna4ExecLo, source, arch);
    if (!instruction)
      return false;
    words.push_back(*instruction);
    return true;
  };
  const auto save_exec = [&](uint16_t destination) {
    const auto instruction = instrumentation::build_s_mov_b64(destination, kRdna4ExecLo, arch);
    if (!instruction)
      return false;
    words.push_back(*instruction);
    return true;
  };
  const auto narrow_vcc = [&] {
    const auto instruction =
        instrumentation::build_s_and_saveexec_b64(exec_base, kRdna4VccLo, arch);
    if (!instruction)
      return false;
    words.push_back(*instruction);
    return true;
  };
  const auto require_equal = [&](size_t offset, uint16_t expected_vgpr) {
    if (!append_load_u32_vgpr_at_offset(words, base, offset, value, arch))
      return false;
    const auto equal =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected_vgpr), value, arch);
    if (!equal)
      return false;
    words.push_back(*equal);
    return narrow_vcc();
  };
  const auto select_entry = [&](uint16_t index) {
    if (!restore_exec(winners))
      return false;
    if (index == 0)
      return save_exec(needed);
    const auto threshold =
        instrumentation::build_v_mov_b32_literal(value, static_cast<uint32_t>(index - 1u), arch);
    const auto present = instrumentation::build_v_cmp_gt_u32_vcc(
        vector_source_vgpr(snapshot_count_vgpr), value, arch);
    if (!threshold || !present)
      return false;
    words.insert(words.end(), threshold->begin(), threshold->end());
    words.push_back(*present);
    if (!narrow_vcc() || !save_exec(needed) || !restore_exec(winners))
      return false;
    const auto absent = instrumentation::build_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(index),
                                                                snapshot_count_vgpr, arch);
    if (!absent)
      return false;
    words.push_back(*absent);
    if (!narrow_vcc())
      return false;
    const auto tail_owner_zero = instrumentation::build_v_cmp_eq_u32_vcc(
        scalar_positive_inline_u32(0), static_cast<uint16_t>(snapshot_owners_vgpr + index - 1u),
        arch);
    const auto tail_epoch_zero = instrumentation::build_v_cmp_eq_u32_vcc(
        scalar_positive_inline_u32(0), static_cast<uint16_t>(snapshot_epochs_vgpr + index - 1u),
        arch);
    if (!tail_owner_zero || !tail_epoch_zero)
      return false;
    words.push_back(*tail_owner_zero);
    if (!narrow_vcc())
      return false;
    words.push_back(*tail_epoch_zero);
    return narrow_vcc() && save_exec(skipped) && restore_exec(needed);
  };
  const auto select_identity = [&](uint16_t index) {
    if (index == 0) {
      words.push_back(
          build_v_mov_b32_e32(direct_owner_vgpr, vector_source_vgpr(saved_direct_owner), arch));
      words.push_back(
          build_v_mov_b32_e32(direct_epoch_vgpr, vector_source_vgpr(saved_direct_epoch), arch));
      return true;
    }
    words.push_back(build_v_mov_b32_e32(
        direct_owner_vgpr,
        vector_source_vgpr(static_cast<uint16_t>(snapshot_owners_vgpr + index - 1u)), arch));
    words.push_back(build_v_mov_b32_e32(
        direct_epoch_vgpr,
        vector_source_vgpr(static_cast<uint16_t>(snapshot_epochs_vgpr + index - 1u)), arch));
    const auto owner_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), direct_owner_vgpr, arch);
    if (!owner_nonzero)
      return false;
    words.push_back(*owner_nonzero);
    if (!narrow_vcc())
      return false;
    const auto owner_not_consumer = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(*options.moi_owner_vgpr), direct_owner_vgpr, arch);
    const auto owner_not_releaser = instrumentation::build_v_cmp_ne_u32_vcc(
        vector_source_vgpr(saved_direct_owner), direct_owner_vgpr, arch);
    const auto epoch_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(
        scalar_positive_inline_u32(0), direct_epoch_vgpr, arch);
    const auto epoch_limit = instrumentation::build_v_mov_b32_literal(
        value, consan_moi_exact_shadow::max_epoch + 1u, arch);
    const auto epoch_in_range =
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(value), direct_epoch_vgpr, arch);
    if (!owner_not_consumer || !owner_not_releaser || !epoch_nonzero || !epoch_limit ||
        !epoch_in_range)
      return false;
    words.insert(words.end(), epoch_limit->begin(), epoch_limit->end());
    for (const auto instruction :
         {*owner_not_consumer, *owner_not_releaser, *epoch_nonzero, *epoch_in_range}) {
      words.push_back(instruction);
      if (!narrow_vcc())
        return false;
    }
    if (index > 1u) {
      const auto canonical = instrumentation::build_v_cmp_gt_u32_vcc(
          vector_source_vgpr(direct_owner_vgpr),
          static_cast<uint16_t>(snapshot_owners_vgpr + index - 2u), arch);
      if (!canonical)
        return false;
      words.push_back(*canonical);
      if (!narrow_vcc())
        return false;
    }
    return true;
  };
  const auto derive_slot = [&] {
    return append_inline_acquired_token_slot_address(
        words, token_table_base, token_table_capacity, workgroup_key_vgpr, *options.moi_owner_vgpr,
        direct_owner_vgpr, base, hash, value, release_sequence, arch);
  };

  words.push_back(
      build_v_mov_b32_e32(saved_direct_owner, vector_source_vgpr(direct_owner_vgpr), arch));
  words.push_back(
      build_v_mov_b32_e32(saved_direct_epoch, vector_source_vgpr(direct_epoch_vgpr), arch));
  words.push_back(
      build_v_mov_b32_e32(saved_source_version, vector_source_vgpr(source_version_vgpr), arch));
  if (!save_exec(eligible) || !save_exec(winners))
    return false;

  constexpr std::array<size_t, 11> kPayloadOffsets = {
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

  // Phase one: reserve every required destination odd. Empty and exact stable
  // pair slots are accepted; all other states remove the lane from winners.
  for (uint16_t index = 0; index < 5u; ++index) {
    if (!select_entry(index) || !select_identity(index) || !derive_slot() ||
        !append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), value, arch) ||
        !save_exec(needed))
      return false;
    words.push_back(build_v_mov_b32_e32(expected, vector_source_vgpr(value), arch));

    const auto empty =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
    if (!empty)
      return false;
    words.push_back(*empty);
    if (!narrow_vcc())
      return false;
    for (size_t offset : kPayloadOffsets) {
      if (!append_load_u32_vgpr_at_offset(words, base, offset, value, arch))
        return false;
      const auto zero =
          instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
      if (!zero)
        return false;
      words.push_back(*zero);
      if (!narrow_vcc())
        return false;
    }
    if (!save_exec(static_cast<uint16_t>(exec_base + 2u)) || !restore_exec(needed))
      return false;

    const auto nonzero =
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), expected, arch);
    const auto parity = instrumentation::build_v_and_b32_literal(value, 1u, expected, arch);
    if (!nonzero || !parity)
      return false;
    words.push_back(*nonzero);
    if (!narrow_vcc())
      return false;
    words.insert(words.end(), parity->begin(), parity->end());
    const auto even =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
    if (!even)
      return false;
    words.push_back(*even);
    if (!narrow_vcc() ||
        !require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
                       workgroup_key_vgpr) ||
        !require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
                       *options.moi_owner_vgpr) ||
        !require_equal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
                       direct_owner_vgpr))
      return false;
    if (!append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one),
            value, arch))
      return false;
    const auto prior_epoch_nonzero =
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), value, arch);
    const auto prior_epoch_limit = instrumentation::build_v_mov_b32_literal(
        hash, consan_moi_exact_shadow::max_epoch + 1u, arch);
    const auto prior_epoch_in_range =
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(hash), value, arch);
    if (!prior_epoch_nonzero || !prior_epoch_limit || !prior_epoch_in_range)
      return false;
    words.insert(words.end(), prior_epoch_limit->begin(), prior_epoch_limit->end());
    words.push_back(*prior_epoch_nonzero);
    if (!narrow_vcc())
      return false;
    words.push_back(*prior_epoch_in_range);
    if (!narrow_vcc() ||
        !append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind), value, arch))
      return false;
    const auto kind_valid =
        instrumentation::build_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(3), value, arch);
    if (!kind_valid)
      return false;
    words.push_back(*kind_valid);
    if (!narrow_vcc() ||
        !append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id), value, arch))
      return false;
    const auto dispatch_low =
        instrumentation::build_v_cmp_eq_u32_vcc(*options.moi_dispatch_id_sgpr, value, arch);
    if (!dispatch_low)
      return false;
    words.push_back(*dispatch_low);
    if (!narrow_vcc() ||
        !append_load_u32_vgpr_at_offset(
            words, base,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) + sizeof(uint32_t), value,
            arch))
      return false;
    const auto dispatch_high = instrumentation::build_v_cmp_eq_u32_vcc(
        static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u), value, arch);
    if (!dispatch_high)
      return false;
    words.push_back(*dispatch_high);
    if (!narrow_vcc() ||
        !append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address),
            value, arch))
      return false;
    // Requiring the low word to be nonzero is a conservative generated-code
    // subset of the host classifier: aligned 4-GiB boundary addresses fail
    // closed instead of becoming ordering authority.
    const auto source_address_valid =
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), value, arch);
    if (!source_address_valid)
      return false;
    words.push_back(*source_address_valid);
    if (!narrow_vcc() ||
        !append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version),
            value, arch))
      return false;
    const auto source_version_nonzero =
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), value, arch);
    const auto source_version_bits =
        instrumentation::build_v_and_b32_literal(hash, 1u, value, arch);
    if (!source_version_nonzero || !source_version_bits)
      return false;
    words.push_back(*source_version_nonzero);
    if (!narrow_vcc())
      return false;
    words.insert(words.end(), source_version_bits->begin(), source_version_bits->end());
    const auto source_version_even =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), hash, arch);
    if (!source_version_even)
      return false;
    words.push_back(*source_version_even);
    if (!narrow_vcc() ||
        !append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reserved), value, arch))
      return false;
    const auto reserved_zero =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), value, arch);
    if (!reserved_zero)
      return false;
    words.push_back(*reserved_zero);
    if (!narrow_vcc())
      return false;
    if (!save_exec(static_cast<uint16_t>(exec_base + 4u)))
      return false;
    const auto valid_union =
        instrumentation::build_s_xor_b64(kRdna4ExecLo, static_cast<uint16_t>(exec_base + 2u),
                                         static_cast<uint16_t>(exec_base + 4u), arch);
    if (!valid_union)
      return false;
    words.push_back(*valid_union);
    words.push_back(build_v_mov_b32_e32(value, vector_source_vgpr(expected), arch));
    const auto increment =
        instrumentation::build_v_add_u32(value, scalar_positive_inline_u32(1), value, arch);
    if (!increment ||
        !append_add_literal_field(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), hash, arch))
      return false;
    words.insert(words.end(), increment->begin(), increment->end());
    const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(
        base, value, value, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    if (!claim)
      return false;
    words.insert(words.end(), claim->begin(), claim->end());
    if (!append_moi_flat_load_wait(words, arch))
      return false;
    const auto won =
        instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(expected), value, arch);
    if (!won)
      return false;
    words.push_back(*won);
    if (!narrow_vcc() ||
        !append_store_u32_vgpr_at_offset(words, base,
                                         offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reserved),
                                         expected, arch) ||
        !save_exec(needed))
      return false;
    const auto combine = index == 0
                             ? instrumentation::build_s_mov_b64(winners, needed, arch)
                             : instrumentation::build_s_xor_b64(winners, skipped, needed, arch);
    if (!combine)
      return false;
    words.push_back(*combine);
  }

  const auto failed = instrumentation::build_s_andn2_b64(kRdna4ExecLo, eligible, winners, arch);
  if (!failed)
    return false;
  words.push_back(*failed);
  if (!append_atomic_fetch_add_one_u32(words,
                                       *options.moi_report_buffer_address +
                                           offsetof(ConSanMoiReportHeader, inline_overflow_count),
                                       value, cdna4 ? base : hash, arch))
    return false;

  // Roll back every journaled reservation for failed lanes. A CAS whose slot
  // was never claimed simply loses and leaves foreign evidence untouched.
  for (uint16_t index = 0; index < 5u; ++index) {
    if (!restore_exec(eligible))
      return false;
    const auto loser = instrumentation::build_s_andn2_b64(kRdna4ExecLo, eligible, winners, arch);
    if (!loser)
      return false;
    words.push_back(*loser);
    if (index != 0) {
      const auto threshold =
          instrumentation::build_v_mov_b32_literal(value, static_cast<uint32_t>(index - 1u), arch);
      const auto present = instrumentation::build_v_cmp_gt_u32_vcc(
          vector_source_vgpr(snapshot_count_vgpr), value, arch);
      if (!threshold || !present)
        return false;
      words.insert(words.end(), threshold->begin(), threshold->end());
      words.push_back(*present);
      if (!narrow_vcc())
        return false;
    }
    if (!select_identity(index) || !derive_slot() ||
        !append_load_u32_vgpr_at_offset(words, base,
                                        offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reserved),
                                        expected, arch) ||
        !append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), value, arch))
      return false;
    const auto odd_bits = instrumentation::build_v_and_b32_literal(hash, 1u, value, arch);
    const auto odd =
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), hash, arch);
    if (!odd_bits || !odd)
      return false;
    words.insert(words.end(), odd_bits->begin(), odd_bits->end());
    words.push_back(*odd);
    if (!narrow_vcc() ||
        !append_add_literal_field(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), hash, arch))
      return false;
    words.push_back(
        build_v_mov_b32_e32(static_cast<uint16_t>(value + 1u), vector_source_vgpr(value), arch));
    words.push_back(build_v_mov_b32_e32(value, vector_source_vgpr(expected), arch));
    const auto rollback = instrumentation::build_flat_atomic_cmpswap_b32(
        base, value, value, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    if (!rollback)
      return false;
    words.insert(words.end(), rollback->begin(), rollback->end());
    if (!append_moi_flat_load_wait(words, arch))
      return false;
  }

  // Phase two: winners fill every payload, drain, then commit each odd version
  // even. Readers therefore see either the old frontier or complete tokens.
  for (uint16_t index = 0; index < 5u; ++index) {
    if (!restore_exec(winners))
      return false;
    if (index != 0) {
      const auto threshold =
          instrumentation::build_v_mov_b32_literal(value, static_cast<uint32_t>(index - 1u), arch);
      const auto present = instrumentation::build_v_cmp_gt_u32_vcc(
          vector_source_vgpr(snapshot_count_vgpr), value, arch);
      if (!threshold || !present)
        return false;
      words.insert(words.end(), threshold->begin(), threshold->end());
      words.push_back(*present);
      if (!narrow_vcc())
        return false;
    }
    if (!select_identity(index) || !derive_slot() ||
        !append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
            *options.moi_owner_vgpr, arch) ||
        !append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id),
            direct_owner_vgpr, arch) ||
        !append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one),
            direct_epoch_vgpr, arch) ||
        !append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
            workgroup_key_vgpr, arch))
      return false;
    const auto kind = instrumentation::build_v_mov_b32_literal(
        value,
        release_sequence ? static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::ReleaseSequence)
        : index == 0     ? static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Direct)
                         : static_cast<uint32_t>(ConSanMoiInlineTokenEvidenceKind::Inherited),
        arch);
    if (!kind)
      return false;
    words.insert(words.end(), kind->begin(), kind->end());
    if (!append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind), value, arch) ||
        !append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address),
            source_address_vgpr, arch) ||
        !append_store_u32_vgpr_at_offset(
            words, base,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address) +
                sizeof(uint32_t),
            static_cast<uint16_t>(source_address_vgpr + 1u), arch) ||
        !append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version),
            saved_source_version, arch))
      return false;
    const auto zero = instrumentation::build_v_mov_b32_literal(value, 0, arch);
    if (!zero)
      return false;
    words.insert(words.end(), zero->begin(), zero->end());
    if (!append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reserved), value, arch))
      return false;
    words.push_back(build_v_mov_b32_e32(value, *options.moi_dispatch_id_sgpr, arch));
    if (!append_store_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id), value, arch))
      return false;
    words.push_back(build_v_mov_b32_e32(
        value, static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u), arch));
    if (!append_store_u32_vgpr_at_offset(
            words, base,
            offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) + sizeof(uint32_t), value,
            arch))
      return false;
    const auto wait = instrumentation::build_s_wait_flat_store0(arch);
    if (!wait ||
        !append_load_u32_vgpr_at_offset(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), value, arch))
      return false;
    words.push_back(*wait);
    words.push_back(build_v_mov_b32_e32(expected, vector_source_vgpr(value), arch));
    const auto ready =
        instrumentation::build_v_add_u32(value, scalar_positive_inline_u32(1), value, arch);
    if (!ready ||
        !append_add_literal_field(
            words, base, offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), hash, arch))
      return false;
    words.insert(words.end(), ready->begin(), ready->end());
    const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
        base, value, value, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
    if (!commit)
      return false;
    words.insert(words.end(), commit->begin(), commit->end());
    if (!append_moi_flat_load_wait(words, arch))
      return false;
  }
  return restore_exec(winners);
}

[[nodiscard]] bool append_inline_atomic_acquire_import(
    std::vector<uint32_t> &words, const ConSanOptions &options, uint16_t address_vgpr,
    uint16_t workgroup_key_vgpr, uint16_t producer_owner_vgpr, uint16_t token_value_vgpr,
    uint16_t temporary_vgpr, uint64_t snapshot_table_base, uint32_t snapshot_table_capacity,
    uint64_t token_table_base, uint32_t token_table_capacity, bool source_slot_claimed,
    bool release_sequence_only, rj_code_arch_t arch, std::vector<std::string> &errors) {
  const uint16_t scratch_vgpr = *options.scratch_vgpr;
  const uint16_t value_vgpr = static_cast<uint16_t>(scratch_vgpr + 2u);
  const uint16_t exec_base = *options.moi_exec_save_sgpr;
  const auto narrow_if_valid =
      instrumentation::build_s_and_saveexec_b64(exec_base, kRdna4VccLo, arch);
  const auto narrow_if_low_matches = instrumentation::build_s_and_saveexec_b64(
      static_cast<uint16_t>(exec_base + 2u), kRdna4VccLo, arch);
  const auto narrow_if_high_matches = instrumentation::build_s_and_saveexec_b64(
      static_cast<uint16_t>(exec_base + 4u), kRdna4VccLo, arch);
  const auto narrow_if_workgroup_matches = instrumentation::build_s_and_saveexec_b64(
      static_cast<uint16_t>(exec_base + 6u), kRdna4VccLo, arch);
  const auto narrow_if_other_owner = instrumentation::build_s_and_saveexec_b64(
      static_cast<uint16_t>(exec_base + 14u), kRdna4VccLo, arch);
  const auto save_matching_exec =
      instrumentation::build_s_mov_b64(static_cast<uint16_t>(exec_base + 16u), kRdna4ExecLo, arch);
  const auto restore_workgroup_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, static_cast<uint16_t>(exec_base + 12u), arch);
  if (!narrow_if_valid || !narrow_if_low_matches || !narrow_if_high_matches ||
      !narrow_if_workgroup_matches || !narrow_if_other_owner || !save_matching_exec ||
      !restore_workgroup_exec) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not encode EXEC ops");
    return false;
  }

  if (!append_atomic_load_u32(words, scratch_vgpr, value_vgpr, arch)) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not load release version");
    return false;
  }
  const uint16_t version_before = static_cast<uint16_t>(scratch_vgpr + 20u);
  std::optional<std::vector<uint32_t>> claimed_version;
  if (source_slot_claimed) {
    claimed_version = instrumentation::build_v_add_u32(
        temporary_vgpr, scalar_positive_inline_u32(1), version_before, arch);
    if (!claimed_version)
      return false;
    words.insert(words.end(), claimed_version->begin(), claimed_version->end());
  }
  const auto version_matches = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(source_slot_claimed ? temporary_vgpr : version_before), value_vgpr, arch);
  const auto version_parity =
      instrumentation::build_v_and_b32_literal(temporary_vgpr, 1u, value_vgpr, arch);
  const auto version_parity_matches =
      source_slot_claimed ? instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                                    temporary_vgpr, arch)
                          : instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0),
                                                                    temporary_vgpr, arch);
  const auto version_nonzero =
      instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), value_vgpr, arch);
  InstructionSequence sequence(words);
  if (!sequence.emit_all(version_matches, narrow_if_valid, version_parity, version_parity_matches,
                         narrow_if_valid, version_nonzero, narrow_if_valid)) {
    errors.emplace_back(
        "ConSan MOI inline atomic acquire patch could not classify release version");
    return false;
  }

  if (!append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                      offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                                      value_vgpr, arch)) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not load address low");
    return false;
  }
  const auto low_eq =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(address_vgpr), value_vgpr, arch);
  if (!low_eq) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not compare address low");
    return false;
  }
  words.push_back(*low_eq);
  words.push_back(*narrow_if_low_matches);

  if (!append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                      offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) +
                                          sizeof(uint32_t),
                                      value_vgpr, arch)) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not load address high");
    return false;
  }
  const auto high_eq = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(static_cast<uint16_t>(address_vgpr + 1u)), value_vgpr, arch);
  if (!high_eq) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not compare address high");
    return false;
  }
  words.push_back(*high_eq);
  words.push_back(*narrow_if_high_matches);

  if (!append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                      offsetof(ConSanMoiInlineAtomicReleaseSlot, workgroup_key),
                                      value_vgpr, arch)) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not load workgroup key");
    return false;
  }
  const auto workgroup_eq = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(workgroup_key_vgpr), value_vgpr, arch);
  if (!workgroup_eq) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not compare workgroup key");
    return false;
  }
  words.push_back(*workgroup_eq);
  words.push_back(*narrow_if_workgroup_matches);

  if (!append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                      offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                                      producer_owner_vgpr, arch)) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not load owner");
    return false;
  }
  const auto owner_ne = instrumentation::build_v_cmp_ne_u32_vcc(
      vector_source_vgpr(*options.moi_owner_vgpr), producer_owner_vgpr, arch);
  if (!owner_ne) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not compare owner");
    return false;
  }
  words.push_back(*owner_ne);
  words.push_back(*narrow_if_other_owner);

  if (!append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                      offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch_plus_one),
                                      token_value_vgpr, arch)) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not load epoch_plus_one");
    return false;
  }
  if (!options.moi_dispatch_id_sgpr ||
      !append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                      offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id),
                                      value_vgpr, arch))
    return false;
  const auto dispatch_low_matches =
      instrumentation::build_v_cmp_eq_u32_vcc(*options.moi_dispatch_id_sgpr, value_vgpr, arch);
  if (!dispatch_low_matches)
    return false;
  words.push_back(*dispatch_low_matches);
  words.push_back(*narrow_if_valid);
  if (!append_load_u32_vgpr_at_offset(words, scratch_vgpr,
                                      offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id) +
                                          sizeof(uint32_t),
                                      value_vgpr, arch))
    return false;
  const auto dispatch_high_matches = instrumentation::build_v_cmp_eq_u32_vcc(
      static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u), value_vgpr, arch);
  if (!dispatch_high_matches)
    return false;
  words.push_back(*dispatch_high_matches);
  words.push_back(*narrow_if_valid);
  // CDNA4 FLAT memory instructions require an even-aligned address pair.
  // Swap the address and temporary lifetimes within the existing register
  // window; the token transaction reuses this space only after the snapshot
  // loads have completed.
  const bool cdna4 = arch == ROCJITSU_CODE_ARCH_CDNA4;
  const uint16_t snapshot_address = static_cast<uint16_t>(scratch_vgpr + (cdna4 ? 18u : 17u));
  const uint16_t snapshot_temporary = static_cast<uint16_t>(scratch_vgpr + (cdna4 ? 17u : 19u));
  const uint16_t snapshot_hash = static_cast<uint16_t>(scratch_vgpr + 21u);
  const uint16_t snapshot_count = static_cast<uint16_t>(scratch_vgpr + 7u);
  const uint16_t snapshot_flags = static_cast<uint16_t>(scratch_vgpr + 8u);
  if (!append_inline_atomic_snapshot_address(words, snapshot_table_base, snapshot_table_capacity,
                                             address_vgpr, snapshot_address, snapshot_temporary,
                                             snapshot_hash, arch) ||
      !append_load_u32_vgpr_at_offset(words, snapshot_address,
                                      offsetof(ConSanMoiInlineCausalSnapshot, entry_count),
                                      snapshot_count, arch) ||
      !append_load_u32_vgpr_at_offset(words, snapshot_address,
                                      offsetof(ConSanMoiInlineCausalSnapshot, flags),
                                      snapshot_flags, arch))
    return false;
  const auto flags_clear =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), snapshot_flags, arch);
  const auto count_in_range = instrumentation::build_v_cmp_gt_u32_vcc(
      scalar_positive_inline_u32(kConSanMoiInlineCausalSnapshotEntryCapacity + 1u), snapshot_count,
      arch);
  if (!sequence.emit_all(flags_clear, narrow_if_valid, count_in_range, narrow_if_valid))
    return false;
  for (uint16_t i = 0; i < kConSanMoiInlineCausalSnapshotEntryCapacity; ++i) {
    const size_t entry_offset = offsetof(ConSanMoiInlineCausalSnapshot, entries) +
                                i * sizeof(ConSanMoiInlineCausalSnapshotEntry);
    if (!append_load_u32_vgpr_at_offset(words, snapshot_address, entry_offset,
                                        static_cast<uint16_t>(scratch_vgpr + 9u + i), arch) ||
        !append_load_u32_vgpr_at_offset(words, snapshot_address, entry_offset + sizeof(uint32_t),
                                        static_cast<uint16_t>(scratch_vgpr + 13u + i), arch))
      return false;
  }
  if (!append_atomic_load_u32(words, scratch_vgpr, value_vgpr, arch))
    return false;
  if (source_slot_claimed) {
    words.insert(words.end(), claimed_version->begin(), claimed_version->end());
  }
  const auto final_version_matches = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(source_slot_claimed ? temporary_vgpr : version_before), value_vgpr, arch);
  if (!final_version_matches)
    return false;
  words.push_back(*final_version_matches);
  words.push_back(*narrow_if_valid);
  words.push_back(*save_matching_exec);
  const auto epoch_nonzero = instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0),
                                                                     token_value_vgpr, arch);
  const auto epoch_limit = instrumentation::build_v_mov_b32_literal(
      value_vgpr, consan_moi_exact_shadow::max_epoch + 1u, arch);
  const auto epoch_in_range = instrumentation::build_v_cmp_gt_u32_vcc(
      vector_source_vgpr(value_vgpr), token_value_vgpr, arch);
  const auto narrow_epoch = instrumentation::build_s_and_saveexec_b64(exec_base, kRdna4VccLo, arch);
  const auto restore_matching_exec =
      instrumentation::build_s_mov_b64(kRdna4ExecLo, static_cast<uint16_t>(exec_base + 16u), arch);
  if (!sequence.emit_all(epoch_nonzero, narrow_epoch, epoch_limit, epoch_in_range, narrow_epoch,
                         restore_matching_exec)) {
    errors.emplace_back("ConSan MOI inline atomic acquire patch could not validate token epoch");
    return false;
  }

  if (!append_inline_acquired_token_transaction(
          words, options, address_vgpr, workgroup_key_vgpr, producer_owner_vgpr, token_value_vgpr,
          version_before, snapshot_count, static_cast<uint16_t>(scratch_vgpr + 9u),
          static_cast<uint16_t>(scratch_vgpr + 13u), token_table_base, token_table_capacity,
          release_sequence_only, arch))
    return false;
  words.push_back(*restore_workgroup_exec);
  return true;
}

// Capture the complete stable causal frontier for the currently active
// release publishers. EXEC on entry and exit is the successful release-claim
// mask. The generated loop scans every ABI-v6 token slot exactly once and
// maintains four canonical (owner, epoch+1) pairs independently per lane.
[[nodiscard]] bool append_inline_release_causal_snapshot_capture(
    std::vector<uint32_t> &words, const ConSanOptions &options, uint16_t atomic_address_vgpr,
    uint16_t workgroup_key_vgpr, uint64_t token_table_base, uint32_t token_table_capacity,
    uint64_t snapshot_table_base, uint32_t snapshot_table_capacity, rj_code_arch_t arch) {
  if (!options.scratch_vgpr || !options.moi_exec_save_sgpr || !options.moi_dispatch_id_sgpr ||
      !options.moi_owner_vgpr || token_table_capacity == 0 || token_table_capacity > 64u ||
      snapshot_table_capacity != token_table_capacity)
    return false;
  const uint16_t base = *options.scratch_vgpr;
  const bool cdna4 = arch == ROCJITSU_CODE_ARCH_CDNA4;
  const uint16_t snapshot_address = static_cast<uint16_t>(base + (cdna4 ? 6u : 5u));
  const uint16_t count = static_cast<uint16_t>(base + (cdna4 ? 5u : 7u));
  const uint16_t flags = static_cast<uint16_t>(base + 8u);
  const uint16_t owners = static_cast<uint16_t>(base + 9u);
  const uint16_t epochs = static_cast<uint16_t>(base + 13u);
  const uint16_t version_before = static_cast<uint16_t>(base + 17u);
  const uint16_t producer = static_cast<uint16_t>(base + 18u);
  const uint16_t producer_epoch = static_cast<uint16_t>(base + 19u);
  const uint16_t temporary = static_cast<uint16_t>(base + 20u);
  const uint16_t loop_index = static_cast<uint16_t>(base + 21u);
  const uint16_t exec_base = *options.moi_exec_save_sgpr;
  const uint16_t narrow_save = exec_base;
  const uint16_t empty_exec = static_cast<uint16_t>(exec_base + 2u);
  const uint16_t ready_exec = static_cast<uint16_t>(exec_base + 4u);
  const uint16_t candidate_exec = static_cast<uint16_t>(exec_base + 6u);
  const uint16_t scan_exec = static_cast<uint16_t>(exec_base + 14u);
  const uint16_t mask_a = static_cast<uint16_t>(exec_base + 16u);
  const uint16_t mask_b = static_cast<uint16_t>(exec_base + 18u);
  // Empty is dead before the first flag update. The narrow-save pair is
  // scratch after each saveexec. Alias those two disjoint lifetimes rather
  // than reserving a third long-lived mask pair.
  const uint16_t flag_exec = empty_exec;
  const uint16_t high_nonzero_exec = narrow_save;

  // The vector loop counter cannot advance with EXEC empty. Branch around the
  // complete scan in that case; scalar execution resumes at the caller with
  // the same empty EXEC mask.
  const auto enter_if_nonempty = instrumentation::build_s_cbranch_execnz(/*offset_dwords=*/1, arch);
  if (!enter_if_nonempty)
    return false;
  words.push_back(*enter_if_nonempty);
  const size_t empty_exec_skip = words.size();
  words.push_back(build_s_branch(/*offset_dwords=*/0, arch));
  const auto save_scan_exec = instrumentation::build_s_mov_b64(scan_exec, kRdna4ExecLo, arch);
  if (!save_scan_exec)
    return false;
  words.push_back(*save_scan_exec);

  const auto restore_exec = [&](uint16_t source) -> bool {
    const auto restore = instrumentation::build_s_mov_b64(kRdna4ExecLo, source, arch);
    if (!restore)
      return false;
    words.push_back(*restore);
    return true;
  };
  const auto save_exec = [&](uint16_t destination) -> bool {
    const auto save = instrumentation::build_s_mov_b64(destination, kRdna4ExecLo, arch);
    if (!save)
      return false;
    words.push_back(*save);
    return true;
  };
  const auto narrow_vcc = [&]() -> bool {
    const auto narrow = instrumentation::build_s_and_saveexec_b64(narrow_save, kRdna4VccLo, arch);
    if (!narrow)
      return false;
    words.push_back(*narrow);
    return true;
  };
  const auto require_literal = [&](uint16_t value, uint32_t literal, bool equal) -> bool {
    const auto compare = equal ? instrumentation::build_v_cmp_eq_u32_vcc(
                                     scalar_positive_inline_u32(literal), value, arch)
                               : instrumentation::build_v_cmp_ne_u32_vcc(
                                     scalar_positive_inline_u32(literal), value, arch);
    if (!compare)
      return false;
    words.push_back(*compare);
    return narrow_vcc();
  };
  const auto load_require_literal = [&](size_t offset, uint32_t literal, bool equal) -> bool {
    return append_load_u32_vgpr_at_offset(words, snapshot_address, offset, temporary, arch) &&
           require_literal(temporary, literal, equal);
  };
  const auto set_flag = [&](ConSanMoiInlineCausalSnapshotFlag flag) -> bool {
    const uint32_t bit = consan_moi_inline_causal_snapshot_flag(flag);
    if (!save_exec(flag_exec))
      return false;
    const auto prior = instrumentation::build_v_and_b32_literal(temporary, bit, flags, arch);
    const auto unset =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), temporary, arch);
    const auto add =
        instrumentation::build_v_add_u32(flags, scalar_positive_inline_u32(bit), flags, arch);
    if (!prior || !unset || !add)
      return false;
    words.insert(words.end(), prior->begin(), prior->end());
    words.push_back(*unset);
    if (!narrow_vcc())
      return false;
    words.insert(words.end(), add->begin(), add->end());
    return restore_exec(flag_exec);
  };
  const auto require_nonzero_u64 = [&](size_t offset) -> bool {
    if (!save_exec(mask_a) ||
        !append_load_u32_vgpr_at_offset(words, snapshot_address, offset, temporary, arch))
      return false;
    const auto low_nonzero =
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), temporary, arch);
    if (!low_nonzero)
      return false;
    words.push_back(*low_nonzero);
    if (!narrow_vcc() || !save_exec(mask_b) || !restore_exec(mask_a))
      return false;
    const auto low_zero =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0), temporary, arch);
    if (!low_zero)
      return false;
    words.push_back(*low_zero);
    if (!narrow_vcc() || !append_load_u32_vgpr_at_offset(
                             words, snapshot_address, offset + sizeof(uint32_t), temporary, arch))
      return false;
    const auto high_nonzero =
        instrumentation::build_v_cmp_ne_u32_vcc(scalar_positive_inline_u32(0), temporary, arch);
    if (!high_nonzero)
      return false;
    words.push_back(*high_nonzero);
    if (!narrow_vcc() || !save_exec(high_nonzero_exec))
      return false;
    const auto combine =
        instrumentation::build_s_xor_b64(kRdna4ExecLo, mask_b, high_nonzero_exec, arch);
    if (!combine)
      return false;
    words.push_back(*combine);
    return true;
  };

  const auto zero = instrumentation::build_v_mov_b32_literal(temporary, 0, arch);
  const auto token_lo = instrumentation::build_v_mov_b32_literal(
      snapshot_address, static_cast<uint32_t>(token_table_base), arch);
  const auto token_hi = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(snapshot_address + 1u), static_cast<uint32_t>(token_table_base >> 32u),
      arch);
  if (!zero || !token_lo || !token_hi)
    return false;
  words.insert(words.end(), zero->begin(), zero->end());
  words.push_back(build_v_mov_b32_e32(count, vector_source_vgpr(temporary), arch));
  words.push_back(build_v_mov_b32_e32(flags, vector_source_vgpr(temporary), arch));
  for (uint16_t i = 0; i < 4u; ++i) {
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(owners + i),
                                        vector_source_vgpr(temporary), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(epochs + i),
                                        vector_source_vgpr(temporary), arch));
  }
  words.push_back(build_v_mov_b32_e32(loop_index, vector_source_vgpr(temporary), arch));
  words.insert(words.end(), token_lo->begin(), token_lo->end());
  words.insert(words.end(), token_hi->begin(), token_hi->end());

  const size_t loop_begin = words.size();
  if (!restore_exec(scan_exec) ||
      !append_load_u32_vgpr_at_offset(words, snapshot_address,
                                      offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version),
                                      version_before, arch) ||
      !require_literal(version_before, 0, /*equal=*/true))
    return false;
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
  for (size_t offset : kTokenPayloadOffsets) {
    if (!load_require_literal(offset, 0, /*equal=*/true))
      return false;
  }
  if (!load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version), 0,
                            /*equal=*/true) ||
      !save_exec(empty_exec))
    return false;

  if (!restore_exec(scan_exec) || !require_literal(version_before, 0, /*equal=*/false))
    return false;
  const auto parity = instrumentation::build_v_and_b32_literal(temporary, 1u, version_before, arch);
  if (!parity)
    return false;
  words.insert(words.end(), parity->begin(), parity->end());
  if (!require_literal(temporary, 0, /*equal=*/true) ||
      !load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id), 0,
                            /*equal=*/false) ||
      !load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id), 0,
                            /*equal=*/false) ||
      !append_load_u32_vgpr_at_offset(
          words, snapshot_address,
          offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_epoch_plus_one), producer_epoch,
          arch) ||
      !require_literal(producer_epoch, 0, /*equal=*/false))
    return false;
  const auto max_epoch = instrumentation::build_v_mov_b32_literal(temporary, 1024u, arch);
  if (!max_epoch)
    return false;
  words.insert(words.end(), max_epoch->begin(), max_epoch->end());
  const auto epoch_in_range =
      instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(temporary), producer_epoch, arch);
  if (!epoch_in_range)
    return false;
  words.push_back(*epoch_in_range);
  if (!narrow_vcc() ||
      !load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key), 0,
                            /*equal=*/false) ||
      !append_load_u32_vgpr_at_offset(words, snapshot_address,
                                      offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, kind),
                                      temporary, arch))
    return false;
  const auto kind_in_range =
      instrumentation::build_v_cmp_gt_u32_vcc(scalar_positive_inline_u32(3), temporary, arch);
  if (!kind_in_range)
    return false;
  words.push_back(*kind_in_range);
  if (!narrow_vcc() ||
      !require_nonzero_u64(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id)) ||
      !require_nonzero_u64(
          offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_address)) ||
      !append_load_u32_vgpr_at_offset(
          words, snapshot_address,
          offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, source_release_version), temporary,
          arch) ||
      !require_literal(temporary, 0, /*equal=*/false))
    return false;
  const auto source_parity =
      instrumentation::build_v_and_b32_literal(producer, 1u, temporary, arch);
  if (!source_parity)
    return false;
  words.insert(words.end(), source_parity->begin(), source_parity->end());
  if (!require_literal(producer, 0, /*equal=*/true) ||
      !load_require_literal(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, reserved), 0,
                            /*equal=*/true) ||
      !append_load_u32_vgpr_at_offset(words, snapshot_address,
                                      offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, version),
                                      temporary, arch))
    return false;
  const auto version_unchanged =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(version_before), temporary, arch);
  if (!version_unchanged)
    return false;
  words.push_back(*version_unchanged);
  if (!narrow_vcc() || !save_exec(ready_exec))
    return false;

  const auto valid_union = instrumentation::build_s_xor_b64(mask_a, empty_exec, ready_exec, arch);
  const auto malformed = instrumentation::build_s_andn2_b64(kRdna4ExecLo, scan_exec, mask_a, arch);
  if (!valid_union || !malformed)
    return false;
  words.push_back(*valid_union);
  words.push_back(*malformed);
  if (!set_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed) || !restore_exec(ready_exec))
    return false;

  const auto require_current_field = [&](size_t offset, uint16_t expected) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, snapshot_address, offset, temporary, arch))
      return false;
    const auto equal = instrumentation::build_v_cmp_eq_u32_vcc(expected, temporary, arch);
    if (!equal)
      return false;
    words.push_back(*equal);
    return narrow_vcc();
  };
  if (!require_current_field(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id),
                             *options.moi_dispatch_id_sgpr) ||
      !require_current_field(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, dispatch_id) +
                                 sizeof(uint32_t),
                             static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u)) ||
      !require_current_field(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, workgroup_key),
                             vector_source_vgpr(workgroup_key_vgpr)) ||
      !require_current_field(offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, consumer_owner_id),
                             vector_source_vgpr(*options.moi_owner_vgpr)) ||
      !append_load_u32_vgpr_at_offset(
          words, snapshot_address,
          offsetof(ConSanMoiInlineAcquiredEpochTokenSlot, producer_owner_id), producer, arch) ||
      !save_exec(candidate_exec))
    return false;

  const auto self_cycle = instrumentation::build_v_cmp_eq_u32_vcc(
      vector_source_vgpr(*options.moi_owner_vgpr), producer, arch);
  if (!self_cycle)
    return false;
  words.push_back(*self_cycle);
  if (!narrow_vcc() || !save_exec(mask_b) ||
      !set_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed))
    return false;
  const auto remove_self =
      instrumentation::build_s_andn2_b64(candidate_exec, candidate_exec, mask_b, arch);
  if (!remove_self)
    return false;
  words.push_back(*remove_self);

  for (uint16_t i = 0; i < 4u; ++i) {
    if (!restore_exec(candidate_exec))
      return false;
    const auto i_literal = instrumentation::build_v_mov_b32_literal(temporary, i, arch);
    const auto exists =
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(count), temporary, arch);
    if (!i_literal || !exists)
      return false;
    words.insert(words.end(), i_literal->begin(), i_literal->end());
    words.push_back(*exists);
    if (!narrow_vcc())
      return false;
    const auto duplicate = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(static_cast<uint16_t>(owners + i)), producer, arch);
    if (!duplicate)
      return false;
    words.push_back(*duplicate);
    if (!narrow_vcc() || !save_exec(mask_b) ||
        !set_flag(ConSanMoiInlineCausalSnapshotFlag::Malformed))
      return false;
    const auto remove_duplicate =
        instrumentation::build_s_andn2_b64(candidate_exec, candidate_exec, mask_b, arch);
    if (!remove_duplicate)
      return false;
    words.push_back(*remove_duplicate);
  }

  if (!restore_exec(candidate_exec))
    return false;
  const auto full =
      instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(4), count, arch);
  if (!full)
    return false;
  words.push_back(*full);
  if (!narrow_vcc() || !save_exec(mask_b) ||
      !set_flag(ConSanMoiInlineCausalSnapshotFlag::CapacityOverflow))
    return false;
  const auto remove_full =
      instrumentation::build_s_andn2_b64(candidate_exec, candidate_exec, mask_b, arch);
  if (!remove_full)
    return false;
  words.push_back(*remove_full);

  for (uint16_t i = 0; i < 4u; ++i) {
    if (!restore_exec(candidate_exec))
      return false;
    const auto destination =
        instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(i), count, arch);
    if (!destination)
      return false;
    words.push_back(*destination);
    if (!narrow_vcc())
      return false;
    words.push_back(
        build_v_mov_b32_e32(static_cast<uint16_t>(owners + i), vector_source_vgpr(producer), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(epochs + i),
                                        vector_source_vgpr(producer_epoch), arch));
  }
  if (!restore_exec(candidate_exec))
    return false;
  const auto increment_count =
      instrumentation::build_v_add_u32(count, scalar_positive_inline_u32(1), count, arch);
  if (!increment_count)
    return false;
  words.insert(words.end(), increment_count->begin(), increment_count->end());

  for (uint16_t right = 3u; right > 0u; --right) {
    if (!restore_exec(candidate_exec))
      return false;
    const auto threshold = instrumentation::build_v_mov_b32_literal(temporary, right, arch);
    const auto exists =
        instrumentation::build_v_cmp_gt_u32_vcc(vector_source_vgpr(count), temporary, arch);
    if (!threshold || !exists)
      return false;
    words.insert(words.end(), threshold->begin(), threshold->end());
    words.push_back(*exists);
    if (!narrow_vcc())
      return false;
    const auto out_of_order = instrumentation::build_v_cmp_gt_u32_vcc(
        vector_source_vgpr(static_cast<uint16_t>(owners + right - 1u)),
        static_cast<uint16_t>(owners + right), arch);
    if (!out_of_order)
      return false;
    words.push_back(*out_of_order);
    if (!narrow_vcc())
      return false;
    words.push_back(build_v_mov_b32_e32(
        producer, vector_source_vgpr(static_cast<uint16_t>(owners + right - 1u)), arch));
    words.push_back(build_v_mov_b32_e32(
        producer_epoch, vector_source_vgpr(static_cast<uint16_t>(epochs + right - 1u)), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(owners + right - 1u),
                                        vector_source_vgpr(static_cast<uint16_t>(owners + right)),
                                        arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(epochs + right - 1u),
                                        vector_source_vgpr(static_cast<uint16_t>(epochs + right)),
                                        arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(owners + right),
                                        vector_source_vgpr(producer), arch));
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(epochs + right),
                                        vector_source_vgpr(producer_epoch), arch));
  }

  if (!restore_exec(scan_exec) ||
      !append_add_literal_field(words, snapshot_address,
                                sizeof(ConSanMoiInlineAcquiredEpochTokenSlot), temporary, arch))
    return false;
  const auto next_index =
      instrumentation::build_v_add_u32(loop_index, scalar_positive_inline_u32(1), loop_index, arch);
  const auto finished = instrumentation::build_v_cmp_eq_u32_vcc(
      scalar_positive_inline_u32(token_table_capacity), loop_index, arch);
  if (!next_index || !finished)
    return false;
  words.insert(words.end(), next_index->begin(), next_index->end());
  words.push_back(*finished);
  const int64_t loop_delta =
      static_cast<int64_t>(loop_begin) - static_cast<int64_t>(words.size()) - 1;
  if (loop_delta < std::numeric_limits<int16_t>::min() ||
      loop_delta > std::numeric_limits<int16_t>::max())
    return false;
  const auto loop = instrumentation::build_s_cbranch_vccz(static_cast<int16_t>(loop_delta), arch);
  if (!loop)
    return false;
  words.push_back(*loop);

  if (!restore_exec(scan_exec) ||
      !append_inline_atomic_snapshot_address(words, snapshot_table_base, snapshot_table_capacity,
                                             atomic_address_vgpr, snapshot_address, temporary,
                                             loop_index, arch) ||
      !append_store_u32_vgpr_at_offset(words, snapshot_address,
                                       offsetof(ConSanMoiInlineCausalSnapshot, entry_count), count,
                                       arch) ||
      !append_store_u32_vgpr_at_offset(words, snapshot_address,
                                       offsetof(ConSanMoiInlineCausalSnapshot, flags), flags, arch))
    return false;
  for (uint16_t i = 0; i < 4u; ++i) {
    const size_t entry_offset = offsetof(ConSanMoiInlineCausalSnapshot, entries) +
                                i * sizeof(ConSanMoiInlineCausalSnapshotEntry);
    if (!append_store_u32_vgpr_at_offset(words, snapshot_address, entry_offset,
                                         static_cast<uint16_t>(owners + i), arch) ||
        !append_store_u32_vgpr_at_offset(words, snapshot_address, entry_offset + sizeof(uint32_t),
                                         static_cast<uint16_t>(epochs + i), arch))
      return false;
  }
  const int64_t skip_delta =
      static_cast<int64_t>(words.size()) - static_cast<int64_t>(empty_exec_skip) - 1;
  if (skip_delta < std::numeric_limits<int16_t>::min() ||
      skip_delta > std::numeric_limits<int16_t>::max())
    return false;
  words[empty_exec_skip] = build_s_branch(static_cast<int16_t>(skip_delta), arch);
  return true;
}

[[nodiscard]] bool append_inline_versioned_release_transaction(
    std::vector<uint32_t> &words, std::span<const uint8_t> bytes,
    const AtomicRecordCandidate &candidate, const ConSanMoiAtomicAddressPlan &address_plan,
    const ConSanMoiWorkgroupSources &workgroup_sources, const ConSanOptions &options,
    rj_code_arch_t arch, uint64_t release_table_base, uint32_t release_capacity,
    uint64_t snapshot_table_base, uint32_t snapshot_capacity, uint64_t token_table_base,
    uint32_t token_capacity, bool emit_guest_instruction, bool import_claimed_predecessor,
    bool predicate_on_compare_exchange_success, uint32_t &guest_instruction_offset,
    std::vector<std::string> &errors) {
  if (!options.scratch_vgpr || !options.moi_exec_save_sgpr || !options.moi_owner_vgpr ||
      !options.moi_epoch_vgpr || !options.moi_dispatch_id_sgpr ||
      address_plan.scratch_vgpr_count < 24u)
    return false;
  const uint16_t base = *options.scratch_vgpr;
  const uint16_t slot_address = base;
  const uint16_t prior_version = static_cast<uint16_t>(base + 2u);
  const uint16_t workgroup_key = static_cast<uint16_t>(base + 3u);
  // CDNA4 FLAT compare-swap requires an even-aligned new/expected pair.
  // These three registers are all recomputed after causal-snapshot capture,
  // so rotating their allocation does not extend any live range.
  const bool cdna4 = arch == ROCJITSU_CODE_ARCH_CDNA4;
  const uint16_t temporary = static_cast<uint16_t>(base + (cdna4 ? 6u : 4u));
  const uint16_t cas_new = static_cast<uint16_t>(base + (cdna4 ? 4u : 5u));
  const uint16_t cas_expected = static_cast<uint16_t>(base + (cdna4 ? 5u : 6u));
  const uint16_t exec_base = *options.moi_exec_save_sgpr;
  const uint16_t narrow_save = exec_base;
  const uint16_t empty_exec = static_cast<uint16_t>(exec_base + 2u);
  const uint16_t ready_exec = static_cast<uint16_t>(exec_base + 4u);
  const uint16_t original_exec = static_cast<uint16_t>(exec_base + 12u);
  const uint16_t claimed_exec = static_cast<uint16_t>(exec_base + 14u);
  const uint16_t committed_exec = static_cast<uint16_t>(exec_base + 16u);
  const uint16_t retry_count_sgpr = static_cast<uint16_t>(exec_base + 20u);
  // The scalar retry counter is dead once the claim loop exits. Ordinary
  // release transactions reuse its pair for the same-wave loser mask. A
  // claimed AcquireRelease instead journals its winning mask there while the
  // acquire helper reuses +14, then reconstructs the loser mask afterward.
  const uint16_t failed_exec = retry_count_sgpr;
  // append_inline_workgroup_key's +6 predicate mask is dead once the key has
  // been formed. The eligible mask dies before causal-snapshot capture, where
  // +6 is reused again as that helper's candidate mask.
  const uint16_t eligible_exec = static_cast<uint16_t>(exec_base + 6u);
  const uint16_t guest_address = address_plan.result_address_vgpr;

  const auto restore_exec = [&](uint16_t source) -> bool {
    const auto restore = instrumentation::build_s_mov_b64(kRdna4ExecLo, source, arch);
    if (!restore)
      return false;
    words.push_back(*restore);
    return true;
  };
  const auto save_exec = [&](uint16_t destination) -> bool {
    const auto save = instrumentation::build_s_mov_b64(destination, kRdna4ExecLo, arch);
    if (!save)
      return false;
    words.push_back(*save);
    return true;
  };
  const auto narrow_vcc = [&]() -> bool {
    const auto narrow = instrumentation::build_s_and_saveexec_b64(narrow_save, kRdna4VccLo, arch);
    if (!narrow)
      return false;
    words.push_back(*narrow);
    return true;
  };
  const auto require_literal = [&](uint16_t value, uint32_t literal, bool equal) -> bool {
    const auto compare = equal ? instrumentation::build_v_cmp_eq_u32_vcc(
                                     scalar_positive_inline_u32(literal), value, arch)
                               : instrumentation::build_v_cmp_ne_u32_vcc(
                                     scalar_positive_inline_u32(literal), value, arch);
    if (!compare)
      return false;
    words.push_back(*compare);
    return narrow_vcc();
  };
  const auto require_field = [&](size_t offset, uint16_t expected) -> bool {
    if (!append_load_u32_vgpr_at_offset(words, slot_address, offset, temporary, arch))
      return false;
    const auto equal = instrumentation::build_v_cmp_eq_u32_vcc(expected, temporary, arch);
    if (!equal)
      return false;
    words.push_back(*equal);
    return narrow_vcc();
  };

  if (!append_save_moi_special_state(words, options, arch) ||
      !append_inline_workgroup_key(words, workgroup_sources, options, workgroup_key,
                                   static_cast<uint16_t>(base + 6u), temporary,
                                   /*original_exec_save_offset=*/12u, arch)) {
    errors.emplace_back("ConSan MOI inline release could not initialize its EXEC transaction");
    return false;
  }
  if (predicate_on_compare_exchange_success) {
    if (!candidate.site.data_vgpr || !candidate.site.dst_vgpr) {
      errors.emplace_back("ConSan MOI inline release CAS has no dynamic outcome operands");
      return false;
    }
    const uint16_t compare_vgpr = static_cast<uint16_t>(*candidate.site.data_vgpr + 1u);
    const auto compare_succeeded = instrumentation::build_v_cmp_eq_u32_vcc(
        vector_source_vgpr(compare_vgpr), *candidate.site.dst_vgpr, arch);
    if (!compare_succeeded) {
      errors.emplace_back("ConSan MOI inline release CAS could not compare its dynamic outcome");
      return false;
    }
    words.push_back(*compare_succeeded);
    if (!narrow_vcc()) {
      errors.emplace_back("ConSan MOI inline release CAS could not predicate publication");
      return false;
    }
  }
  if (!save_exec(eligible_exec) ||
      !append_inline_atomic_slot_address(words, release_table_base, release_capacity, guest_address,
                                         slot_address, arch)) {
    errors.emplace_back("ConSan MOI inline release could not initialize its version transaction");
    return false;
  }

  // Another wave can hold the direct-mapped release slot in its odd
  // publishing state for the complete causal-snapshot capture and guest RMW.
  // Retry only when this invocation has not claimed the slot at all; if one
  // of several local lanes wins, retain the existing fail-closed treatment of
  // the other lanes rather than letting multiple publishers write one slot.
  // The critical section includes a 64-slot causal-snapshot scan. Coherent
  // version polls plus a short yield let a resident publisher complete, while
  // the finite bound still terminates on a permanently stuck odd slot.
  constexpr uint32_t kReleaseContentionRetryLimit = 4096u;
  constexpr uint16_t kScalarLiteralSource = 255u;
  words.push_back(build_s_mov_b32(retry_count_sgpr, kScalarLiteralSource, arch));
  words.push_back(kReleaseContentionRetryLimit);
  const size_t retry_begin = words.size();
  if (!append_atomic_load_u32(words, slot_address, prior_version, arch)) {
    errors.emplace_back("ConSan MOI inline release could not begin its version retry");
    return false;
  }

  if (!require_literal(prior_version, 0, /*equal=*/true) || !save_exec(empty_exec) ||
      !restore_exec(eligible_exec) || !require_literal(prior_version, 0, /*equal=*/false))
    return false;
  const auto parity = instrumentation::build_v_and_b32_literal(temporary, 1u, prior_version, arch);
  const auto terminal = instrumentation::build_v_mov_b32_literal(
      temporary, std::numeric_limits<uint32_t>::max() - 1u, arch);
  if (!parity)
    return false;
  words.insert(words.end(), parity->begin(), parity->end());
  if (!require_literal(temporary, 0, /*equal=*/true) || !terminal)
    return false;
  words.insert(words.end(), terminal->begin(), terminal->end());
  const auto below_terminal =
      instrumentation::build_v_cmp_ne_u32_vcc(vector_source_vgpr(temporary), prior_version, arch);
  if (!below_terminal)
    return false;
  words.push_back(*below_terminal);
  // This is a direct-mapped object table: any stable even record is a legal
  // replacement candidate. Identity is checked by acquire lookup, not by a
  // publisher, so stale records from earlier dispatches and ordinary hash
  // collisions cannot permanently pin a slot.
  if (!narrow_vcc() || !append_atomic_load_u32(words, slot_address, temporary, arch))
    return false;
  const auto version_unchanged =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(prior_version), temporary, arch);
  if (!version_unchanged)
    return false;
  words.push_back(*version_unchanged);
  if (!narrow_vcc() || !save_exec(ready_exec))
    return false;

  const auto valid_union =
      instrumentation::build_s_xor_b64(kRdna4ExecLo, empty_exec, ready_exec, arch);
  const auto claim_new =
      instrumentation::build_v_add_u32(cas_new, scalar_positive_inline_u32(1), prior_version, arch);
  if (!valid_union || !claim_new ||
      !append_add_literal_field(words, slot_address,
                                offsetof(ConSanMoiInlineAtomicReleaseSlot, version), temporary,
                                arch))
    return false;
  words.push_back(*valid_union);
  words.insert(words.end(), claim_new->begin(), claim_new->end());
  words.push_back(build_v_mov_b32_e32(cas_expected, vector_source_vgpr(prior_version), arch));
  const auto claim = instrumentation::build_flat_atomic_cmpswap_b32(
      slot_address, cas_new, cas_new, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  if (!claim)
    return false;
  words.insert(words.end(), claim->begin(), claim->end());
  if (!append_moi_flat_load_wait(words, arch))
    return false;
  const auto claim_won =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(cas_expected), cas_new, arch);
  if (!claim_won)
    return false;
  words.push_back(*claim_won);
  if (!narrow_vcc() || !save_exec(claimed_exec))
    return false;

  const auto select_failed_attempt =
      instrumentation::build_s_andn2_b64(committed_exec, eligible_exec, claimed_exec, arch);
  const auto inspect_claimed = instrumentation::build_s_mov_b64(kRdna4ExecLo, claimed_exec, arch);
  const auto restore_failed = instrumentation::build_s_mov_b64(kRdna4ExecLo, committed_exec, arch);
  const auto decrement_retry = instrumentation::build_s_sub_u32(
      retry_count_sgpr, retry_count_sgpr, scalar_positive_inline_u32(1), arch);
  const auto retries_remain =
      instrumentation::build_s_cmp_lg_u32(retry_count_sgpr, scalar_positive_inline_u32(0), arch);
  if (!select_failed_attempt || !inspect_claimed || !restore_failed || !decrement_retry ||
      !retries_remain)
    return false;
  words.push_back(*select_failed_attempt);
  words.push_back(*inspect_claimed);
  const size_t claimed_exit_branch = words.size();
  words.push_back(build_s_branch(/*offset_dwords=*/0, arch));
  words.push_back(*restore_failed);
  words.push_back(build_s_sleep(/*delay=*/1, arch));
  words.push_back(*decrement_retry);
  words.push_back(*retries_remain);
  const auto retries_exhausted = instrumentation::build_s_cbranch_scc0(/*simm16=*/1, arch);
  if (!retries_exhausted)
    return false;
  words.push_back(*retries_exhausted);
  const size_t retry_branch = words.size();
  const int64_t retry_delta =
      static_cast<int64_t>(retry_begin) - static_cast<int64_t>(retry_branch + 1u);
  if (retry_delta < std::numeric_limits<int16_t>::min() ||
      retry_delta > std::numeric_limits<int16_t>::max())
    return false;
  const auto retry =
      instrumentation::build_s_cbranch_execnz(static_cast<int16_t>(retry_delta), arch);
  if (!retry)
    return false;
  words.push_back(*retry);
  const int64_t claimed_exit_delta =
      static_cast<int64_t>(words.size()) - static_cast<int64_t>(claimed_exit_branch + 1u);
  if (claimed_exit_delta < std::numeric_limits<int16_t>::min() ||
      claimed_exit_delta > std::numeric_limits<int16_t>::max())
    return false;
  const auto claimed_exit =
      instrumentation::build_s_cbranch_execnz(static_cast<int16_t>(claimed_exit_delta), arch);
  if (!claimed_exit)
    return false;
  words[claimed_exit_branch] = *claimed_exit;

  if (import_claimed_predecessor) {
    if (!restore_exec(claimed_exec) || !save_exec(failed_exec))
      return false;
  } else {
    const auto preguest_failed =
        instrumentation::build_s_andn2_b64(kRdna4ExecLo, eligible_exec, claimed_exec, arch);
    if (!preguest_failed)
      return false;
    words.push_back(*preguest_failed);
    if (!save_exec(failed_exec))
      return false;
  }

  if (import_claimed_predecessor) {
    const uint16_t source_version = static_cast<uint16_t>(base + 20u);
    // The successful claim returns the authoritative prior even version in
    // cas_new. Keep the acquire source tied to that returned value instead of
    // the retry-loop temporary, whose lifetime ends at the claim boundary.
    words.push_back(build_v_mov_b32_e32(source_version, vector_source_vgpr(cas_new), arch));
    if (!restore_exec(original_exec) || !append_restore_moi_special_state(words, options, arch))
      return false;
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
      words.push_back(word);
    }
    if (!append_moi_flat_load_wait(words, arch))
      return false;
    const auto guest_store_drain = instrumentation::build_s_wait_flat_store0(arch);
    if (!guest_store_drain)
      return false;
    words.push_back(*guest_store_drain);
    // The token transaction journals its source version in scratch+8, but an
    // empty predecessor narrows EXEC before that transaction is entered. Seed
    // the same journal slot here so both the empty and nonempty paths retain
    // the claim's prior even version for the successor commit.
    words.push_back(build_v_mov_b32_e32(static_cast<uint16_t>(base + 8u),
                                        vector_source_vgpr(source_version), arch));
    if (!append_save_moi_special_state(words, options, arch) || !restore_exec(claimed_exec) ||
        !append_inline_atomic_acquire_import(
            words, options, guest_address, workgroup_key, cas_new, cas_expected, temporary,
            snapshot_table_base, snapshot_capacity, token_table_base, token_capacity,
            /*source_slot_claimed=*/true,
            /*release_sequence_only=*/
            candidate.event_kind == ConSanMoiAtomicEventKind::Release, arch, errors))
      return false;
    // Acquire import reuses the release slot-address and scratch+20 source
    // version for its token transaction. That transaction journals the source
    // version in scratch+8; recover the claim's prior even version from there
    // before snapshotting and committing the still-owned release reservation.
    // The acquire helper uses +14 for its owner predicate and token journal.
    // Recover the outer claim from +20, then rebuild the same-object loser
    // mask from the original invocation before either is needed below. Slot
    // address reconstruction uses scratch+2 as its hash temporary, so recover
    // the prior version from the token journal only after that reconstruction.
    if (!restore_exec(failed_exec) || !save_exec(claimed_exec))
      return false;
    const auto rebuild_failed =
        instrumentation::build_s_andn2_b64(kRdna4ExecLo, original_exec, claimed_exec, arch);
    if (!rebuild_failed)
      return false;
    words.push_back(*rebuild_failed);
    if (!save_exec(failed_exec) || !restore_exec(claimed_exec) ||
        !append_inline_atomic_slot_address(words, release_table_base, release_capacity,
                                           guest_address, slot_address, arch))
      return false;
    words.push_back(build_v_mov_b32_e32(
        prior_version, vector_source_vgpr(static_cast<uint16_t>(base + 8u)), arch));
  }

  if (!restore_exec(claimed_exec) ||
      !append_inline_release_causal_snapshot_capture(words, options, guest_address, workgroup_key,
                                                     token_table_base, token_capacity,
                                                     snapshot_table_base, snapshot_capacity, arch))
    return false;

  const auto epoch_mask = instrumentation::build_v_and_b32_literal(
      temporary, consan_moi_exact_shadow::max_epoch, *options.moi_epoch_vgpr, arch);
  const auto epoch_increment =
      instrumentation::build_v_add_u32(temporary, scalar_positive_inline_u32(1), temporary, arch);
  const auto epoch_saturate = instrumentation::build_v_min_u32_literal(
      temporary, consan_moi_exact_shadow::max_epoch, temporary, arch);
  if (!epoch_mask || !epoch_increment || !epoch_saturate)
    return false;
  words.insert(words.end(), epoch_mask->begin(), epoch_mask->end());
  words.insert(words.end(), epoch_increment->begin(), epoch_increment->end());
  words.insert(words.end(), epoch_saturate->begin(), epoch_saturate->end());
  if (!append_store_u32_vgpr_at_offset(words, slot_address,
                                       offsetof(ConSanMoiInlineAtomicReleaseSlot, owner_id),
                                       *options.moi_owner_vgpr, arch) ||
      !append_store_u32_vgpr_at_offset(words, slot_address,
                                       offsetof(ConSanMoiInlineAtomicReleaseSlot, epoch_plus_one),
                                       temporary, arch) ||
      !append_store_u32_vgpr_at_offset(words, slot_address,
                                       offsetof(ConSanMoiInlineAtomicReleaseSlot, workgroup_key),
                                       workgroup_key, arch) ||
      !append_store_u32_vgpr_at_offset(words, slot_address,
                                       offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                                       guest_address, arch) ||
      !append_store_u32_vgpr_at_offset(words, slot_address,
                                       offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) +
                                           sizeof(uint32_t),
                                       static_cast<uint16_t>(guest_address + 1u), arch))
    return false;
  words.push_back(build_v_mov_b32_e32(temporary, *options.moi_dispatch_id_sgpr, arch));
  if (!append_store_u32_vgpr_at_offset(words, slot_address,
                                       offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id),
                                       temporary, arch))
    return false;
  words.push_back(build_v_mov_b32_e32(
      temporary, static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u), arch));
  if (!append_store_u32_vgpr_at_offset(words, slot_address,
                                       offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id) +
                                           sizeof(uint32_t),
                                       temporary, arch))
    return false;
  const auto stage_drain = instrumentation::build_s_wait_flat_store0(arch);
  if (!stage_drain)
    return false;
  words.push_back(*stage_drain);

  // A wave can issue the same release RMW from several lanes. The direct
  // object table can publish only one representative at a time, but the
  // winning transaction represents every losing lane with the same exact
  // dispatch/workgroup/address identity. Preserve fail-closed accounting for
  // actual direct-map collisions instead of treating benign same-object
  // coalescing as detector undercoverage.
  if (!restore_exec(failed_exec) ||
      !require_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id),
                     *options.moi_dispatch_id_sgpr) ||
      !require_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, dispatch_id) + sizeof(uint32_t),
                     static_cast<uint16_t>(*options.moi_dispatch_id_sgpr + 1u)) ||
      !require_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, workgroup_key),
                     vector_source_vgpr(workgroup_key)) ||
      !require_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address),
                     vector_source_vgpr(guest_address)) ||
      !require_field(offsetof(ConSanMoiInlineAtomicReleaseSlot, atomic_address) + sizeof(uint32_t),
                     vector_source_vgpr(static_cast<uint16_t>(guest_address + 1u))) ||
      !save_exec(committed_exec))
    return false;
  const auto select_distinct_collision =
      instrumentation::build_s_andn2_b64(kRdna4ExecLo, failed_exec, committed_exec, arch);
  if (!select_distinct_collision)
    return false;
  words.push_back(*select_distinct_collision);
  if (!append_atomic_fetch_add_one_u32(words,
                                       *options.moi_report_buffer_address +
                                           offsetof(ConSanMoiReportHeader, inline_overflow_count),
                                       temporary, cas_new, arch))
    return false;
  if (!restore_exec(original_exec) || !append_restore_moi_special_state(words, options, arch))
    return false;

  if (emit_guest_instruction && !import_claimed_predecessor) {
    guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
    for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
      uint32_t word = 0;
      std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
      words.push_back(word);
    }
  }
  if (!append_moi_flat_load_wait(words, arch))
    return false;
  const auto guest_store_drain = instrumentation::build_s_wait_flat_store0(arch);
  if (!guest_store_drain)
    return false;
  words.push_back(*guest_store_drain);

  if (!append_save_moi_special_state(words, options, arch) || !restore_exec(claimed_exec))
    return false;
  const auto commit_new =
      instrumentation::build_v_add_u32(cas_new, scalar_positive_inline_u32(2), prior_version, arch);
  const auto commit_expected = instrumentation::build_v_add_u32(
      cas_expected, scalar_positive_inline_u32(1), prior_version, arch);
  if (!commit_new || !commit_expected ||
      !append_add_literal_field(words, slot_address,
                                offsetof(ConSanMoiInlineAtomicReleaseSlot, version), temporary,
                                arch))
    return false;
  words.insert(words.end(), commit_new->begin(), commit_new->end());
  words.insert(words.end(), commit_expected->begin(), commit_expected->end());
  const auto commit = instrumentation::build_flat_atomic_cmpswap_b32(
      slot_address, cas_new, cas_new, /*return_old_value=*/true, kRdna4ScopeDevice, arch);
  if (!commit)
    return false;
  words.insert(words.end(), commit->begin(), commit->end());
  if (!append_moi_flat_load_wait(words, arch))
    return false;
  const auto commit_won =
      instrumentation::build_v_cmp_eq_u32_vcc(vector_source_vgpr(cas_expected), cas_new, arch);
  if (!commit_won)
    return false;
  words.push_back(*commit_won);
  if (!narrow_vcc() || !save_exec(committed_exec))
    return false;
  const auto postguest_failed =
      instrumentation::build_s_andn2_b64(kRdna4ExecLo, claimed_exec, committed_exec, arch);
  if (!postguest_failed)
    return false;
  words.push_back(*postguest_failed);
  if (!append_atomic_fetch_add_one_u32(words,
                                       *options.moi_report_buffer_address +
                                           offsetof(ConSanMoiReportHeader, inline_overflow_count),
                                       temporary, cas_new, arch) ||
      !restore_exec(original_exec) || !append_restore_moi_special_state(words, options, arch))
    return false;
  return true;
}

[[nodiscard]] std::optional<std::vector<uint32_t>> build_inline_atomic_ordering_cave_words(
    std::span<const uint8_t> bytes, const AtomicRecordCandidate &candidate,
    const ConSanMoiAtomicAddressPlan &address_plan, const ConSanOptions &input_options,
    const VgprSpillSequence *spill, rj_code_arch_t arch, size_t inline_atomic_release_slots_offset,
    uint32_t inline_atomic_release_capacity, size_t inline_causal_snapshots_offset,
    uint32_t inline_causal_snapshot_capacity, size_t inline_acquired_token_slots_offset,
    uint32_t inline_acquired_token_capacity, uint64_t cave_text_offset, uint64_t return_text_offset,
    uint32_t &guest_instruction_offset, std::vector<std::string> &errors) {
  ConSanOptions options = input_options;
  if (!options.scratch_vgpr) {
    errors.emplace_back("ConSan MOI inline atomic patch requires RJ_CONSAN_TMP_VGPR");
    return std::nullopt;
  }
  const uint16_t scratch_count = address_plan.scratch_vgpr_count;
  const uint16_t required_scratch_count =
      inline_shadow_atomic_scratch_count(candidate.site, candidate.event_kind, arch);
  if (!address_plan.supported()) {
    errors.emplace_back("ConSan MOI inline atomic patch requires a supported address plan");
    return std::nullopt;
  }
  if (scratch_count < required_scratch_count ||
      static_cast<uint32_t>(*options.scratch_vgpr) + scratch_count > kMaxVgprs) {
    errors.emplace_back("ConSan MOI inline atomic patch has an invalid scratch VGPR window");
    return std::nullopt;
  }
  const bool scalar_persistent =
      options.moi_persistent_owner_sgpr && options.moi_persistent_epoch_sgpr;
  if ((!options.moi_owner_vgpr || !options.moi_epoch_vgpr) && !scalar_persistent) {
    errors.emplace_back("ConSan MOI inline atomic patch requires RJ_CONSAN_MOI_OWNER_VGPR and "
                        "RJ_CONSAN_MOI_EPOCH_VGPR");
    return std::nullopt;
  }
  if (!validate_inline_atomic_exec_save_sgpr(options, errors))
    return std::nullopt;
  if (reject_optional_scratch_range_overlap(options.moi_owner_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI owner", errors) ||
      reject_optional_scratch_range_overlap(options.moi_epoch_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI epoch", errors) ||
      reject_optional_scratch_range_overlap(options.moi_workgroup_key_vgpr, *options.scratch_vgpr,
                                            scratch_count, "MOI workgroup key", errors))
    return std::nullopt;
  if (candidate.site.file_offset > bytes.size() ||
      candidate.site.size > bytes.size() - candidate.site.file_offset) {
    errors.emplace_back("ConSan MOI inline atomic patch site exceeds ELF bytes");
    return std::nullopt;
  }

  const uint64_t slot_base =
      *options.moi_report_buffer_address + inline_atomic_release_slots_offset;
  const uint64_t snapshot_table_base =
      *options.moi_report_buffer_address + inline_causal_snapshots_offset;
  const uint64_t token_table_base =
      *options.moi_report_buffer_address + inline_acquired_token_slots_offset;
  const uint16_t value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 2u);
  const uint16_t workgroup_key_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 3u);
  const uint16_t producer_owner_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 4u);
  const uint16_t token_value_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 5u);
  const uint16_t temporary_vgpr = static_cast<uint16_t>(*options.scratch_vgpr + 6u);
  const uint16_t address_vgpr = address_plan.result_address_vgpr;
  const bool is_compare_exchange = candidate.site.mnemonic.find("cmpswap") != std::string::npos ||
                                   candidate.site.mnemonic.find("cmpxchg") != std::string::npos;

  std::vector<uint32_t> words;
  words.reserve(candidate.site.size / sizeof(uint32_t) + 96u +
                (spill ? spill->save_words.size() + spill->restore_words.size() : 0u));
  if (spill)
    words.insert(words.end(), spill->save_words.begin(), spill->save_words.end());
  if (address_plan.requires_materialization()) {
    const auto special_state = moi_special_state_sgprs(options);
    if (!special_state) {
      errors.emplace_back(
          "ConSan MOI inline atomic address materialization requires special-state SGPRs");
      return std::nullopt;
    }
    const auto address_words = build_consan_moi_atomic_address_materialization(
        address_plan, special_state->vcc_save_sgpr, special_state->scc_save_sgpr, arch);
    if (!address_words) {
      errors.emplace_back("ConSan MOI inline atomic patch could not materialize its address");
      return std::nullopt;
    }
    words.insert(words.end(), address_words->begin(), address_words->end());
  }
  if (scalar_persistent) {
    const uint16_t materialized_owner =
        static_cast<uint16_t>(*options.scratch_vgpr + required_scratch_count - 4u);
    const uint16_t materialized_epoch = static_cast<uint16_t>(materialized_owner + 1u);
    words.push_back(
        build_v_mov_b32_e32(materialized_owner, *options.moi_persistent_owner_sgpr, arch));
    words.push_back(
        build_v_mov_b32_e32(materialized_epoch, *options.moi_persistent_epoch_sgpr, arch));
    options.moi_owner_vgpr = materialized_owner;
    options.moi_epoch_vgpr = materialized_epoch;
  }
  if (candidate.event_kind == ConSanMoiAtomicEventKind::Release && !is_compare_exchange) {
    if (!candidate.kernel_descriptor_file_offset) {
      errors.emplace_back("ConSan MOI inline release transaction requires an owning descriptor");
      return std::nullopt;
    }
    const auto workgroup_sources = moi_descriptor_workgroup_sources(
        bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
    if (!workgroup_sources ||
        !append_inline_versioned_release_transaction(
            words, bytes, candidate, address_plan, *workgroup_sources, options, arch, slot_base,
            inline_atomic_release_capacity, snapshot_table_base, inline_causal_snapshot_capacity,
            token_table_base, inline_acquired_token_capacity,
            /*emit_guest_instruction=*/true,
            /*import_claimed_predecessor=*/is_rdna4_family_arch(arch),
            /*predicate_on_compare_exchange_success=*/false, guest_instruction_offset, errors)) {
      errors.emplace_back(
          "ConSan MOI inline release could not emit a versioned causal transaction");
      return std::nullopt;
    }
    if (spill)
      words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
    if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                              arch)) {
      errors.emplace_back("ConSan MOI inline release could not encode its return");
      return std::nullopt;
    }
    return words;
  }
  if (!candidate.kernel_descriptor_file_offset) {
    errors.emplace_back("ConSan MOI inline atomic patch requires an owning descriptor");
    return std::nullopt;
  }
  const auto workgroup_sources = moi_descriptor_workgroup_sources(
      bytes, *candidate.kernel_descriptor_file_offset, arch, errors);
  if (!workgroup_sources)
    return std::nullopt;

  // CDNA4 can let a following wave reach the guest RMW before the preceding
  // wave has published an initially empty release slot. Serialize non-CAS
  // AcquireRelease RMWs with the release reservation itself: the claimed odd
  // slot retains the predecessor payload across the guest, acquire import,
  // causal snapshot, and successor commit.
  const bool claimed_acquire_release =
      (arch == ROCJITSU_CODE_ARCH_CDNA4 || is_rdna4_family_arch(arch)) &&
      candidate.event_kind == ConSanMoiAtomicEventKind::AcquireRelease && !is_compare_exchange;
  if (claimed_acquire_release) {
    if (!append_inline_versioned_release_transaction(
            words, bytes, candidate, address_plan, *workgroup_sources, options, arch, slot_base,
            inline_atomic_release_capacity, snapshot_table_base, inline_causal_snapshot_capacity,
            token_table_base, inline_acquired_token_capacity,
            /*emit_guest_instruction=*/true,
            /*import_claimed_predecessor=*/true,
            /*predicate_on_compare_exchange_success=*/false, guest_instruction_offset, errors)) {
      errors.emplace_back(
          "ConSan MOI inline acquire-release could not emit a claimed causal transaction");
      return std::nullopt;
    }
    if (spill)
      words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
    if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                              arch)) {
      errors.emplace_back("ConSan MOI inline acquire-release could not encode its return");
      return std::nullopt;
    }
    return words;
  }

  // A successful guest acquire must observe one immutable release
  // transaction. Retain the first release version across the guest; the
  // importer rechecks it only after loading the complete release payload and
  // causal snapshot. The 24-VGPR spill-backed plan keeps this value disjoint
  // from the final address pair reserved by address materialization.
  if (candidate.event_kind == ConSanMoiAtomicEventKind::Acquire ||
      candidate.event_kind == ConSanMoiAtomicEventKind::AcquireRelease) {
    const uint16_t version_before = static_cast<uint16_t>(*options.scratch_vgpr + 20u);
    const uint16_t version_retry_count = static_cast<uint16_t>(*options.moi_exec_save_sgpr + 20u);
    const auto restore_original_exec = instrumentation::build_s_mov_b64(
        kRdna4ExecLo, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 12u), arch);
    if (!restore_original_exec || !append_save_moi_special_state(words, options, arch) ||
        !append_inline_workgroup_key(words, *workgroup_sources, options, workgroup_key_vgpr,
                                     temporary_vgpr, value_vgpr,
                                     /*original_exec_save_offset=*/12u, arch) ||
        !append_inline_atomic_slot_address(words, slot_base, inline_atomic_release_capacity,
                                           address_vgpr, *options.scratch_vgpr, arch)) {
      errors.emplace_back("ConSan MOI inline acquire could not snapshot release version");
      return std::nullopt;
    }
    // A release publisher keeps the direct-mapped slot odd while its guest
    // RMW and causal snapshot are in flight. In particular, a consumer can
    // observe the guest RMW before that publisher commits its metadata. Wait
    // for an even version before executing an acquire guest so that its saved
    // source version and a following AcquireRelease publication cannot race
    // the still-active publisher. The bounded retry preserves fail-closed
    // behavior for malformed or abandoned odd slots.
    constexpr uint32_t kAcquireSourceRetryLimit = 4096u;
    constexpr uint16_t kScalarLiteralSource = 255u;
    words.push_back(build_s_mov_b32(version_retry_count, kScalarLiteralSource, arch));
    words.push_back(kAcquireSourceRetryLimit);
    const size_t version_retry_begin = words.size();
    if (!append_atomic_load_u32(words, *options.scratch_vgpr, version_before, arch)) {
      errors.emplace_back("ConSan MOI inline acquire could not load its release version");
      return std::nullopt;
    }
    const auto version_parity =
        instrumentation::build_v_and_b32_literal(temporary_vgpr, 1u, version_before, arch);
    const auto version_even = instrumentation::build_v_cmp_eq_u32_vcc(scalar_positive_inline_u32(0),
                                                                      temporary_vgpr, arch);
    const auto decrement_retry = instrumentation::build_s_sub_u32(
        version_retry_count, version_retry_count, scalar_positive_inline_u32(1), arch);
    const auto retries_remain = instrumentation::build_s_cmp_lg_u32(
        version_retry_count, scalar_positive_inline_u32(0), arch);
    if (!version_parity || !version_even || !decrement_retry || !retries_remain) {
      errors.emplace_back("ConSan MOI inline acquire could not encode its version retry");
      return std::nullopt;
    }
    words.insert(words.end(), version_parity->begin(), version_parity->end());
    words.push_back(*version_even);
    const size_t version_even_exit = words.size();
    words.push_back(*instrumentation::build_s_cbranch_vccnz(/*simm16=*/0, arch));
    words.push_back(build_s_sleep(/*delay=*/1, arch));
    words.push_back(*decrement_retry);
    words.push_back(*retries_remain);
    const size_t version_exhausted_exit = words.size();
    words.push_back(*instrumentation::build_s_cbranch_scc0(/*simm16=*/0, arch));
    const size_t version_retry_branch = words.size();
    const int64_t retry_delta =
        static_cast<int64_t>(version_retry_begin) - static_cast<int64_t>(version_retry_branch + 1u);
    if (retry_delta < std::numeric_limits<int16_t>::min() ||
        retry_delta > std::numeric_limits<int16_t>::max()) {
      errors.emplace_back("ConSan MOI inline acquire version retry is out of branch range");
      return std::nullopt;
    }
    words.push_back(build_s_branch(static_cast<int16_t>(retry_delta), arch));
    const size_t version_retry_exit = words.size();
    const auto patch_exit = [&](size_t branch_index, bool vcc) -> bool {
      const int64_t delta =
          static_cast<int64_t>(version_retry_exit) - static_cast<int64_t>(branch_index + 1u);
      if (delta < std::numeric_limits<int16_t>::min() ||
          delta > std::numeric_limits<int16_t>::max())
        return false;
      const auto branch =
          vcc ? instrumentation::build_s_cbranch_vccnz(static_cast<int16_t>(delta), arch)
              : instrumentation::build_s_cbranch_scc0(static_cast<int16_t>(delta), arch);
      if (!branch)
        return false;
      words[branch_index] = *branch;
      return true;
    };
    if (!patch_exit(version_even_exit, /*vcc=*/true) ||
        !patch_exit(version_exhausted_exit, /*vcc=*/false)) {
      errors.emplace_back("ConSan MOI inline acquire version exit is out of branch range");
      return std::nullopt;
    }
    words.push_back(*restore_original_exec);
    if (!append_restore_moi_special_state(words, options, arch)) {
      errors.emplace_back("ConSan MOI inline acquire could not restore pre-guest state");
      return std::nullopt;
    }
  }
  guest_instruction_offset = static_cast<uint32_t>(words.size() * sizeof(uint32_t));
  for (uint64_t offset = 0; offset < candidate.site.size; offset += sizeof(uint32_t)) {
    uint32_t word = 0;
    std::memcpy(&word, bytes.data() + candidate.site.file_offset + offset, sizeof(word));
    words.push_back(word);
  }
  if (!append_moi_flat_load_wait(words, arch))
    return std::nullopt;

  if (!append_save_moi_special_state(words, options, arch) ||
      !append_inline_workgroup_key(words, *workgroup_sources, options, workgroup_key_vgpr,
                                   temporary_vgpr, value_vgpr,
                                   /*original_exec_save_offset=*/12u, arch) ||
      !append_inline_atomic_slot_address(words, slot_base, inline_atomic_release_capacity,
                                         address_vgpr, *options.scratch_vgpr, arch)) {
    errors.emplace_back("ConSan MOI inline atomic patch could not derive an address-indexed slot");
    return std::nullopt;
  }

  if ((candidate.event_kind == ConSanMoiAtomicEventKind::Acquire ||
       candidate.event_kind == ConSanMoiAtomicEventKind::AcquireRelease) &&
      !append_inline_atomic_acquire_import(words, options, address_vgpr, workgroup_key_vgpr,
                                           producer_owner_vgpr, token_value_vgpr, temporary_vgpr,
                                           snapshot_table_base, inline_causal_snapshot_capacity,
                                           token_table_base, inline_acquired_token_capacity,
                                           /*source_slot_claimed=*/false,
                                           /*release_sequence_only=*/false, arch, errors)) {
    return std::nullopt;
  }

  if (candidate.event_kind != ConSanMoiAtomicEventKind::Acquire) {
    // AcquireRelease must first import the release observed by the guest RMW;
    // all release-capable events then publish that complete ancestry through
    // the same ABI-v6 transaction. CAS publication is narrowed to lanes whose
    // dynamic compare succeeded inside the transaction, after its original
    // EXEC mask has been preserved.
    if (!append_restore_moi_special_state(words, options, arch) ||
        !append_inline_versioned_release_transaction(
            words, bytes, candidate, address_plan, *workgroup_sources, options, arch, slot_base,
            inline_atomic_release_capacity, snapshot_table_base, inline_causal_snapshot_capacity,
            token_table_base, inline_acquired_token_capacity,
            /*emit_guest_instruction=*/false,
            /*import_claimed_predecessor=*/false,
            /*predicate_on_compare_exchange_success=*/is_compare_exchange, guest_instruction_offset,
            errors)) {
      errors.emplace_back(
          "ConSan MOI inline release could not emit a versioned causal transaction");
      return std::nullopt;
    }
    if (spill)
      words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
    if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                              arch)) {
      errors.emplace_back("ConSan MOI inline release could not encode its return");
      return std::nullopt;
    }
    return words;
  }

  const auto restore_original_exec = instrumentation::build_s_mov_b64(
      kRdna4ExecLo, static_cast<uint16_t>(*options.moi_exec_save_sgpr + 12u), arch);
  if (!restore_original_exec) {
    errors.emplace_back("ConSan MOI inline atomic patch could not restore original EXEC");
    return std::nullopt;
  }
  words.push_back(*restore_original_exec);

  if (!append_restore_moi_special_state(words, options, arch)) {
    errors.emplace_back("ConSan MOI inline atomic patch could not restore VCC/SCC");
    return std::nullopt;
  }

  if (spill)
    words.insert(words.end(), spill->restore_words.begin(), spill->restore_words.end());
  if (!append_moi_direct_or_indirect_return(words, cave_text_offset, return_text_offset, options,
                                            arch)) {
    errors.emplace_back("ConSan MOI inline atomic could not encode its return");
    return std::nullopt;
  }
  return words;
}
