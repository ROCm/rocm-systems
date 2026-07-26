enum class AtomicFaultEncoding : uint8_t {
  Unsupported,
  CdnaFlat,
  FlatLike,
  Buffer,
  Ds,
};

[[nodiscard]] AtomicFaultEncoding atomic_fault_encoding(std::string_view mnemonic, uint32_t size,
                                                        rj_code_arch_t arch) {
  if ((arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4) &&
      mnemonic.starts_with("flat_atomic") && size == sizeof(cdna4::FlatMachineInst))
    return AtomicFaultEncoding::CdnaFlat;
  if ((arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250) &&
      (mnemonic.starts_with("flat_atomic") || mnemonic.starts_with("global_atomic")) &&
      size == sizeof(rdna4::VflatMachineInst))
    return AtomicFaultEncoding::FlatLike;
  if ((arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250) &&
      mnemonic.starts_with("buffer_atomic") && size == sizeof(rdna4::VbufferMachineInst))
    return AtomicFaultEncoding::Buffer;
  if ((arch == ROCJITSU_CODE_ARCH_RDNA4 || arch == ROCJITSU_CODE_ARCH_GFX1250) &&
      mnemonic.starts_with("ds_") && is_ds_atomic(mnemonic) && lds_width_bits(mnemonic) == 32u &&
      size == sizeof(rdna4::VdsMachineInst))
    return AtomicFaultEncoding::Ds;
  return AtomicFaultEncoding::Unsupported;
}

[[nodiscard]] bool atomic_fault_supports_scope(AtomicFaultEncoding encoding) {
  return encoding == AtomicFaultEncoding::FlatLike || encoding == AtomicFaultEncoding::Buffer;
}

[[nodiscard]] bool atomic_fault_supports_order(AtomicFaultEncoding encoding) {
  return encoding == AtomicFaultEncoding::CdnaFlat || encoding == AtomicFaultEncoding::FlatLike ||
         encoding == AtomicFaultEncoding::Buffer;
}

[[nodiscard]] bool atomic_fault_supports_address(AtomicFaultEncoding encoding) {
  return encoding == AtomicFaultEncoding::FlatLike || encoding == AtomicFaultEncoding::Buffer ||
         encoding == AtomicFaultEncoding::Ds;
}

template <typename Container>
void append_fault_sites_for_container(std::span<const uint8_t> bytes, std::string_view fingerprint,
                                      const Container &container, bool in_kernel,
                                      rj_code_arch_t arch, std::vector<ConSanFaultSite> &sites) {
  const std::string container_kind = in_kernel ? "kernel" : "function";
  uint32_t barrier_occurrence = 0;
  for (const ConSanBarrierSite &barrier : container.barrier_sites) {
    if (barrier.size != sizeof(uint32_t))
      continue;
    ConSanFaultSite site;
    site.kind = ConSanFaultSiteKind::Barrier;
    site.code_object_fingerprint = fingerprint;
    site.container_name = container.name;
    site.in_kernel = in_kernel;
    site.occurrence = barrier_occurrence++;
    site.text_offset = barrier.text_offset;
    site.file_offset = barrier.file_offset;
    site.size = barrier.size;
    site.mnemonic = barrier.mnemonic;
    site.semantic_role = barrier.mnemonic.find("signal") != std::string::npos ? "barrier-signal"
                         : barrier.mnemonic.find("wait") != std::string::npos ? "barrier-wait"
                                                                              : "workgroup-barrier";
    uint32_t encoding = 0;
    if (barrier.file_offset <= bytes.size() &&
        sizeof(encoding) <= bytes.size() - barrier.file_offset) {
      std::memcpy(&encoding, bytes.data() + barrier.file_offset, sizeof(encoding));
      site.decoded_operands = barrier_decoded_operands(barrier, encoding);
    } else {
      site.decoded_operands = "-";
    }
    site.identity = std::string(fingerprint) + "|" + container_kind + "=" + container.name +
                    "|kind=barrier|pc=0x" + fixed_hex(site.text_offset, 16) +
                    "|mnemonic=" + site.mnemonic + "|occurrence=" + std::to_string(site.occurrence);
    sites.push_back(std::move(site));
  }

  uint32_t atomic_occurrence = 0;
  for (const ConSanAtomicSite &atomic : container.atomic_sites) {
    if (atomic_fault_encoding(atomic.mnemonic, atomic.size, arch) ==
        AtomicFaultEncoding::Unsupported) {
      continue;
    }
    ConSanFaultSite site;
    site.kind = ConSanFaultSiteKind::Atomic;
    site.code_object_fingerprint = fingerprint;
    site.container_name = container.name;
    site.in_kernel = in_kernel;
    site.occurrence = atomic_occurrence++;
    site.text_offset = atomic.text_offset;
    site.file_offset = atomic.file_offset;
    site.size = atomic.size;
    site.width_bits = atomic.width_bits;
    site.mnemonic = atomic.mnemonic;
    site.semantic_role = atomic_semantic_role(atomic);
    site.decoded_operands = atomic_decoded_operands(atomic);
    site.identity = std::string(fingerprint) + "|" + container_kind + "=" + container.name +
                    "|kind=atomic|pc=0x" + fixed_hex(site.text_offset, 16) +
                    "|mnemonic=" + site.mnemonic + "|occurrence=" + std::to_string(site.occurrence);
    sites.push_back(std::move(site));
  }

  uint32_t ordinary_memory_occurrence = 0;
  for (const ConSanOrdinaryMemorySite &memory : container.ordinary_memory_sites) {
    ConSanFaultSite site;
    site.kind = ConSanFaultSiteKind::OrdinaryMemory;
    site.code_object_fingerprint = fingerprint;
    site.container_name = container.name;
    site.in_kernel = in_kernel;
    site.occurrence = ordinary_memory_occurrence++;
    site.text_offset = memory.text_offset;
    site.file_offset = memory.file_offset;
    site.size = memory.size;
    site.width_bits = memory.width_bits;
    site.mnemonic = memory.mnemonic;
    site.semantic_role = memory.operation == ConSanOrdinaryMemoryOperation::Load ? "ordinary-load"
                                                                                 : "ordinary-store";
    site.decoded_operands = ordinary_memory_decoded_operands(memory);
    site.ordinary_memory_support_reason = memory.support_reason;
    site.identity = std::string(fingerprint) + "|" + container_kind + "=" + container.name +
                    "|kind=ordinary-memory|pc=0x" + fixed_hex(site.text_offset, 16) +
                    "|mnemonic=" + site.mnemonic + "|occurrence=" + std::to_string(site.occurrence);
    sites.push_back(std::move(site));
  }
}

void build_fault_site_inventory(std::span<const uint8_t> bytes, rj_code_arch_t arch,
                                ConSanResult &result) {
  const std::string fingerprint = code_object_fingerprint(bytes);
  result.fault_sites.clear();
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!is_rocclr_runtime_kernel_name(kernel.name))
      append_fault_sites_for_container(bytes, fingerprint, kernel, true, arch, result.fault_sites);
  }
  for (const ConSanFunctionInfo &function : result.functions)
    append_fault_sites_for_container(bytes, fingerprint, function, false, arch, result.fault_sites);
}

[[nodiscard]] ConSanSyncAddressSource sync_address_source(ConSanAtomicAddressSpaceHint hint) {
  switch (hint) {
  case ConSanAtomicAddressSpaceHint::Lds:
    return ConSanSyncAddressSource::LdsVector;
  case ConSanAtomicAddressSpaceHint::FlatGroup:
  case ConSanAtomicAddressSpaceHint::FlatPrivate:
  case ConSanAtomicAddressSpaceHint::FlatMaybeGroup:
  case ConSanAtomicAddressSpaceHint::FlatMaybePrivate:
  case ConSanAtomicAddressSpaceHint::FlatGlobal:
  case ConSanAtomicAddressSpaceHint::FlatUnknown:
    return ConSanSyncAddressSource::FlatVector;
  case ConSanAtomicAddressSpaceHint::Global:
    return ConSanSyncAddressSource::GlobalScalarVector;
  case ConSanAtomicAddressSpaceHint::Scratch:
    return ConSanSyncAddressSource::ScratchVector;
  case ConSanAtomicAddressSpaceHint::Buffer:
    return ConSanSyncAddressSource::BufferResource;
  case ConSanAtomicAddressSpaceHint::Scalar:
  case ConSanAtomicAddressSpaceHint::Unknown:
    return ConSanSyncAddressSource::Unknown;
  }
  return ConSanSyncAddressSource::Unknown;
}

void classify_atomic_sync_confidence(const ConSanAtomicSite &site, ConSanSyncEvent &event) {
  switch (site.address_space_hint) {
  case ConSanAtomicAddressSpaceHint::FlatMaybeGroup:
  case ConSanAtomicAddressSpaceHint::FlatMaybePrivate:
    event.confidence = ConSanSemanticConfidence::Ambiguous;
    event.confidence_reason =
        "flat atomic address space is not statically distinguishable; ordering requires "
        "sequence association";
    return;
  case ConSanAtomicAddressSpaceHint::FlatUnknown:
    event.confidence = ConSanSemanticConfidence::Unsupported;
    event.confidence_reason =
        "flat atomic address space has no usable provenance; ordering requires sequence "
        "association";
    return;
  case ConSanAtomicAddressSpaceHint::Scalar:
  case ConSanAtomicAddressSpaceHint::Unknown:
    event.confidence = ConSanSemanticConfidence::Unsupported;
    event.confidence_reason =
        "atomic address encoding is not represented; ordering requires sequence association";
    return;
  default:
    event.confidence = ConSanSemanticConfidence::Conservative;
    event.confidence_reason =
        "instruction shape is decoded; ordering requires surrounding fence/cache sequence "
        "association";
    return;
  }
}

template <typename Container>
void append_sync_events_for_container(std::string_view fingerprint, const Container &container,
                                      bool in_kernel, std::vector<ConSanSyncEvent> &events) {
  const size_t container_begin = events.size();
  const std::string container_kind = in_kernel ? "kernel" : "function";
  const auto finish_event = [&](ConSanSyncEvent &event, std::string_view kind) {
    event.code_object_fingerprint = fingerprint;
    event.container_name = container.name;
    event.in_kernel = in_kernel;
    event.identity = std::string(fingerprint) + "|" + container_kind + "=" + container.name +
                     "|event=" + std::string(kind) + "|pc=0x" + fixed_hex(event.text_offset, 16) +
                     "|mnemonic=" + event.mnemonic +
                     "|occurrence=" + std::to_string(event.occurrence);
    events.push_back(std::move(event));
  };

  uint32_t barrier_occurrence = 0;
  for (const ConSanBarrierSite &site : container.barrier_sites) {
    ConSanSyncEvent event;
    event.kind = ConSanSyncEventKind::Barrier;
    event.address_source = ConSanSyncAddressSource::NotApplicable;
    event.rmw_outcome = ConSanSyncRmwOutcome::NotApplicable;
    event.barrier_id = site.barrier_id;
    event.barrier_operand_source = site.operand_source;
    event.barrier_raw_operand_selector = site.raw_operand_selector;
    event.barrier_literal_width_bits = site.literal_width_bits;
    event.barrier_literal_value = site.literal_value;
    event.barrier_raw_simm16 = site.raw_simm16;
    event.barrier_scope = site.scope;
    event.participant_count = site.participant_count;
    event.participant_mask = site.participant_mask;
    const bool lifecycle_operation = site.operation == ConSanBarrierSite::Operation::Init ||
                                     site.operation == ConSanBarrierSite::Operation::Join ||
                                     site.operation == ConSanBarrierSite::Operation::Leave ||
                                     site.operation == ConSanBarrierSite::Operation::Wakeup ||
                                     site.operation == ConSanBarrierSite::Operation::StateQuery;
    if (lifecycle_operation) {
      event.confidence = ConSanSemanticConfidence::Unsupported;
      event.confidence_reason =
          "barrier lifecycle operation is inventoried but participant semantics are unavailable";
    } else if ((site.operand_source == ConSanBarrierSite::OperandSource::Immediate ||
                site.operand_source == ConSanBarrierSite::OperandSource::Literal32) &&
               site.barrier_id && site.scope != ConSanBarrierSite::Scope::Unknown) {
      event.confidence = ConSanSemanticConfidence::Conservative;
      event.confidence_reason =
          "barrier ID and scope are decoded but participant set is unavailable";
    } else if (site.operand_source == ConSanBarrierSite::OperandSource::DynamicM0) {
      event.confidence = ConSanSemanticConfidence::Unsupported;
      event.confidence_reason = "dynamic M0 barrier ID and participant state are unavailable";
    } else {
      event.confidence = ConSanSemanticConfidence::Unsupported;
      event.confidence_reason = "barrier ID or scope is not a supported decoded form";
    }
    event.occurrence = barrier_occurrence++;
    event.text_offset = site.text_offset;
    event.file_offset = site.file_offset;
    event.size = site.size;
    event.mnemonic = site.mnemonic;
    switch (site.operation) {
    case ConSanBarrierSite::Operation::Signal:
      event.operation = ConSanSyncOperation::BarrierSignal;
      event.memory_role = ConSanSyncMemoryRole::Release;
      event.memory_role_confidence = ConSanSemanticConfidence::Conservative;
      break;
    case ConSanBarrierSite::Operation::Wait:
      event.operation = ConSanSyncOperation::BarrierWait;
      event.memory_role = ConSanSyncMemoryRole::Acquire;
      event.memory_role_confidence = ConSanSemanticConfidence::Conservative;
      break;
    case ConSanBarrierSite::Operation::Full:
      event.operation = ConSanSyncOperation::BarrierFull;
      event.memory_role = ConSanSyncMemoryRole::AcquireRelease;
      event.memory_role_confidence = ConSanSemanticConfidence::Conservative;
      break;
    case ConSanBarrierSite::Operation::Init:
      event.operation = ConSanSyncOperation::BarrierInit;
      break;
    case ConSanBarrierSite::Operation::Join:
      event.operation = ConSanSyncOperation::BarrierJoin;
      break;
    case ConSanBarrierSite::Operation::Leave:
      event.operation = ConSanSyncOperation::BarrierLeave;
      break;
    case ConSanBarrierSite::Operation::Wakeup:
      event.operation = ConSanSyncOperation::BarrierWakeup;
      break;
    case ConSanBarrierSite::Operation::StateQuery:
      event.operation = ConSanSyncOperation::BarrierStateQuery;
      break;
    case ConSanBarrierSite::Operation::Unknown:
      event.operation = ConSanSyncOperation::Unknown;
      break;
    }
    if (lifecycle_operation || site.operation == ConSanBarrierSite::Operation::Unknown) {
      event.memory_role = ConSanSyncMemoryRole::Unknown;
      event.memory_role_confidence = ConSanSemanticConfidence::Unsupported;
    }
    finish_event(event, "barrier");
  }

  uint32_t fence_occurrence = 0;
  for (const ConSanFenceSite &site : container.fence_sites) {
    ConSanSyncEvent event;
    event.kind = ConSanSyncEventKind::Fence;
    event.operation = ConSanSyncOperation::Fence;
    event.address_source = ConSanSyncAddressSource::NotApplicable;
    event.memory_role = ConSanSyncMemoryRole::Unknown;
    event.rmw_outcome = ConSanSyncRmwOutcome::NotApplicable;
    event.confidence = ConSanSemanticConfidence::Conservative;
    event.confidence_reason =
        "cache operation is decoded but its high-level memory role requires sequence association";
    event.occurrence = fence_occurrence++;
    event.text_offset = site.text_offset;
    event.file_offset = site.file_offset;
    event.size = site.size;
    event.mnemonic = site.mnemonic;
    finish_event(event, "fence");
  }

  uint32_t ordinary_memory_occurrence = 0;
  for (const ConSanOrdinaryMemorySite &site : container.ordinary_memory_sites) {
    // Unsupported ordinary-memory records remain available through the fault
    // inventory, but cannot become synchronization events or sequences.
    if (site.support_reason != ConSanOrdinaryMemorySupportReason::Supported &&
        site.support_reason != ConSanOrdinaryMemorySupportReason::SupportedSynchronizationOnly)
      continue;
    ConSanSyncEvent event;
    event.kind = ConSanSyncEventKind::OrdinaryMemory;
    event.operation = site.operation == ConSanOrdinaryMemoryOperation::Load
                          ? ConSanSyncOperation::OrdinaryLoad
                          : ConSanSyncOperation::OrdinaryStore;
    event.address_source =
        site.mnemonic.starts_with("global_")
            ? ConSanSyncAddressSource::GlobalScalarVector
            : (site.mnemonic.starts_with("buffer_") ? ConSanSyncAddressSource::BufferResource
                                                    : ConSanSyncAddressSource::FlatVector);
    event.memory_role = ConSanSyncMemoryRole::Unknown;
    event.rmw_outcome = ConSanSyncRmwOutcome::NotApplicable;
    event.confidence = ConSanSemanticConfidence::Conservative;
    event.confidence_reason = "bounded ordinary-memory encoding is decoded; ordering requires "
                              "same-block cache-sequence association";
    event.occurrence = ordinary_memory_occurrence++;
    event.text_offset = site.text_offset;
    event.file_offset = site.file_offset;
    event.size = site.size;
    event.width_bits = site.width_bits;
    event.mnemonic = site.mnemonic;
    if (site.raw_ioffset)
      event.static_byte_offset = *site.raw_ioffset;
    event.raw_scope = site.raw_scope;
    finish_event(event, "ordinary-memory");
  }

  uint32_t atomic_occurrence = 0;
  for (const ConSanAtomicSite &site : container.atomic_sites) {
    ConSanSyncEvent event;
    event.kind = ConSanSyncEventKind::Atomic;
    const bool is_compare_exchange = consan_atomic_is_compare_exchange(site);
    event.operation = is_compare_exchange ? ConSanSyncOperation::AtomicCompareExchange
                                          : ConSanSyncOperation::AtomicRmw;
    event.address_source = sync_address_source(site.address_space_hint);
    event.memory_role = ConSanSyncMemoryRole::Unknown;
    if (is_compare_exchange) {
      event.rmw_outcome = ConSanSyncRmwOutcome::CompareExchange;
    } else if (site.returns_old_value) {
      event.rmw_outcome = *site.returns_old_value ? ConSanSyncRmwOutcome::ReturnsOldValue
                                                  : ConSanSyncRmwOutcome::NoReturn;
    } else {
      event.rmw_outcome = ConSanSyncRmwOutcome::Unknown;
    }
    classify_atomic_sync_confidence(site, event);
    event.occurrence = atomic_occurrence++;
    event.text_offset = site.text_offset;
    event.file_offset = site.file_offset;
    event.size = site.size;
    event.width_bits = site.width_bits;
    event.mnemonic = site.mnemonic;
    if (site.raw_ioffset)
      event.static_byte_offset = *site.raw_ioffset;
    event.raw_scope = site.raw_scope;
    finish_event(event, "atomic");
  }

  std::stable_sort(events.begin() + static_cast<std::ptrdiff_t>(container_begin), events.end(),
                   [](const ConSanSyncEvent &lhs, const ConSanSyncEvent &rhs) {
                     return lhs.text_offset < rhs.text_offset;
                   });
}

void build_sync_event_inventory(std::span<const uint8_t> bytes, ConSanResult &result) {
  const std::string fingerprint = code_object_fingerprint(bytes);
  result.sync_events.clear();
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!is_rocclr_runtime_kernel_name(kernel.name))
      append_sync_events_for_container(fingerprint, kernel, true, result.sync_events);
  }
  for (const ConSanFunctionInfo &function : result.functions)
    append_sync_events_for_container(fingerprint, function, false, result.sync_events);
}

[[nodiscard]] bool sync_event_semantics_equal(const ConSanSyncEvent &lhs,
                                              const ConSanSyncEvent &rhs) {
  return std::tie(lhs.kind, lhs.operation, lhs.address_source, lhs.memory_role, lhs.rmw_outcome,
                  lhs.confidence, lhs.memory_role_confidence, lhs.confidence_reason,
                  lhs.code_object_fingerprint, lhs.in_kernel, lhs.occurrence, lhs.text_offset,
                  lhs.file_offset, lhs.size, lhs.width_bits, lhs.mnemonic, lhs.static_byte_offset,
                  lhs.raw_scope, lhs.barrier_id, lhs.barrier_operand_source,
                  lhs.barrier_raw_operand_selector, lhs.barrier_literal_width_bits,
                  lhs.barrier_literal_value, lhs.barrier_raw_simm16, lhs.barrier_scope,
                  lhs.participant_count, lhs.participant_mask) ==
         std::tie(rhs.kind, rhs.operation, rhs.address_source, rhs.memory_role, rhs.rmw_outcome,
                  rhs.confidence, rhs.memory_role_confidence, rhs.confidence_reason,
                  rhs.code_object_fingerprint, rhs.in_kernel, rhs.occurrence, rhs.text_offset,
                  rhs.file_offset, rhs.size, rhs.width_bits, rhs.mnemonic, rhs.static_byte_offset,
                  rhs.raw_scope, rhs.barrier_id, rhs.barrier_operand_source,
                  rhs.barrier_raw_operand_selector, rhs.barrier_literal_width_bits,
                  rhs.barrier_literal_value, rhs.barrier_raw_simm16, rhs.barrier_scope,
                  rhs.participant_count, rhs.participant_mask);
}

[[nodiscard]] bool canonicalize_sync_events_by_physical_site(ConSanResult &result) {
  std::unordered_map<std::string_view, uint64_t> kernel_entries;
  std::unordered_map<std::string_view, uint64_t> function_entries;
  kernel_entries.reserve(result.kernels.size());
  function_entries.reserve(result.functions.size());
  for (const ConSanKernelInfo &kernel : result.kernels)
    kernel_entries.emplace(kernel.name, kernel.entry_text_offset);
  for (const ConSanFunctionInfo &function : result.functions)
    function_entries.emplace(function.name, function.entry_text_offset);
  const auto container_entry = [&](const ConSanSyncEvent &event) -> std::optional<uint64_t> {
    const auto &entries = event.in_kernel ? kernel_entries : function_entries;
    const auto entry = entries.find(event.container_name);
    return entry == entries.end() ? std::nullopt : std::optional<uint64_t>(entry->second);
  };

  std::unordered_map<uint64_t, size_t> canonical_index_by_file_offset;
  canonical_index_by_file_offset.reserve(result.sync_events.size());
  std::vector<ConSanSyncEvent> canonical_events;
  canonical_events.reserve(result.sync_events.size());
  for (ConSanSyncEvent &event : result.sync_events) {
    const auto [entry, inserted] =
        canonical_index_by_file_offset.emplace(event.file_offset, canonical_events.size());
    if (inserted) {
      canonical_events.push_back(std::move(event));
      continue;
    }
    const ConSanSyncEvent &canonical = canonical_events[entry->second];
    if (container_entry(canonical) != container_entry(event) ||
        !sync_event_semantics_equal(canonical, event)) {
      result.errors.emplace_back(
          "ConSan synchronization site at file offset " + std::to_string(event.file_offset) +
          " was decoded inconsistently through aliases '" + canonical.container_name + "' and '" +
          event.container_name + "'");
      return false;
    }
  }
  result.sync_events = std::move(canonical_events);
  return true;
}

[[nodiscard]] ConSanSyncSequenceKind sync_sequence_kind(ConSanSyncEventKind kind) {
  switch (kind) {
  case ConSanSyncEventKind::Barrier:
    return ConSanSyncSequenceKind::Barrier;
  case ConSanSyncEventKind::Fence:
    return ConSanSyncSequenceKind::Fence;
  case ConSanSyncEventKind::Atomic:
    return ConSanSyncSequenceKind::Atomic;
  case ConSanSyncEventKind::OrdinaryMemory:
    return ConSanSyncSequenceKind::OrdinaryMemory;
  }
  return ConSanSyncSequenceKind::Fence;
}

[[nodiscard]] std::vector<std::unique_ptr<BasicBlock>>
build_sync_basic_blocks(const AmdGpuCodeObject &code_object, Decoder &decoder, rj_code_arch_t arch,
                        const ConSanResult &result) {
  std::vector<uint64_t> leaders;
  leaders.reserve(result.kernels.size() + result.functions.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.has_text_range)
      leaders.push_back(kernel.entry_text_offset);
  }
  for (const ConSanFunctionInfo &function : result.functions)
    leaders.push_back(function.entry_text_offset);
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());

  std::vector<BasicBlock::CodeRange> code_ranges;
  code_ranges.reserve(code_object.functions().size());
  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    if (function.code_size != 0)
      code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
  }
  return BasicBlock::build(code_object, decoder, arch, leaders, code_ranges);
}

[[nodiscard]] std::vector<std::unique_ptr<BasicBlock>>
build_original_proof_basic_blocks(const AmdGpuCodeObject &code_object, Decoder &decoder,
                                  rj_code_arch_t arch) {
  std::vector<uint64_t> leaders;
  leaders.reserve(code_object.kernels().size() + code_object.functions().size());
  for (const AmdGpuKernelInfo &kernel : code_object.kernels()) {
    if (kernel.has_text_range)
      leaders.push_back(kernel.entry_text_offset);
  }
  for (const AmdGpuFunctionInfo &function : code_object.functions())
    leaders.push_back(function.entry_text_offset);
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());

  std::vector<BasicBlock::CodeRange> code_ranges;
  code_ranges.reserve(code_object.functions().size());
  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    if (function.code_size != 0)
      code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
  }
  return BasicBlock::build(code_object, decoder, arch, leaders, code_ranges);
}

[[nodiscard]] bool block_is_in_cycle(const BasicBlock *start) {
  if (start == nullptr)
    return false;
  std::vector<const BasicBlock *> pending(start->successors().begin(), start->successors().end());
  std::unordered_set<const BasicBlock *> visited;
  while (!pending.empty()) {
    const BasicBlock *block = pending.back();
    pending.pop_back();
    if (block == start)
      return true;
    if (block == nullptr || !visited.insert(block).second)
      continue;
    pending.insert(pending.end(), block->successors().begin(), block->successors().end());
  }
  return false;
}

[[nodiscard]] bool instruction_is_inside_scalar_clause(BasicBlock &block, uint64_t text_offset) {
  uint32_t clause_remaining = 0;
  for (const Instruction &instruction : block.instructions()) {
    const bool inside_clause = clause_remaining > 0;
    if (clause_remaining > 0)
      --clause_remaining;
    if (instruction.src_loc() == text_offset)
      return inside_clause;
    if (is_s_clause(instruction))
      clause_remaining = s_clause_following_instruction_count(instruction);
  }
  return false;
}

void build_singleton_sync_sequences(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks, ConSanResult &result) {
  const BlockOffsetIndex block_offset_index = build_block_offset_index(blocks);
  std::unordered_map<const BasicBlock *, uint32_t> block_indices;
  block_indices.reserve(blocks.size());
  for (size_t index = 0; index < blocks.size(); ++index) {
    if (blocks[index] != nullptr)
      block_indices.emplace(blocks[index].get(), static_cast<uint32_t>(index));
  }
  std::vector<std::optional<bool>> cyclic_block_cache(blocks.size());
  result.sync_sequences.clear();
  result.sync_sequences.reserve(result.sync_events.size());
  for (const ConSanSyncEvent &event : result.sync_events) {
    ConSanSyncSequence sequence;
    sequence.kind = sync_sequence_kind(event.kind);
    sequence.operation = event.operation;
    sequence.address_source = event.address_source;
    sequence.memory_role = event.memory_role;
    sequence.rmw_outcome = event.rmw_outcome;
    sequence.confidence = event.confidence;
    sequence.memory_role_confidence = event.memory_role_confidence;
    sequence.confidence_reason = event.confidence_reason;
    sequence.identity = event.identity + "|sequence=singleton";
    sequence.container_name = event.container_name;
    sequence.in_kernel = event.in_kernel;
    sequence.begin_text_offset = event.text_offset;
    sequence.end_text_offset = event.text_offset + event.size;
    sequence.member_event_identities.push_back(event.identity);
    sequence.width_bits = event.width_bits;
    sequence.static_byte_offset = event.static_byte_offset;
    sequence.raw_scope = event.raw_scope;
    sequence.barrier_id = event.barrier_id;
    sequence.barrier_operand_source = event.barrier_operand_source;
    sequence.barrier_raw_operand_selector = event.barrier_raw_operand_selector;
    sequence.barrier_literal_width_bits = event.barrier_literal_width_bits;
    sequence.barrier_literal_value = event.barrier_literal_value;
    sequence.barrier_raw_simm16 = event.barrier_raw_simm16;
    sequence.barrier_scope = event.barrier_scope;
    sequence.participant_count = event.participant_count;
    sequence.participant_mask = event.participant_mask;

    BasicBlock *block = block_for_offset(block_offset_index, event.text_offset);
    if (block != nullptr) {
      const auto instruction =
          std::ranges::find_if(block->instructions(), [&](const Instruction &candidate) {
            return candidate.src_loc() == event.text_offset &&
                   static_cast<uint32_t>(candidate.size()) == event.size;
          });
      if (instruction != block->instructions().end()) {
        const auto block_index = block_indices.find(block);
        if (block_index != block_indices.end()) {
          sequence.basic_block_index = block_index->second;
          std::optional<bool> &in_cycle = cyclic_block_cache[block_index->second];
          if (!in_cycle)
            in_cycle = block_is_in_cycle(block);
          sequence.in_cyclic_cfg_component = *in_cycle;
          sequence.inside_scalar_clause =
              instruction_is_inside_scalar_clause(*block, event.text_offset);
        }
      }
    }
    if (!sequence.basic_block_index) {
      sequence.confidence = combine_consan_sync_confidence(sequence.confidence,
                                                           ConSanSemanticConfidence::Unsupported);
      if (!sequence.confidence_reason.empty())
        sequence.confidence_reason += "; ";
      sequence.confidence_reason += "event is not an exact instruction in a decoded basic block";
    }
    result.sync_sequences.push_back(std::move(sequence));
  }
}

[[nodiscard]] bool compatible_barrier_participants(const ConSanSyncSequence &signal,
                                                   const ConSanSyncSequence &wait) {
  const auto is_static_id = [](ConSanBarrierSite::OperandSource source) {
    return source == ConSanBarrierSite::OperandSource::Immediate ||
           source == ConSanBarrierSite::OperandSource::Literal32;
  };
  if (!is_static_id(signal.barrier_operand_source) || !is_static_id(wait.barrier_operand_source) ||
      !signal.barrier_id || !wait.barrier_id || signal.barrier_id != wait.barrier_id ||
      signal.barrier_scope == ConSanBarrierSite::Scope::Unknown ||
      wait.barrier_scope == ConSanBarrierSite::Scope::Unknown ||
      signal.barrier_scope != wait.barrier_scope) {
    return false;
  }
  if (signal.participant_count && wait.participant_count &&
      signal.participant_count != wait.participant_count) {
    return false;
  }
  if (signal.participant_mask && wait.participant_mask &&
      signal.participant_mask != wait.participant_mask) {
    return false;
  }
  return true;
}

[[nodiscard]] bool is_bounded_barrier_bookkeeping(const Instruction &instruction) {
  const std::string_view mnemonic = instruction.mnemonic();
  // These are exactly the four pointer/index bookkeeping instructions in the
  // retained RDNA4 hip-moi barrier handoff. Keep this list closed: accepting
  // arbitrary move/add families would include special-register and EXEC
  // updates that can change which lanes execute the wait.
  return mnemonic == "v_add_nc_u32_e32" || mnemonic == "s_add_nc_u64" || mnemonic == "s_mov_b32";
}

[[nodiscard]] bool compatible_barrier_pair_identity(const ConSanSyncSequence &signal,
                                                    const ConSanSyncSequence &wait) {
  return signal.kind == ConSanSyncSequenceKind::Barrier &&
         signal.operation == ConSanSyncOperation::BarrierSignal &&
         wait.kind == ConSanSyncSequenceKind::Barrier &&
         wait.operation == ConSanSyncOperation::BarrierWait && signal.in_kernel == wait.in_kernel &&
         signal.container_name == wait.container_name && signal.basic_block_index &&
         wait.basic_block_index && !signal.inside_scalar_clause && !wait.inside_scalar_clause &&
         compatible_barrier_participants(signal, wait);
}

[[nodiscard]] bool compatible_barrier_pair_metadata(const ConSanSyncSequence &signal,
                                                    const ConSanSyncSequence &wait) {
  return compatible_barrier_pair_identity(signal, wait) &&
         signal.basic_block_index == wait.basic_block_index;
}

[[nodiscard]] bool
is_safe_bounded_barrier_pair(const ConSanSyncSequence &signal, const ConSanSyncSequence &wait,
                             const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                             bool allow_extended_pair) {
  constexpr uint32_t kMaximumBookkeepingInstructions = 4u;
  constexpr uint64_t kMaximumBookkeepingBytes = 24u;
  // Exact whole-pair fault injection and Sampled sequence metadata may span
  // the useful work between a signal and its completing wait. Keep that
  // relaxation local to those consumers: other synchronization semantics
  // retain the closed bookkeeping pattern above.
  // Extended consumers already require the signal and wait to be consecutive
  // synchronization events in one exact decoded basic block, with the same
  // static ID/scope and no call edge or intervening barrier.  Those structural
  // checks prove the pair independently of how much straight-line useful work
  // the compiler schedules between its members.  Do not impose a corpus-sized
  // byte/instruction cap on that otherwise exact shape.
  const uint32_t maximum_instructions =
      allow_extended_pair ? std::numeric_limits<uint32_t>::max() : kMaximumBookkeepingInstructions;
  const uint64_t maximum_bytes =
      allow_extended_pair ? std::numeric_limits<uint64_t>::max() : kMaximumBookkeepingBytes;
  if (!compatible_barrier_pair_metadata(signal, wait) ||
      signal.end_text_offset > wait.begin_text_offset ||
      wait.begin_text_offset - signal.end_text_offset > maximum_bytes) {
    return false;
  }
  const size_t block_index = *signal.basic_block_index;
  if (block_index >= blocks.size() || blocks[block_index] == nullptr ||
      !blocks[block_index]->call_edges().empty()) {
    return false;
  }
  uint64_t expected_offset = signal.end_text_offset;
  uint32_t instruction_count = 0;
  for (const Instruction &instruction : blocks[block_index]->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset < signal.end_text_offset)
      continue;
    if (offset >= wait.begin_text_offset)
      break;
    if (offset != expected_offset || instruction.size() <= 0 ||
        ++instruction_count > maximum_instructions || is_barrier_instruction(instruction) ||
        (!allow_extended_pair && !is_bounded_barrier_bookkeeping(instruction))) {
      return false;
    }
    expected_offset += static_cast<uint64_t>(instruction.size());
  }
  return expected_offset == wait.begin_text_offset;
}

[[nodiscard]] bool
is_safe_cluster_triangle_barrier_pair(const ConSanSyncSequence &signal,
                                      const ConSanSyncSequence &wait,
                                      const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  // Cluster arrive lowers to a conditional diamond: one path executes the
  // signal and the other bypasses it, then both paths join at the completing
  // wait. Admit only that exact local CFG shape. This is deliberately narrower
  // than general cross-block pairing: the signal block must be the complete
  // fallthrough arm, and the join may contain only a short straight-line prefix
  // before the wait.
  constexpr uint32_t kMaximumJoinPrefixInstructions = 4u;
  constexpr uint64_t kMaximumJoinPrefixBytes = 24u;
  if (!compatible_barrier_pair_identity(signal, wait) ||
      signal.barrier_scope != ConSanBarrierSite::Scope::Cluster ||
      signal.basic_block_index == wait.basic_block_index ||
      signal.end_text_offset > wait.begin_text_offset ||
      wait.begin_text_offset - signal.end_text_offset > kMaximumJoinPrefixBytes) {
    return false;
  }

  const size_t signal_index = *signal.basic_block_index;
  const size_t wait_index = *wait.basic_block_index;
  if (signal_index >= blocks.size() || wait_index >= blocks.size() ||
      blocks[signal_index] == nullptr || blocks[wait_index] == nullptr) {
    return false;
  }
  BasicBlock *signal_block = blocks[signal_index].get();
  BasicBlock *wait_block = blocks[wait_index].get();
  if (!signal_block->call_edges().empty() || !wait_block->call_edges().empty() ||
      signal_block->start_offset() != signal.begin_text_offset ||
      signal_block->end_offset() != signal.end_text_offset ||
      wait_block->start_offset() > wait.begin_text_offset ||
      signal_block->predecessors().size() != 1u || signal_block->successors().size() != 1u ||
      signal_block->successors().front() != wait_block) {
    return false;
  }

  BasicBlock *guard = signal_block->predecessors().front();
  if (guard == nullptr || !guard->call_edges().empty() || guard->successors().size() != 2u ||
      std::ranges::find(guard->successors(), signal_block) == guard->successors().end() ||
      std::ranges::find(guard->successors(), wait_block) == guard->successors().end() ||
      wait_block->predecessors().size() != 2u ||
      std::ranges::find(wait_block->predecessors(), guard) == wait_block->predecessors().end() ||
      std::ranges::find(wait_block->predecessors(), signal_block) ==
          wait_block->predecessors().end()) {
    return false;
  }
  const Instruction *guard_terminator = guard->terminator();
  if (guard_terminator == nullptr || (guard_terminator->flags() & COND_BRANCH) == 0u) {
    return false;
  }

  uint64_t expected_offset = wait_block->start_offset();
  uint32_t instruction_count = 0;
  for (const Instruction &instruction : wait_block->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset >= wait.begin_text_offset)
      break;
    if (offset != expected_offset || instruction.size() <= 0 ||
        ++instruction_count > kMaximumJoinPrefixInstructions ||
        (instruction.flags() &
         (BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR)) != 0u ||
        is_barrier_instruction(instruction)) {
      return false;
    }
    expected_offset += static_cast<uint64_t>(instruction.size());
  }
  return expected_offset == wait.begin_text_offset;
}

void associate_barrier_sync_sequences(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks, bool allow_extended_pairs,
    ConSanResult &result) {
  const auto has_unresolved_preceding_signal = [&](size_t signal_index) {
    for (size_t cursor = signal_index; cursor > 0; --cursor) {
      const ConSanSyncSequence &prior = result.sync_sequences[cursor - 1u];
      if (prior.kind != ConSanSyncSequenceKind::Barrier)
        continue;
      if (prior.operation == ConSanSyncOperation::BarrierSignal) {
        const ConSanSyncSequence &signal = result.sync_sequences[signal_index];
        return prior.in_kernel == signal.in_kernel &&
               prior.container_name == signal.container_name &&
               prior.basic_block_index == signal.basic_block_index;
      }
      // Any non-signal barrier terminates this ambiguity search. In
      // particular, a proven init/join lifecycle may immediately precede its
      // completing signal/wait pair.
      return false;
    }
    return false;
  };
  std::vector<bool> unresolved_preceding_signal(result.sync_sequences.size(), false);
  for (size_t index = 0; index < result.sync_sequences.size(); ++index)
    unresolved_preceding_signal[index] = has_unresolved_preceding_signal(index);
  std::vector<ConSanSyncSequence> associated;
  associated.reserve(result.sync_sequences.size());
  for (size_t i = 0; i < result.sync_sequences.size();) {
    if (i + 1 < result.sync_sequences.size() &&
        (is_safe_bounded_barrier_pair(result.sync_sequences[i], result.sync_sequences[i + 1],
                                      blocks, allow_extended_pairs) ||
         is_safe_cluster_triangle_barrier_pair(result.sync_sequences[i],
                                               result.sync_sequences[i + 1], blocks)) &&
        !unresolved_preceding_signal[i]) {
      ConSanSyncSequence sequence = std::move(result.sync_sequences[i]);
      ConSanSyncSequence &wait = result.sync_sequences[i + 1];
      sequence.operation = ConSanSyncOperation::BarrierFull;
      sequence.memory_role = ConSanSyncMemoryRole::AcquireRelease;
      sequence.confidence = combine_consan_sync_confidence(sequence.confidence, wait.confidence);
      sequence.memory_role_confidence = combine_consan_sync_confidence(
          sequence.memory_role_confidence, wait.memory_role_confidence);
      const bool cross_block = sequence.basic_block_index != wait.basic_block_index;
      sequence.confidence_reason =
          cross_block            ? "bounded cluster barrier signal/wait pair in a "
                                   "single conditional CFG triangle with matching decoded "
                                   "ID and scope; participant set is unavailable"
          : allow_extended_pairs ? "bounded same-owner same-block barrier signal/wait pair "
                                   "with matching decoded ID and scope, no intervening "
                                   "barrier, and bounded intervening instruction run; "
                                   "participant set is unavailable"
                                 : "bounded same-block barrier signal/wait pair with "
                                   "matching decoded ID and scope and only the retained "
                                   "RDNA4 register bookkeeping pattern; participant set "
                                   "is unavailable";
      sequence.identity += "|associated-with=" + wait.identity;
      sequence.end_text_offset = wait.end_text_offset;
      sequence.in_cyclic_cfg_component |= wait.in_cyclic_cfg_component;
      sequence.inside_scalar_clause |= wait.inside_scalar_clause;
      sequence.member_event_identities.insert(sequence.member_event_identities.end(),
                                              wait.member_event_identities.begin(),
                                              wait.member_event_identities.end());
      if (!sequence.participant_count)
        sequence.participant_count = wait.participant_count;
      if (!sequence.participant_mask)
        sequence.participant_mask = wait.participant_mask;
      associated.push_back(std::move(sequence));
      i += 2;
      continue;
    }

    ConSanSyncSequence sequence = std::move(result.sync_sequences[i]);
    if (sequence.kind == ConSanSyncSequenceKind::Barrier &&
        (sequence.operation == ConSanSyncOperation::BarrierSignal ||
         sequence.operation == ConSanSyncOperation::BarrierWait)) {
      sequence.confidence =
          combine_consan_sync_confidence(sequence.confidence, ConSanSemanticConfidence::Ambiguous);
      sequence.confidence_reason =
          "barrier signal/wait component has no safe bounded same-block partner";
    }
    associated.push_back(std::move(sequence));
    ++i;
  }
  result.sync_sequences = std::move(associated);
}

void build_barrier_lifecycle_group_inventory(ConSanResult &result) {
  result.barrier_lifecycle_groups.clear();
  for (size_t begin = 0; begin < result.sync_sequences.size(); ++begin) {
    const ConSanSyncSequence &init = result.sync_sequences[begin];
    if (init.operation != ConSanSyncOperation::BarrierInit)
      continue;

    ConSanBarrierLifecycleGroup group;
    group.identity = init.identity + "|lifecycle-group=static-run";
    group.container_name = init.container_name;
    group.in_kernel = init.in_kernel;
    group.basic_block_index = init.basic_block_index;
    group.begin_text_offset = init.begin_text_offset;
    group.end_text_offset = init.end_text_offset;
    group.member_event_identities = init.member_event_identities;
    group.barrier_id = init.barrier_id;
    group.barrier_scope = init.barrier_scope;

    const auto append_member = [&](const ConSanSyncSequence &sequence) {
      group.end_text_offset = sequence.end_text_offset;
      group.member_event_identities.insert(group.member_event_identities.end(),
                                           sequence.member_event_identities.begin(),
                                           sequence.member_event_identities.end());
    };
    const auto is_contiguous = [&](const ConSanSyncSequence &sequence) {
      return group.basic_block_index && sequence.basic_block_index == group.basic_block_index &&
             sequence.container_name == group.container_name &&
             sequence.in_kernel == group.in_kernel &&
             sequence.begin_text_offset == group.end_text_offset;
    };
    const auto has_matching_static_id = [&](const ConSanSyncSequence &sequence) {
      return group.barrier_id && sequence.barrier_id == group.barrier_id &&
             group.barrier_scope != ConSanBarrierSite::Scope::Unknown &&
             sequence.barrier_scope == group.barrier_scope &&
             (sequence.barrier_operand_source == ConSanBarrierSite::OperandSource::Immediate ||
              sequence.barrier_operand_source == ConSanBarrierSite::OperandSource::Literal32);
    };

    if (!group.basic_block_index || !group.barrier_id ||
        group.barrier_scope == ConSanBarrierSite::Scope::Unknown ||
        (init.barrier_operand_source != ConSanBarrierSite::OperandSource::Immediate &&
         init.barrier_operand_source != ConSanBarrierSite::OperandSource::StaticM0Literal32 &&
         init.barrier_operand_source != ConSanBarrierSite::OperandSource::Literal32)) {
      group.rejection_reason = "lifecycle init has no proven static ID and scope";
      result.barrier_lifecycle_groups.push_back(std::move(group));
      continue;
    }

    size_t cursor = begin + 1u;
    size_t join_count = 0;
    bool mismatch = false;
    while (cursor < result.sync_sequences.size() &&
           result.sync_sequences[cursor].operation == ConSanSyncOperation::BarrierJoin) {
      const ConSanSyncSequence &join = result.sync_sequences[cursor];
      if (!is_contiguous(join)) {
        group.rejection_reason = "lifecycle run crosses a block, container, or instruction gap";
        mismatch = true;
        break;
      }
      append_member(join);
      if (!has_matching_static_id(join)) {
        group.rejection_reason = "lifecycle members do not have one matching static ID and scope";
        mismatch = true;
        break;
      }
      ++join_count;
      ++cursor;
    }
    if (mismatch) {
      result.barrier_lifecycle_groups.push_back(std::move(group));
      continue;
    }
    if (join_count == 0) {
      group.rejection_reason = "lifecycle leave has no preceding matching static join association";
      result.barrier_lifecycle_groups.push_back(std::move(group));
      continue;
    }
    if (cursor >= result.sync_sequences.size() ||
        result.sync_sequences[cursor].operation != ConSanSyncOperation::BarrierFull ||
        !is_contiguous(result.sync_sequences[cursor])) {
      group.rejection_reason = "lifecycle run has no contiguous same-block completing barrier pair";
      result.barrier_lifecycle_groups.push_back(std::move(group));
      continue;
    }
    const ConSanSyncSequence &barrier = result.sync_sequences[cursor++];
    append_member(barrier);
    if (!has_matching_static_id(barrier)) {
      group.rejection_reason = "lifecycle members do not have one matching static ID and scope";
      result.barrier_lifecycle_groups.push_back(std::move(group));
      continue;
    }
    if (cursor >= result.sync_sequences.size() ||
        result.sync_sequences[cursor].operation != ConSanSyncOperation::BarrierLeave ||
        !is_contiguous(result.sync_sequences[cursor])) {
      group.rejection_reason = "lifecycle run has no contiguous same-block leave operation";
      result.barrier_lifecycle_groups.push_back(std::move(group));
      continue;
    }
    const ConSanSyncSequence &leave = result.sync_sequences[cursor];
    append_member(leave);
    // LLVM's GFX12 ISA model defines S_BARRIER_LEAVE as a zero-operand SOPP
    // with a fixed-zero simm16. Its execution-synchronization specification
    // defines it as dropping the last named barrier joined by this thread.
    // Therefore the matching static join, rather than the fixed encoding,
    // supplies the leave association for this lifecycle.
    if (leave.end_text_offset - leave.begin_text_offset != sizeof(uint32_t) ||
        leave.barrier_raw_simm16 != 0u) {
      group.rejection_reason = "lifecycle leave is not the fixed-zero GFX12 encoding";
      result.barrier_lifecycle_groups.push_back(std::move(group));
      continue;
    }
    group.admissible = true;
    group.confidence = ConSanSemanticConfidence::Conservative;
    result.barrier_lifecycle_groups.push_back(std::move(group));
  }
}

using SyncEventIdentityIndex =
    std::unordered_map<std::string_view, const ConSanSyncEvent *>;

[[nodiscard]] SyncEventIdentityIndex build_sync_event_identity_index(const ConSanResult &result) {
  SyncEventIdentityIndex index;
  index.reserve(result.sync_events.size());
  for (const ConSanSyncEvent &event : result.sync_events)
    index.emplace(event.identity, &event);
  return index;
}

[[nodiscard]] const ConSanSyncEvent *
find_sync_event(const SyncEventIdentityIndex &index, std::string_view identity) {
  const auto event = index.find(identity);
  return event == index.end() ? nullptr : event->second;
}

[[nodiscard]] const ConSanSyncEvent *
first_sequence_event(const SyncEventIdentityIndex &events,
                     const ConSanSyncSequence &sequence);

[[nodiscard]] bool is_relocatable_barrier_move_destination(const Instruction &instruction,
                                                           uint64_t offset,
                                                           std::span<const uint8_t> text,
                                                           rj_code_arch_t arch,
                                                           std::string *error_out) {
  const int size = instruction.size();
  if (size == 4 || size == 8)
    return is_relocatable_anchor(instruction, offset, text, arch, error_out);
  if (offset % sizeof(uint32_t) != 0 || size <= 0 || size % sizeof(uint32_t) != 0) {
    *error_out = "wide destination must have a positive dword-aligned size and offset";
    return false;
  }
  const uint64_t size_u = static_cast<uint64_t>(size);
  if (offset > text.size() || size_u > text.size() - offset ||
      instruction.raw_encoding() == nullptr) {
    *error_out = "wide destination encoding is unavailable or exceeds .text";
    return false;
  }
  constexpr uint64_t kControlFlowFlags =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  if ((instruction.flags() & kControlFlowFlags) || instruction.branch_offset_bytes() ||
      is_s_clause(instruction)) {
    *error_out = "wide destination has control-flow or clause semantics";
    return false;
  }
  return true;
}

[[nodiscard]] bool is_sc_relay_reservoir_instruction(const Instruction &instruction,
                                                     uint64_t offset, std::span<const uint8_t> text,
                                                     rj_code_arch_t arch) {
  const int size = instruction.size();
  const std::string_view mnemonic = instruction.mnemonic();
  if (size <= 0 || size % static_cast<int>(sizeof(uint32_t)) != 0 ||
      instruction.raw_encoding() == nullptr || offset % sizeof(uint32_t) != 0 ||
      mnemonic.starts_with("ds_") || mnemonic == "s_clause" || mnemonic == "s_delay_alu")
    return false;
  if (size == static_cast<int>(sizeof(uint32_t)) || size == 2 * static_cast<int>(sizeof(uint32_t)))
    return is_relocatable_anchor(instruction, offset, text, arch);
  constexpr uint64_t kControlFlowFlags =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  return size == 3 * static_cast<int>(sizeof(uint32_t)) &&
         (mnemonic.starts_with("flat_load") || mnemonic.starts_with("flat_store")) &&
         !(instruction.flags() & kControlFlowFlags) && !instruction.branch_offset_bytes();
}

struct StructuredExecDiamondProof {
  uint32_t guard_block_index = 0;
  uint32_t destination_block_index = 0;
  uint32_t source_block_index = 0;
  uint64_t guard_offset = 0;
  uint64_t destination_offset = 0;
  uint64_t source_offset = 0;
};

[[nodiscard]] bool block_reaches(const BasicBlock *begin, const BasicBlock *target,
                                 std::unordered_set<const BasicBlock *> &visited) {
  if (begin == target)
    return true;
  if (begin == nullptr || !visited.insert(begin).second)
    return false;
  return std::ranges::any_of(begin->successors(), [&](const BasicBlock *successor) {
    return block_reaches(successor, target, visited);
  });
}

[[nodiscard]] bool is_matching_exec_restore(const Instruction &restore,
                                            const Instruction &saveexec) {
  if (restore.mnemonic() != std::string_view("s_mov_b64") ||
      saveexec.mnemonic().find("_saveexec_") == std::string_view::npos ||
      restore.raw_encoding() == nullptr || saveexec.raw_encoding() == nullptr) {
    return false;
  }
  const uint32_t restore_word = restore.raw_encoding()[0];
  const uint32_t save_word = saveexec.raw_encoding()[0];
  constexpr uint32_t kExecLo = 126;
  const uint32_t restore_destination = (restore_word >> 16u) & 0x7fu;
  const uint32_t restore_source = restore_word & 0xffu;
  const uint32_t saved_destination = (save_word >> 16u) & 0x7fu;
  return restore_destination == kExecLo && restore_source == saved_destination;
}

[[nodiscard]] bool is_matching_cmpx_exec_restore(const Instruction &restore,
                                                 const Instruction &save) {
  if (restore.mnemonic() != std::string_view("s_or_b32") ||
      save.mnemonic() != std::string_view("s_mov_b32") || restore.raw_encoding() == nullptr ||
      save.raw_encoding() == nullptr) {
    return false;
  }
  const uint32_t restore_word = restore.raw_encoding()[0];
  const uint32_t save_word = save.raw_encoding()[0];
  constexpr uint32_t kExecLo = 126;
  const uint32_t restore_destination = (restore_word >> 16u) & 0x7fu;
  const uint32_t restore_source0 = restore_word & 0xffu;
  const uint32_t restore_source1 = (restore_word >> 8u) & 0xffu;
  const uint32_t saved_destination = (save_word >> 16u) & 0x7fu;
  const uint32_t saved_source = save_word & 0xffu;
  return restore_destination == kExecLo && restore_source0 == kExecLo &&
         restore_source1 == saved_destination && saved_source == kExecLo;
}

[[nodiscard]] std::optional<StructuredExecDiamondProof>
prove_structured_exec_diamond(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                              uint32_t source_index, uint32_t destination_index,
                              uint64_t barrier_begin) {
  if (source_index >= blocks.size() || destination_index >= blocks.size() ||
      source_index == destination_index) {
    return std::nullopt;
  }
  BasicBlock *source = blocks[source_index].get();
  BasicBlock *destination = blocks[destination_index].get();
  if (source == nullptr || destination == nullptr || source->num_instructions() == 0u ||
      destination->successors().size() != 1u || destination->successors().front() != source ||
      !destination->call_edges().empty()) {
    return std::nullopt;
  }
  const Instruction &restore = *source->instructions().begin();
  if (restore.src_loc() != source->start_offset() ||
      restore.size() != static_cast<int>(sizeof(uint32_t)) ||
      barrier_begin != source->start_offset() + sizeof(uint32_t)) {
    return std::nullopt;
  }

  std::vector<BasicBlock *> guards;
  for (BasicBlock *candidate : destination->predecessors()) {
    if (candidate != nullptr &&
        std::ranges::find(source->predecessors(), candidate) != source->predecessors().end()) {
      guards.push_back(candidate);
    }
  }
  if (guards.size() != 1u || source->predecessors().size() != 2u ||
      destination->predecessors().size() != 1u) {
    return std::nullopt;
  }
  BasicBlock *guard = guards.front();
  if (guard->successors().size() != 2u || !guard->call_edges().empty() ||
      std::ranges::find(guard->successors(), source) == guard->successors().end() ||
      std::ranges::find(guard->successors(), destination) == guard->successors().end() ||
      guard->num_instructions() < 2u) {
    return std::nullopt;
  }
  const Instruction *terminator = guard->terminator();
  if (terminator == nullptr || (terminator->mnemonic() != std::string_view("s_cbranch_execz") &&
                                terminator->mnemonic() != std::string_view("s_cbranch_execnz"))) {
    return std::nullopt;
  }
  const Instruction *before_narrow = nullptr;
  const Instruction *narrow = nullptr;
  for (const Instruction &instruction : guard->instructions()) {
    if (&instruction == terminator)
      break;
    before_narrow = narrow;
    narrow = &instruction;
  }
  if (narrow == nullptr || narrow->src_loc() + narrow->size() != terminator->src_loc()) {
    return std::nullopt;
  }
  const bool matched_saveexec = is_matching_exec_restore(restore, *narrow);
  const bool matched_cmpx = narrow->mnemonic().starts_with("v_cmpx_") && before_narrow != nullptr &&
                            before_narrow->src_loc() + before_narrow->size() == narrow->src_loc() &&
                            is_matching_cmpx_exec_restore(restore, *before_narrow);
  if (!matched_saveexec && !matched_cmpx) {
    return std::nullopt;
  }

  std::unordered_set<const BasicBlock *> visited;
  if (block_reaches(source, guard, visited))
    return std::nullopt;
  visited.clear();
  if (block_reaches(source, destination, visited))
    return std::nullopt;

  const auto guard_it = std::ranges::find(blocks, guard, &std::unique_ptr<BasicBlock>::get);
  if (guard_it == blocks.end())
    return std::nullopt;
  return StructuredExecDiamondProof{
      .guard_block_index = static_cast<uint32_t>(guard_it - blocks.begin()),
      .destination_block_index = destination_index,
      .source_block_index = source_index,
      .guard_offset = terminator->src_loc(),
      .destination_offset = destination->start_offset(),
      .source_offset = source->start_offset(),
  };
}

/// Prove the narrow completing cross-block contract used by conditional
/// placement controls. The destination is in a two-way conditional guard, the
/// pair starts the reconverged source block, and each arm is a single acyclic
/// block whose only successor is that source. Thus every dynamic execution
/// that reached the original pair first executes the destination guard exactly
/// once, independent of which arm is selected.
[[nodiscard]] std::optional<StructuredExecDiamondProof>
prove_completing_structured_diamond(const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                                    uint32_t source_index, uint32_t destination_index,
                                    uint64_t barrier_begin, uint64_t destination_offset) {
  if (source_index >= blocks.size() || destination_index >= blocks.size() ||
      source_index == destination_index) {
    return std::nullopt;
  }
  BasicBlock *source = blocks[source_index].get();
  BasicBlock *guard = blocks[destination_index].get();
  if (source == nullptr || guard == nullptr || barrier_begin != source->start_offset() ||
      destination_offset < guard->start_offset() || destination_offset >= guard->end_offset() ||
      source->predecessors().size() != 2u || guard->successors().size() != 2u ||
      !source->call_edges().empty() || !guard->call_edges().empty()) {
    return std::nullopt;
  }
  const Instruction *terminator = guard->terminator();
  if (terminator == nullptr || !(terminator->flags() & COND_BRANCH) ||
      terminator->branch_offset_bytes() == 0) {
    return std::nullopt;
  }

  std::unordered_set<const BasicBlock *> expected_predecessors;
  for (BasicBlock *arm : guard->successors()) {
    if (arm == nullptr || arm == source || arm == guard || !arm->call_edges().empty() ||
        arm->predecessors().size() != 1u || arm->predecessors().front() != guard ||
        arm->successors().size() != 1u || arm->successors().front() != source) {
      return std::nullopt;
    }
    expected_predecessors.insert(arm);
  }
  if (expected_predecessors.size() != 2u ||
      !std::ranges::all_of(source->predecessors(), [&](const BasicBlock *predecessor) {
        return expected_predecessors.contains(predecessor);
      })) {
    return std::nullopt;
  }
  std::unordered_set<const BasicBlock *> visited;
  if (block_reaches(source, guard, visited))
    return std::nullopt;

  return StructuredExecDiamondProof{
      .guard_block_index = destination_index,
      .destination_block_index = destination_index,
      .source_block_index = source_index,
      .guard_offset = terminator->src_loc(),
      .destination_offset = destination_offset,
      .source_offset = source->start_offset(),
  };
}

void build_barrier_move_destination_inventory(
    const AmdGpuCodeObject &code_object,
    const std::vector<std::unique_ptr<BasicBlock>> &blocks,
    const SyncEventIdentityIndex &events, rj_code_arch_t arch, ConSanResult &result) {
  result.barrier_move_destinations.clear();
  if (code_object.text_sections().size() != 1u)
    return;
  const Section *text_section = code_object.text_sections().front();
  const auto text = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(text_section->data()), text_section->size());
  std::unordered_set<std::string> seen;

  for (const ConSanSyncSequence &sequence : result.sync_sequences) {
    if (sequence.kind != ConSanSyncSequenceKind::Barrier ||
        sequence.operation != ConSanSyncOperation::BarrierFull ||
        !consan_sync_confidence_meets(sequence.confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        sequence.member_event_identities.size() != 2u) {
      continue;
    }
    const ConSanSyncEvent *event = first_sequence_event(events, sequence);
    if (event == nullptr)
      continue;

    uint64_t container_begin = 0;
    uint64_t container_end = 0;
    uint64_t text_file_offset = 0;
    bool found_container = false;
    if (sequence.in_kernel) {
      const auto container =
          std::ranges::find(result.kernels, sequence.container_name, &ConSanKernelInfo::name);
      if (container != result.kernels.end() && container->has_text_range) {
        container_begin = container->entry_text_offset;
        container_end = container_begin + container->code_size;
        text_file_offset = container->text_file_offset;
        found_container = true;
      }
    } else {
      const auto container =
          std::ranges::find(result.functions, sequence.container_name, &ConSanFunctionInfo::name);
      if (container != result.functions.end()) {
        container_begin = container->entry_text_offset;
        container_end = container_begin + container->code_size;
        text_file_offset = container->text_file_offset;
        found_container = true;
      }
    }
    if (!found_container)
      continue;

    for (size_t block_index = 0; block_index < blocks.size(); ++block_index) {
      BasicBlock *block = blocks[block_index].get();
      if (block == nullptr || block->start_offset() < container_begin ||
          block->start_offset() >= container_end) {
        continue;
      }
      uint32_t clause_remaining = 0;
      for (const Instruction &instruction : block->instructions()) {
        const bool inside_clause = clause_remaining > 0;
        if (clause_remaining > 0)
          --clause_remaining;
        if (is_s_clause(instruction))
          clause_remaining = s_clause_following_instruction_count(instruction);
        if (instruction.src_loc() < container_begin || instruction.src_loc() >= container_end)
          continue;

        ConSanBarrierMoveDestination destination;
        destination.code_object_fingerprint = event->code_object_fingerprint;
        destination.container_name = sequence.container_name;
        destination.in_kernel = sequence.in_kernel;
        destination.basic_block_index = static_cast<uint32_t>(block_index);
        destination.text_offset = instruction.src_loc();
        destination.file_offset = text_file_offset + instruction.src_loc();
        destination.size = static_cast<uint32_t>(instruction.size());
        destination.mnemonic = std::string(instruction.mnemonic());
        destination.memory_operation = instruction.is_memory_op();
        destination.identity =
            event->code_object_fingerprint + "|" +
            std::string(sequence.in_kernel ? "kernel=" : "function=") + sequence.container_name +
            "|destination=instruction|pc=0x" + fixed_hex(destination.text_offset, 16) +
            "|mnemonic=" + destination.mnemonic + "|size=" + std::to_string(destination.size);
        if (!seen.insert(destination.identity).second)
          continue;

        std::string relocatable_error;
        if (inside_clause) {
          destination.rejection_reason = "inside-s-clause";
        } else if (is_barrier_instruction(instruction)) {
          destination.rejection_reason = "barrier-source-or-lifecycle";
        } else if (is_fence_like(instruction.mnemonic())) {
          destination.rejection_reason = "fence-operation";
        } else if (!instruction.is_memory_op()) {
          destination.rejection_reason = "not-memory-operation";
        } else if (!is_relocatable_barrier_move_destination(instruction, instruction.src_loc(),
                                                            text, arch, &relocatable_error)) {
          destination.rejection_reason = "not-relocatable:" + relocatable_error;
        } else {
          destination.suitable = true;
        }
        if (destination.suitable && sequence.basic_block_index &&
            destination.basic_block_index != *sequence.basic_block_index) {
          if (const auto proof = prove_completing_structured_diamond(
                  blocks, *sequence.basic_block_index, destination.basic_block_index,
                  sequence.begin_text_offset, destination.text_offset)) {
            destination.cfg_contract = ConSanBarrierMoveCfgContract::CompletingStructuredDiamond;
            destination.structured_guard_block_index = proof->guard_block_index;
            destination.structured_source_block_index = proof->source_block_index;
            destination.structured_guard_offset = proof->guard_offset;
            destination.structured_source_offset = proof->source_offset;
          } else if (const auto proof = prove_structured_exec_diamond(
                         blocks, *sequence.basic_block_index, destination.basic_block_index,
                         sequence.begin_text_offset)) {
            destination.cfg_contract =
                ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond;
            destination.structured_guard_block_index = proof->guard_block_index;
            destination.structured_source_block_index = proof->source_block_index;
            destination.structured_guard_offset = proof->guard_offset;
            destination.structured_source_offset = proof->source_offset;
          }
        }
        result.barrier_move_destinations.push_back(std::move(destination));
      }
    }
  }
}

[[nodiscard]] bool admits_cross_block_barrier_move(const ConSanBarrierMoveDestination &destination,
                                                   const ConSanOptions &options) {
  if (options.fault_barrier_move_direction != ConSanBarrierMoveDirection::Earlier ||
      !destination.structured_guard_block_index || !destination.structured_source_block_index ||
      !destination.structured_guard_offset || !destination.structured_source_offset) {
    return false;
  }
  switch (destination.cfg_contract) {
  case ConSanBarrierMoveCfgContract::SameBlock:
    return false;
  case ConSanBarrierMoveCfgContract::CompletingStructuredDiamond:
    return options.fault_allow_completing_conditional_barrier_move;
  case ConSanBarrierMoveCfgContract::DestructiveStructuredExecDiamond:
    return options.fault_allow_destructive_divergent_barrier_move;
  }
  return false;
}

[[nodiscard]] const ConSanSyncEvent *
first_sequence_event(const SyncEventIdentityIndex &events,
                     const ConSanSyncSequence &sequence) {
  if (sequence.member_event_identities.empty())
    return nullptr;
  return find_sync_event(events, sequence.member_event_identities.front());
}

[[nodiscard]] bool is_release_cache_event(const ConSanSyncEvent *event) {
  return event != nullptr && (event->mnemonic == "global_wb" || event->mnemonic == "buffer_wb" ||
                              event->mnemonic == "buffer_wbl2");
}

[[nodiscard]] bool is_acquire_cache_event(const ConSanSyncEvent *event) {
  return event != nullptr && (event->mnemonic == "global_inv" || event->mnemonic == "buffer_inv" ||
                              event->mnemonic == "s_dcache_inv");
}

[[nodiscard]] bool has_only_waits_between(const ConSanSyncSequence &before,
                                          const ConSanSyncSequence &after,
                                          const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  if (!before.basic_block_index || before.basic_block_index != after.basic_block_index ||
      before.container_name != after.container_name || before.in_kernel != after.in_kernel ||
      before.end_text_offset > after.begin_text_offset ||
      *before.basic_block_index >= blocks.size()) {
    return false;
  }
  BasicBlock *block = blocks[*before.basic_block_index].get();
  if (block == nullptr)
    return false;
  uint64_t expected_offset = before.end_text_offset;
  for (const Instruction &instruction : block->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset < before.end_text_offset)
      continue;
    if (offset >= after.begin_text_offset)
      break;
    if (offset != expected_offset || !instruction.mnemonic().starts_with("s_wait"))
      return false;
    expected_offset += static_cast<uint64_t>(instruction.size());
  }
  return expected_offset == after.begin_text_offset;
}

enum class BoundedAcquirePathKind { SameBlock, ExactFallthroughJoin, ExactSelfLoopExit };

[[nodiscard]] bool is_bounded_acquire_bookkeeping(const Instruction &instruction) {
  constexpr uint64_t kControlFlowFlags =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  return instruction.size() > 0 &&
         (!instruction.is_memory_op() || instruction.mnemonic().starts_with("s_wait")) &&
         !is_barrier_instruction(instruction) && !is_fence_like(instruction.mnemonic()) &&
         !is_s_clause(instruction) && (instruction.flags() & kControlFlowFlags) == 0u &&
         !instruction.branch_offset_bytes();
}

[[nodiscard]] bool scan_bounded_acquire_bookkeeping(BasicBlock &block, uint64_t begin, uint64_t end,
                                                    bool &saw_wait) {
  uint64_t expected_offset = begin;
  for (const Instruction &instruction : block.instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset < begin)
      continue;
    if (offset >= end)
      break;
    if (offset != expected_offset || !is_bounded_acquire_bookkeeping(instruction))
      return false;
    saw_wait |= instruction.mnemonic().starts_with("s_wait");
    expected_offset += static_cast<uint64_t>(instruction.size());
  }
  return expected_offset == end;
}

[[nodiscard]] std::optional<BoundedAcquirePathKind>
prove_bounded_acquire_path(const ConSanSyncSequence &before, const ConSanSyncSequence &after,
                           const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                           bool require_wait_for_same_block = false) {
  constexpr uint64_t kMaximumBookkeepingBytes = 64u;
  constexpr uint64_t kMaximumPollingLoopPrefixBytes = 128u;
  if (!before.basic_block_index || !after.basic_block_index ||
      before.container_name != after.container_name || before.in_kernel != after.in_kernel ||
      before.end_text_offset > after.begin_text_offset ||
      after.begin_text_offset - before.end_text_offset > kMaximumBookkeepingBytes ||
      before.inside_scalar_clause || after.inside_scalar_clause ||
      *before.basic_block_index >= blocks.size() || *after.basic_block_index >= blocks.size()) {
    return std::nullopt;
  }
  BasicBlock *source = blocks[*before.basic_block_index].get();
  BasicBlock *boundary = blocks[*after.basic_block_index].get();
  if (source == nullptr || boundary == nullptr || !source->call_edges().empty() ||
      !boundary->call_edges().empty()) {
    return std::nullopt;
  }

  bool saw_wait = false;
  if (source == boundary) {
    if (!scan_bounded_acquire_bookkeeping(*source, before.end_text_offset, after.begin_text_offset,
                                          saw_wait) ||
        (!saw_wait &&
         (require_wait_for_same_block || before.end_text_offset != after.begin_text_offset))) {
      return std::nullopt;
    }
    return BoundedAcquirePathKind::SameBlock;
  }

  // Cache operations can be direct branch targets even when the atomic path
  // reaches them by ordinary fallthrough. This is the shape emitted for an
  // acquire fence after a conditionally executed RMW: an inactive path may
  // enter at the cache operation, while the active RMW path has one exact,
  // bounded, non-branching route to it. Prove that route locally rather than
  // requiring the join block to have a unique predecessor. Other predecessors
  // do not invalidate the atomic-to-cache edge established by this fallthrough
  // path.
  if (!source->has_terminator() && source->end_offset() == boundary->start_offset() &&
      source->successors().size() == 1u && source->successors().front() == boundary) {
    if (!scan_bounded_acquire_bookkeeping(*source, before.end_text_offset, source->end_offset(),
                                          saw_wait) ||
        !scan_bounded_acquire_bookkeeping(*boundary, boundary->start_offset(),
                                          after.begin_text_offset, saw_wait) ||
        !saw_wait) {
      return std::nullopt;
    }
    return BoundedAcquirePathKind::ExactFallthroughJoin;
  }

  // Admit one exact compiler-lowered polling loop: the communication event is
  // the loop header, the conditional terminator either returns to that header
  // or falls through to the unique boundary block, and no other block can
  // enter the boundary. This is deliberately narrower than general dominance.
  const Instruction *terminator = source->terminator();
  if (terminator == nullptr || (terminator->flags() & COND_BRANCH) == 0u ||
      !terminator->branch_offset_bytes() || before.begin_text_offset < source->start_offset() ||
      before.begin_text_offset - source->start_offset() > kMaximumPollingLoopPrefixBytes ||
      source->end_offset() != boundary->start_offset() || source->successors().size() != 2u ||
      std::ranges::find(source->successors(), source) == source->successors().end() ||
      std::ranges::find(source->successors(), boundary) == source->successors().end() ||
      boundary->predecessors().size() != 1u || boundary->predecessors().front() != source) {
    return std::nullopt;
  }
  const int64_t branch_target = static_cast<int64_t>(source->end_offset()) +
                                static_cast<int64_t>(*terminator->branch_offset_bytes());
  if (branch_target < 0 || static_cast<uint64_t>(branch_target) != source->start_offset() ||
      terminator->src_loc() < before.end_text_offset ||
      terminator->src_loc() + static_cast<uint64_t>(terminator->size()) != source->end_offset()) {
    return std::nullopt;
  }
  bool prefix_wait = false;
  if (!scan_bounded_acquire_bookkeeping(*source, source->start_offset(), before.begin_text_offset,
                                        prefix_wait) ||
      !scan_bounded_acquire_bookkeeping(*source, before.end_text_offset, terminator->src_loc(),
                                        saw_wait)) {
    return std::nullopt;
  }
  saw_wait |= terminator->mnemonic().starts_with("s_wait");
  if (!scan_bounded_acquire_bookkeeping(*boundary, boundary->start_offset(),
                                        after.begin_text_offset, saw_wait) ||
      !saw_wait) {
    return std::nullopt;
  }
  return BoundedAcquirePathKind::ExactSelfLoopExit;
}

[[nodiscard]] ConSanSyncSequence
associate_atomic_cache_sequence(const ConSanSyncSequence *release, const ConSanSyncSequence &atomic,
                                const ConSanSyncSequence *acquire) {
  ConSanSyncSequence sequence = atomic;
  sequence.memory_role = release != nullptr && acquire != nullptr
                             ? ConSanSyncMemoryRole::AcquireRelease
                         : release != nullptr ? ConSanSyncMemoryRole::Release
                                              : ConSanSyncMemoryRole::Acquire;
  sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
  // A standalone flat atomic with no static address-space provenance is kept
  // unsupported because its ordering cannot be inferred from the instruction
  // alone. Reaching this helper supplies that missing evidence through an
  // exact release pattern or a bounded acquire-side compiler-bookkeeping
  // proof. Preserve the decoded address source for the later provenance gate,
  // but let the proven sequence carry conservative semantic confidence.
  sequence.confidence = ConSanSemanticConfidence::Conservative;
  sequence.confidence_reason = acquire != nullptr
                                   ? "bounded atomic/acquire-cache compiler-bookkeeping proof"
                                   : "same-block RDNA4 release-cache/atomic waits-only proof";
  std::vector<std::string> members;
  if (release != nullptr) {
    sequence.identity += "|release-cache=" + release->identity;
    sequence.begin_text_offset = release->begin_text_offset;
    sequence.in_cyclic_cfg_component |= release->in_cyclic_cfg_component;
    sequence.inside_scalar_clause |= release->inside_scalar_clause;
    members.insert(members.end(), release->member_event_identities.begin(),
                   release->member_event_identities.end());
  }
  members.insert(members.end(), atomic.member_event_identities.begin(),
                 atomic.member_event_identities.end());
  if (acquire != nullptr) {
    sequence.identity += "|acquire-cache=" + acquire->identity;
    sequence.end_text_offset = acquire->end_text_offset;
    sequence.in_cyclic_cfg_component |= acquire->in_cyclic_cfg_component;
    sequence.inside_scalar_clause |= acquire->inside_scalar_clause;
    members.insert(members.end(), acquire->member_event_identities.begin(),
                   acquire->member_event_identities.end());
  }
  sequence.member_event_identities = std::move(members);
  return sequence;
}

[[nodiscard]] std::optional<uint64_t>
atomic_release_wait_boundary(const ConSanSyncSequence &atomic,
                             const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                             rj_code_arch_t arch) {
  if (atomic.kind != ConSanSyncSequenceKind::Atomic || !atomic.basic_block_index ||
      *atomic.basic_block_index >= blocks.size()) {
    return std::nullopt;
  }
  const auto expected_combined = build_s_wait_storecnt_dscnt0(arch);
  const auto expected_store = build_s_wait_storecnt0(arch);
  BasicBlock *block = blocks[*atomic.basic_block_index].get();
  if (!expected_combined || !expected_store || block == nullptr)
    return std::nullopt;
  std::vector<const Instruction *> preceding;
  for (const Instruction &instruction : block->instructions()) {
    if (instruction.src_loc() >= atomic.begin_text_offset)
      break;
    preceding.push_back(&instruction);
  }
  uint64_t suffix_begin = atomic.begin_text_offset;
  for (auto instruction = preceding.rbegin(); instruction != preceding.rend(); ++instruction) {
    const int size = (*instruction)->size();
    if (size <= 0 || (*instruction)->src_loc() + static_cast<uint64_t>(size) != suffix_begin ||
        !(*instruction)->mnemonic().starts_with("s_wait"))
      break;
    suffix_begin = (*instruction)->src_loc();
    if (size == static_cast<int>(sizeof(uint32_t)) && (*instruction)->raw_encoding() != nullptr &&
        ((*instruction)->raw_encoding()[0] == *expected_combined ||
         (*instruction)->raw_encoding()[0] == *expected_store))
      return (*instruction)->src_loc();
  }
  return std::nullopt;
}

void associate_atomic_sync_sequences(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks, rj_code_arch_t arch,
    const SyncEventIdentityIndex &events, ConSanResult &result) {
  std::vector<ConSanSyncSequence> associated;
  associated.reserve(result.sync_sequences.size());
  for (size_t i = 0; i < result.sync_sequences.size();) {
    const ConSanSyncSequence &current = result.sync_sequences[i];
    if (current.kind == ConSanSyncSequenceKind::Fence &&
        is_release_cache_event(first_sequence_event(events, current)) &&
        i + 1 < result.sync_sequences.size()) {
      const ConSanSyncSequence &atomic = result.sync_sequences[i + 1];
      if (atomic.kind == ConSanSyncSequenceKind::Atomic &&
          has_only_waits_between(current, atomic, blocks)) {
        const ConSanSyncSequence *acquire = nullptr;
        if (i + 2 < result.sync_sequences.size()) {
          const ConSanSyncSequence &candidate = result.sync_sequences[i + 2];
          if (candidate.kind == ConSanSyncSequenceKind::Fence &&
              is_acquire_cache_event(first_sequence_event(events, candidate)) &&
              prove_bounded_acquire_path(atomic, candidate, blocks)) {
            acquire = &candidate;
          }
        }
        associated.push_back(associate_atomic_cache_sequence(&current, atomic, acquire));
        i += acquire != nullptr ? 3 : 2;
        continue;
      }
    }
    if (current.kind == ConSanSyncSequenceKind::Atomic && i + 1 < result.sync_sequences.size()) {
      const ConSanSyncSequence &acquire = result.sync_sequences[i + 1];
      if (acquire.kind == ConSanSyncSequenceKind::Fence &&
          is_acquire_cache_event(first_sequence_event(events, acquire)) &&
          prove_bounded_acquire_path(current, acquire, blocks)) {
        associated.push_back(associate_atomic_cache_sequence(nullptr, current, &acquire));
        i += 2;
        continue;
      }
    }
    associated.push_back(current);
    ++i;
  }
  result.sync_sequences = std::move(associated);

  // gfx12 lowers the release half of an atomic to an immediately preceding
  // store-count-zero wait, either combined or inside a contiguous suffix of
  // counter-specific waits, without a separate global_wb event. This applies
  // both to release-only no-return atomics and to returning acquire-release
  // atomics followed by an acquire invalidate. Admit only that exact,
  // same-block encoding. The wait is a boundary of the addressed atomic
  // sequence, not a synchronization operation on its own.
  for (ConSanSyncSequence &sequence : result.sync_sequences) {
    if (sequence.kind != ConSanSyncSequenceKind::Atomic ||
        (sequence.memory_role != ConSanSyncMemoryRole::Unknown &&
         sequence.memory_role != ConSanSyncMemoryRole::Acquire)) {
      continue;
    }
    const auto boundary = atomic_release_wait_boundary(sequence, blocks, arch);
    if (!boundary)
      continue;
    const bool already_acquire = sequence.memory_role == ConSanSyncMemoryRole::Acquire;
    sequence.memory_role =
        already_acquire ? ConSanSyncMemoryRole::AcquireRelease : ConSanSyncMemoryRole::Release;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence_reason = already_acquire
                                     ? "exact same-block gfx12 release wait immediately before "
                                       "bounded atomic/acquire-cache sequence"
                                     : "exact same-block gfx12 s_wait_storecnt_dscnt 0 "
                                       "immediately before atomic release";
    sequence.release_wait_text_offset = *boundary;
    sequence.identity += "|release-wait=pc=0x" + fixed_hex(*boundary, 16);
    sequence.begin_text_offset = *boundary;
  }
}

void build_moi_fence_candidate_inventory(ConSanResult &result) {
  result.moi_fence_candidates.clear();

  struct SequenceMembership {
    const ConSanSyncSequence *sequence = nullptr;
    bool unique = true;
  };
  std::unordered_map<std::string_view, SequenceMembership> sequence_by_event_identity;
  sequence_by_event_identity.reserve(result.sync_events.size());
  for (const ConSanSyncSequence &sequence : result.sync_sequences) {
    for (const std::string &identity : sequence.member_event_identities) {
      const auto [membership, inserted] =
          sequence_by_event_identity.try_emplace(identity, SequenceMembership{&sequence, true});
      if (!inserted) {
        membership->second.sequence = nullptr;
        membership->second.unique = false;
      }
    }
  }

  struct UniqueSyncEvent {
    const ConSanSyncEvent *event = nullptr;
    bool unique = true;
  };
  std::unordered_map<std::string_view, UniqueSyncEvent> event_by_identity;
  event_by_identity.reserve(result.sync_events.size());
  for (const ConSanSyncEvent &event : result.sync_events) {
    const auto [indexed, inserted] =
        event_by_identity.try_emplace(event.identity, UniqueSyncEvent{&event, true});
    if (!inserted) {
      indexed->second.event = nullptr;
      indexed->second.unique = false;
    }
  }

  for (const ConSanSyncEvent &fence : result.sync_events) {
    if (fence.kind != ConSanSyncEventKind::Fence)
      continue;

    ConSanMoiFenceCandidate candidate;
    candidate.identity = fence.identity + "|moi-fence-candidate";
    candidate.fence_event_identity = fence.identity;
    candidate.container_name = fence.container_name;
    candidate.in_kernel = fence.in_kernel;
    candidate.text_offset = fence.text_offset;
    candidate.file_offset = fence.file_offset;
    candidate.size = fence.size;

    const auto membership = sequence_by_event_identity.find(fence.identity);
    if (membership == sequence_by_event_identity.end() || !membership->second.unique ||
        membership->second.sequence == nullptr) {
      candidate.rejection_reason = "missing-or-ambiguous-sequence-membership";
      result.moi_fence_candidates.push_back(std::move(candidate));
      continue;
    }
    const ConSanSyncSequence *sequence = membership->second.sequence;
    candidate.sequence_identity = sequence->identity;

    const bool addressed_atomic =
        sequence->kind == ConSanSyncSequenceKind::Atomic &&
        (sequence->operation == ConSanSyncOperation::AtomicRmw ||
         sequence->operation == ConSanSyncOperation::AtomicCompareExchange);
    const bool addressed_ordinary = sequence->kind == ConSanSyncSequenceKind::OrdinaryMemory &&
                                    (sequence->operation == ConSanSyncOperation::OrdinaryLoad ||
                                     sequence->operation == ConSanSyncOperation::OrdinaryStore);
    if (!addressed_atomic && !addressed_ordinary) {
      candidate.rejection_reason = "fence-not-associated-with-addressed-communication-sequence";
      result.moi_fence_candidates.push_back(std::move(candidate));
      continue;
    }
    if (!consan_sync_confidence_meets(sequence->confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        !consan_sync_confidence_meets(sequence->memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative)) {
      candidate.rejection_reason = "communication-sequence-confidence-insufficient";
      result.moi_fence_candidates.push_back(std::move(candidate));
      continue;
    }

    if (is_release_cache_event(&fence))
      candidate.memory_role = ConSanSyncMemoryRole::Release;
    else if (is_acquire_cache_event(&fence))
      candidate.memory_role = ConSanSyncMemoryRole::Acquire;
    else {
      candidate.rejection_reason = "unsupported-cache-operation";
      result.moi_fence_candidates.push_back(std::move(candidate));
      continue;
    }
    const bool role_is_carried =
        sequence->memory_role == candidate.memory_role ||
        sequence->memory_role == ConSanSyncMemoryRole::AcquireRelease ||
        sequence->memory_role == ConSanSyncMemoryRole::SequentiallyConsistent;
    if (!role_is_carried) {
      candidate.rejection_reason = "cache-role-not-carried-by-communication-sequence";
      result.moi_fence_candidates.push_back(std::move(candidate));
      continue;
    }

    const ConSanSyncEvent *communication = nullptr;
    bool malformed_member = false;
    for (const std::string &identity : sequence->member_event_identities) {
      const auto indexed = event_by_identity.find(identity);
      if (indexed == event_by_identity.end() || !indexed->second.unique ||
          indexed->second.event == nullptr) {
        malformed_member = true;
        break;
      }
      const ConSanSyncEvent *member = indexed->second.event;
      if (member->kind != ConSanSyncEventKind::Atomic &&
          member->kind != ConSanSyncEventKind::OrdinaryMemory)
        continue;
      if (communication != nullptr) {
        malformed_member = true;
        break;
      }
      communication = member;
    }
    if (malformed_member || communication == nullptr || communication->identity.empty()) {
      candidate.rejection_reason = "missing-or-ambiguous-communication-event";
      result.moi_fence_candidates.push_back(std::move(candidate));
      continue;
    }
    if (communication->address_source != ConSanSyncAddressSource::GlobalScalarVector &&
        communication->address_source != ConSanSyncAddressSource::BufferResource &&
        communication->address_source != ConSanSyncAddressSource::FlatVector) {
      candidate.rejection_reason = "communication-address-source-unsupported";
      result.moi_fence_candidates.push_back(std::move(candidate));
      continue;
    }
    if (!communication->raw_scope || *communication->raw_scope == 0u ||
        *communication->raw_scope > 3u) {
      candidate.rejection_reason = "communication-scope-unsupported";
      result.moi_fence_candidates.push_back(std::move(candidate));
      continue;
    }

    candidate.communication_event_identity = communication->identity;
    candidate.communication_address_source = communication->address_source;
    candidate.communication_static_byte_offset = communication->static_byte_offset;
    candidate.raw_scope = communication->raw_scope;
    candidate.eligible = true;
    result.moi_fence_candidates.push_back(std::move(candidate));
  }
}

struct OrdinaryReleaseBoundaries {
  std::vector<uint64_t> offsets;
  bool saw_wait = false;
};

[[nodiscard]] bool is_permitted_ordinary_release_bookkeeping(const Instruction &instruction) {
  const std::string_view mnemonic = instruction.mnemonic();
  return mnemonic.starts_with("s_wait") || mnemonic == "s_delay_alu" || mnemonic == "s_nop";
}

[[nodiscard]] OrdinaryReleaseBoundaries
ordinary_release_cache_boundaries(const ConSanSyncSequence &store,
                                  const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  OrdinaryReleaseBoundaries boundaries;
  if (!store.basic_block_index || *store.basic_block_index >= blocks.size())
    return boundaries;
  BasicBlock *block = blocks[*store.basic_block_index].get();
  if (block == nullptr)
    return boundaries;
  std::optional<uint64_t> prior_end;
  for (const Instruction &instruction : block->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset >= store.begin_text_offset)
      break;
    const int size = instruction.size();
    if (size <= 0 || (prior_end && offset != *prior_end)) {
      boundaries = {};
      prior_end.reset();
      continue;
    }
    prior_end = offset + static_cast<uint64_t>(size);
    if (instruction.mnemonic() == std::string_view("global_wb")) {
      boundaries.offsets.push_back(offset);
    } else if (!is_permitted_ordinary_release_bookkeeping(instruction)) {
      boundaries = {};
    } else if (!boundaries.offsets.empty() && instruction.mnemonic().starts_with("s_wait")) {
      boundaries.saw_wait = true;
    }
  }
  if (!prior_end || *prior_end != store.begin_text_offset)
    boundaries = {};
  return boundaries;
}

[[nodiscard]] std::optional<uint64_t>
ordinary_release_cache_after_store(const ConSanSyncSequence &store,
                                   const std::vector<std::unique_ptr<BasicBlock>> &blocks) {
  if (!store.basic_block_index || *store.basic_block_index >= blocks.size())
    return std::nullopt;
  BasicBlock *block = blocks[*store.basic_block_index].get();
  if (block == nullptr)
    return std::nullopt;
  uint64_t expected = store.end_text_offset;
  bool saw_wait = false;
  for (const Instruction &instruction : block->instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset < expected)
      continue;
    const int size = instruction.size();
    if (offset != expected || size <= 0)
      return std::nullopt;
    expected += static_cast<uint64_t>(size);
    if (instruction.mnemonic() == std::string_view("global_wb"))
      return saw_wait ? std::optional<uint64_t>(offset) : std::nullopt;
    if (!is_permitted_ordinary_release_bookkeeping(instruction))
      return std::nullopt;
    saw_wait |= instruction.mnemonic().starts_with("s_wait");
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<uint64_t>
ordinary_release_wait_boundary(const ConSanSyncSequence &store,
                               const std::vector<std::unique_ptr<BasicBlock>> &blocks,
                               rj_code_arch_t arch) {
  if (store.kind != ConSanSyncSequenceKind::OrdinaryMemory ||
      store.operation != ConSanSyncOperation::OrdinaryStore || !store.basic_block_index ||
      *store.basic_block_index >= blocks.size()) {
    return std::nullopt;
  }
  const auto expected_store = build_s_wait_storecnt0(arch);
  const auto expected_load_ds = build_s_wait_loadcnt_dscnt0(arch);
  BasicBlock *block = blocks[*store.basic_block_index].get();
  if (!expected_store || !expected_load_ds || block == nullptr)
    return std::nullopt;

  const Instruction *load_ds_wait = nullptr;
  for (const Instruction &instruction : block->instructions()) {
    const int size = instruction.size();
    if (size != static_cast<int>(sizeof(uint32_t)) || instruction.raw_encoding() == nullptr)
      continue;
    const uint64_t end = instruction.src_loc() + sizeof(uint32_t);
    if (end == store.begin_text_offset &&
        instruction.mnemonic() == std::string_view("s_wait_loadcnt_dscnt") &&
        instruction.raw_encoding()[0] == *expected_load_ds) {
      load_ds_wait = &instruction;
    }
  }
  if (load_ds_wait == nullptr)
    return std::nullopt;
  for (const Instruction &instruction : block->instructions()) {
    const int size = instruction.size();
    if (size == static_cast<int>(sizeof(uint32_t)) && instruction.raw_encoding() != nullptr &&
        instruction.src_loc() + sizeof(uint32_t) == load_ds_wait->src_loc() &&
        instruction.mnemonic() == std::string_view("s_wait_storecnt") &&
        instruction.raw_encoding()[0] == *expected_store) {
      return instruction.src_loc();
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::vector<size_t> fence_sequence_indices_by_begin(const ConSanResult &result) {
  std::vector<size_t> indices;
  indices.reserve(result.sync_sequences.size());
  for (size_t index = 0; index < result.sync_sequences.size(); ++index) {
    if (result.sync_sequences[index].kind == ConSanSyncSequenceKind::Fence)
      indices.push_back(index);
  }
  std::ranges::sort(indices, [&](size_t lhs, size_t rhs) {
    const uint64_t lhs_offset = result.sync_sequences[lhs].begin_text_offset;
    const uint64_t rhs_offset = result.sync_sequences[rhs].begin_text_offset;
    return lhs_offset != rhs_offset ? lhs_offset < rhs_offset : lhs < rhs;
  });
  return indices;
}

[[nodiscard]] auto fence_sequence_lower_bound(const ConSanResult &result,
                                              const std::vector<size_t> &indices,
                                              uint64_t text_offset) {
  return std::lower_bound(indices.begin(), indices.end(), text_offset,
                          [&](size_t index, uint64_t offset) {
                            return result.sync_sequences[index].begin_text_offset < offset;
                          });
}

void associate_ordinary_release_sync_sequences(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks, rj_code_arch_t arch,
    const SyncEventIdentityIndex &events, ConSanResult &result) {
  const std::vector<size_t> fence_indices = fence_sequence_indices_by_begin(result);
  std::unordered_map<size_t, size_t> associations;
  std::unordered_set<size_t> claimed_cache_sequences;
  for (size_t store_index = 0; store_index < result.sync_sequences.size(); ++store_index) {
    const ConSanSyncSequence &store_sequence = result.sync_sequences[store_index];
    if (store_sequence.kind != ConSanSyncSequenceKind::OrdinaryMemory ||
        store_sequence.operation != ConSanSyncOperation::OrdinaryStore) {
      continue;
    }
    const ConSanSyncEvent *store = first_sequence_event(events, store_sequence);
    if (store == nullptr)
      continue;
    const OrdinaryReleaseBoundaries boundaries =
        ordinary_release_cache_boundaries(store_sequence, blocks);
    std::optional<uint64_t> cache_offset;
    if (boundaries.saw_wait && boundaries.offsets.size() == 1u)
      cache_offset = boundaries.offsets.front();
    if (!cache_offset)
      cache_offset = ordinary_release_cache_after_store(store_sequence, blocks);
    if (!cache_offset)
      continue;
    std::vector<size_t> cache_matches;
    for (auto it = fence_sequence_lower_bound(result, fence_indices, *cache_offset);
         it != fence_indices.end(); ++it) {
      const size_t cache_index = *it;
      const ConSanSyncSequence &cache_sequence = result.sync_sequences[cache_index];
      if (cache_sequence.begin_text_offset != *cache_offset)
        break;
      const ConSanSyncEvent *cache = first_sequence_event(events, cache_sequence);
      if (cache != nullptr && consan_ordinary_release_metadata_compatible(*cache, cache_sequence,
                                                                          *store, store_sequence)) {
        cache_matches.push_back(cache_index);
      }
    }
    if (cache_matches.size() != 1u || claimed_cache_sequences.contains(cache_matches.front()))
      continue;
    associations.emplace(store_index, cache_matches.front());
    claimed_cache_sequences.insert(cache_matches.front());
  }

  std::vector<ConSanSyncSequence> associated;
  associated.reserve(result.sync_sequences.size());
  for (size_t index = 0; index < result.sync_sequences.size(); ++index) {
    if (claimed_cache_sequences.contains(index))
      continue;
    const auto association = associations.find(index);
    if (association == associations.end()) {
      associated.push_back(std::move(result.sync_sequences[index]));
      continue;
    }
    ConSanSyncSequence sequence = std::move(result.sync_sequences[index]);
    ConSanSyncSequence &cache = result.sync_sequences[association->second];
    sequence.memory_role = ConSanSyncMemoryRole::Release;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = combine_consan_sync_confidence(sequence.confidence, cache.confidence);
    sequence.confidence_reason =
        "exact same-block global_wb/waits/ordinary store release pattern with only "
        "permitted bookkeeping between members";
    sequence.identity += "|release-cache=" + cache.identity;
    sequence.begin_text_offset = cache.begin_text_offset;
    sequence.in_cyclic_cfg_component |= cache.in_cyclic_cfg_component;
    sequence.inside_scalar_clause |= cache.inside_scalar_clause;
    std::vector<std::string> members = cache.member_event_identities;
    members.insert(members.end(), sequence.member_event_identities.begin(),
                   sequence.member_event_identities.end());
    sequence.member_event_identities = std::move(members);
    associated.push_back(std::move(sequence));
  }
  result.sync_sequences = std::move(associated);

  // gfx12 release stores in optimized STREAM-K helpers have no separate
  // `global_wb` event. The compiler instead emits this exact same-block pair
  // immediately before a device/system-scoped b32 store. Keep this admission
  // separate from the broader cache-associated form above: either missing
  // wait, a nonzero counter, intervening code, or weaker scope stays unknown.
  for (ConSanSyncSequence &sequence : result.sync_sequences) {
    if (sequence.kind != ConSanSyncSequenceKind::OrdinaryMemory ||
        sequence.operation != ConSanSyncOperation::OrdinaryStore ||
        sequence.memory_role != ConSanSyncMemoryRole::Unknown) {
      continue;
    }
    const ConSanSyncEvent *store = first_sequence_event(events, sequence);
    if (store == nullptr || store->width_bits != 32u || !store->raw_scope ||
        (*store->raw_scope != 2u && *store->raw_scope != 3u) ||
        store->confidence != ConSanSemanticConfidence::Conservative ||
        store->execution_owners.empty() ||
        !std::ranges::equal(
            store->execution_owners, sequence.execution_owners,
            [](const ConSanExecutionOwner &left, const ConSanExecutionOwner &right) {
              return left.descriptor_file_offset == right.descriptor_file_offset &&
                     left.proof == right.proof;
            })) {
      continue;
    }
    const auto boundary = ordinary_release_wait_boundary(sequence, blocks, arch);
    if (!boundary)
      continue;
    sequence.memory_role = ConSanSyncMemoryRole::Release;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence_reason =
        "exact same-block gfx12 s_wait_storecnt 0 / s_wait_loadcnt_dscnt 0 immediately "
        "before scoped ordinary release store";
    sequence.release_wait_text_offset = *boundary;
    sequence.identity += "|release-waits=pc=0x" + fixed_hex(*boundary, 16);
    sequence.begin_text_offset = *boundary;
  }
}

void associate_ordinary_acquire_sync_sequences(
    const std::vector<std::unique_ptr<BasicBlock>> &blocks,
    const SyncEventIdentityIndex &events, ConSanResult &result) {
  const std::vector<size_t> fence_indices = fence_sequence_indices_by_begin(result);
  std::unordered_map<size_t, size_t> associations;
  std::unordered_set<size_t> claimed_cache_sequences;
  for (size_t load_index = 0; load_index < result.sync_sequences.size(); ++load_index) {
    const ConSanSyncSequence &load_sequence = result.sync_sequences[load_index];
    if (load_sequence.kind != ConSanSyncSequenceKind::OrdinaryMemory ||
        load_sequence.operation != ConSanSyncOperation::OrdinaryLoad) {
      continue;
    }
    const ConSanSyncEvent *load = first_sequence_event(events, load_sequence);
    if (load == nullptr)
      continue;
    std::vector<size_t> cache_matches;
    std::vector<size_t> bounded_metadata_matches;
    const uint64_t maximum_cache_offset = load_sequence.end_text_offset > UINT64_MAX - 64u
                                              ? UINT64_MAX
                                              : load_sequence.end_text_offset + 64u;
    for (auto it = fence_sequence_lower_bound(result, fence_indices, load_sequence.end_text_offset);
         it != fence_indices.end(); ++it) {
      const size_t cache_index = *it;
      const ConSanSyncSequence &cache_sequence = result.sync_sequences[cache_index];
      if (cache_sequence.begin_text_offset > maximum_cache_offset)
        break;
      const ConSanSyncEvent *cache = first_sequence_event(events, cache_sequence);
      if (cache == nullptr ||
          !ordinary_acquire_metadata_compatible_impl(*load, load_sequence, *cache, cache_sequence,
                                                     /*require_same_block=*/false) ||
          load_sequence.end_text_offset > cache_sequence.begin_text_offset ||
          cache_sequence.begin_text_offset - load_sequence.end_text_offset > 64u) {
        continue;
      }
      bounded_metadata_matches.push_back(cache_index);
      const auto path = prove_bounded_acquire_path(load_sequence, cache_sequence, blocks,
                                                   /*require_wait_for_same_block=*/true);
      if (!path)
        continue;
      const bool same_block = *path == BoundedAcquirePathKind::SameBlock;
      if (ordinary_acquire_metadata_compatible_impl(*load, load_sequence, *cache, cache_sequence,
                                                    /*require_same_block=*/same_block))
        cache_matches.push_back(cache_index);
    }
    if (bounded_metadata_matches.size() != 1u || cache_matches.size() != 1u ||
        bounded_metadata_matches.front() != cache_matches.front() ||
        claimed_cache_sequences.contains(cache_matches.front()))
      continue;
    associations.emplace(load_index, cache_matches.front());
    claimed_cache_sequences.insert(cache_matches.front());
  }

  std::vector<ConSanSyncSequence> associated;
  associated.reserve(result.sync_sequences.size());
  for (size_t index = 0; index < result.sync_sequences.size(); ++index) {
    if (claimed_cache_sequences.contains(index))
      continue;
    const auto association = associations.find(index);
    if (association == associations.end()) {
      associated.push_back(std::move(result.sync_sequences[index]));
      continue;
    }
    ConSanSyncSequence sequence = std::move(result.sync_sequences[index]);
    ConSanSyncSequence &cache = result.sync_sequences[association->second];
    sequence.memory_role = ConSanSyncMemoryRole::Acquire;
    sequence.memory_role_confidence = ConSanSemanticConfidence::Conservative;
    sequence.confidence = combine_consan_sync_confidence(sequence.confidence, cache.confidence);
    sequence.confidence_reason = "bounded ordinary-load/acquire-cache compiler-bookkeeping proof";
    sequence.identity += "|acquire-cache=" + cache.identity;
    sequence.end_text_offset = cache.end_text_offset;
    sequence.in_cyclic_cfg_component |= cache.in_cyclic_cfg_component;
    sequence.inside_scalar_clause |= cache.inside_scalar_clause;
    sequence.member_event_identities.insert(sequence.member_event_identities.end(),
                                            cache.member_event_identities.begin(),
                                            cache.member_event_identities.end());
    associated.push_back(std::move(sequence));
  }
  result.sync_sequences = std::move(associated);
}

struct CommunicationAddressOperands {
  ConSanCommunicationAddressKind kind = ConSanCommunicationAddressKind::Atomic;
  ConSanSyncAddressSource source = ConSanSyncAddressSource::Unknown;
  uint16_t address_vgpr = 0;
  uint16_t address_vgpr_count = 0;
  std::optional<uint16_t> address_sgpr;
  uint16_t address_sgpr_count = 0;
  int64_t static_byte_offset = 0;
};

[[nodiscard]] std::optional<CommunicationAddressOperands>
communication_address_operands(const ConSanResult &result, const ConSanSyncEvent &event) {
  size_t match_count = 0;
  std::optional<CommunicationAddressOperands> operands;
  const auto visit = [&](const auto &container) {
    if (container.name != event.container_name)
      return;
    if (event.kind == ConSanSyncEventKind::Atomic) {
      for (const ConSanAtomicSite &site : container.atomic_sites) {
        if (site.text_offset != event.text_offset)
          continue;
        ++match_count;
        if (!event.static_byte_offset || !site.raw_ioffset ||
            *event.static_byte_offset != *site.raw_ioffset)
          continue;
        if (event.address_source == ConSanSyncAddressSource::GlobalScalarVector && site.addr_vgpr &&
            site.saddr_sgpr) {
          operands = {.kind = ConSanCommunicationAddressKind::Atomic,
                      .source = event.address_source,
                      .address_vgpr = *site.addr_vgpr,
                      .address_vgpr_count = 1,
                      .address_sgpr = site.saddr_sgpr,
                      .address_sgpr_count = 2,
                      .static_byte_offset = *site.raw_ioffset};
        } else if (event.address_source == ConSanSyncAddressSource::FlatVector && site.addr_vgpr) {
          operands = {.kind = ConSanCommunicationAddressKind::Atomic,
                      .source = event.address_source,
                      .address_vgpr = *site.addr_vgpr,
                      .address_vgpr_count = 2,
                      .address_sgpr = std::nullopt,
                      .address_sgpr_count = 0,
                      .static_byte_offset = *site.raw_ioffset};
        }
      }
      return;
    }
    if (event.kind != ConSanSyncEventKind::OrdinaryMemory)
      return;
    for (const ConSanOrdinaryMemorySite &site : container.ordinary_memory_sites) {
      if (site.text_offset != event.text_offset)
        continue;
      ++match_count;
      if (site.support_reason != ConSanOrdinaryMemorySupportReason::Supported ||
          !event.static_byte_offset || !site.raw_ioffset ||
          *event.static_byte_offset != *site.raw_ioffset)
        continue;
      if (event.address_source == ConSanSyncAddressSource::GlobalScalarVector &&
          site.address_vgpr && site.address_sgpr) {
        operands = {.kind = ConSanCommunicationAddressKind::Ordinary,
                    .source = event.address_source,
                    .address_vgpr = *site.address_vgpr,
                    .address_vgpr_count = 1,
                    .address_sgpr = site.address_sgpr,
                    .address_sgpr_count = 2,
                    .static_byte_offset = *site.raw_ioffset};
      } else if (event.address_source == ConSanSyncAddressSource::FlatVector && site.address_vgpr) {
        operands = {.kind = ConSanCommunicationAddressKind::Ordinary,
                    .source = event.address_source,
                    .address_vgpr = *site.address_vgpr,
                    .address_vgpr_count = 2,
                    .address_sgpr = std::nullopt,
                    .address_sgpr_count = 0,
                    .static_byte_offset = *site.raw_ioffset};
      }
    }
  };
  if (event.in_kernel) {
    for (const ConSanKernelInfo &kernel : result.kernels)
      visit(kernel);
  } else {
    for (const ConSanFunctionInfo &function : result.functions)
      visit(function);
  }
  if (match_count != 1)
    return std::nullopt;
  return operands;
}

[[nodiscard]] bool communication_role_is_supported(ConSanSyncMemoryRole role) {
  return role == ConSanSyncMemoryRole::Acquire || role == ConSanSyncMemoryRole::Release ||
         role == ConSanSyncMemoryRole::AcquireRelease ||
         role == ConSanSyncMemoryRole::SequentiallyConsistent;
}

void build_communication_address_recipes(
    const AmdGpuCodeObject &code_object,
    const std::vector<std::unique_ptr<BasicBlock>> &blocks,
    const SyncEventIdentityIndex &events, rj_code_arch_t arch, ConSanResult &result) {
  result.communication_address_recipes.clear();
  const std::vector<BasicBlock *> block_ptrs = block_ptrs_for(blocks);
  if (block_ptrs.empty())
    return;
  const LivenessAnalysis liveness{KernelBlockScope(block_ptrs)};
  const auto original_bytes = std::span<const uint8_t>(
      reinterpret_cast<const uint8_t *>(code_object.image_data()), code_object.image_size());

  for (const ConSanSyncSequence &sequence : result.sync_sequences) {
    if (sequence.kind != ConSanSyncSequenceKind::Atomic &&
        sequence.kind != ConSanSyncSequenceKind::OrdinaryMemory)
      continue;
    ConSanCommunicationAddressRecipe recipe;
    recipe.sequence_identity = sequence.identity;
    recipe.post_sequence_text_offset = sequence.end_text_offset;
    if (!communication_role_is_supported(sequence.memory_role) || sequence.identity.empty() ||
        sequence.in_cyclic_cfg_component || sequence.inside_scalar_clause) {
      recipe.support = ConSanCommunicationAddressSupport::UnsupportedSequence;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }

    const ConSanSyncEvent *communication = nullptr;
    for (const std::string &identity : sequence.member_event_identities) {
      const ConSanSyncEvent *event = find_sync_event(events, identity);
      if (event == nullptr || (event->kind != ConSanSyncEventKind::Atomic &&
                               event->kind != ConSanSyncEventKind::OrdinaryMemory))
        continue;
      if (communication != nullptr) {
        communication = nullptr;
        break;
      }
      communication = event;
    }
    if (communication == nullptr) {
      recipe.support = ConSanCommunicationAddressSupport::AmbiguousCommunicationEvent;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    recipe.communication_event_identity = communication->identity;
    recipe.address_source = communication->address_source;
    if (communication->address_source != ConSanSyncAddressSource::GlobalScalarVector &&
        communication->address_source != ConSanSyncAddressSource::FlatVector) {
      recipe.support = ConSanCommunicationAddressSupport::UnsupportedAddressSource;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    const auto operands = communication_address_operands(result, *communication);
    if (!operands) {
      recipe.support = communication->static_byte_offset
                           ? ConSanCommunicationAddressSupport::MissingAddressOperands
                           : ConSanCommunicationAddressSupport::InvalidStaticOffset;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    recipe.kind = operands->kind;
    recipe.address_vgpr = operands->address_vgpr;
    recipe.address_vgpr_count = operands->address_vgpr_count;
    recipe.address_sgpr = operands->address_sgpr;
    recipe.address_sgpr_count = operands->address_sgpr_count;
    recipe.static_byte_offset = operands->static_byte_offset;

    const Instruction *post_sequence =
        find_instruction_at_text_offset(KernelBlockScope(block_ptrs), sequence.end_text_offset);
    if (post_sequence == nullptr) {
      recipe.support = ConSanCommunicationAddressSupport::MissingPostSequenceInstruction;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    if (!liveness.is_live_before(*post_sequence,
                                 RegisterRef{RegClass::VGPR, recipe.address_vgpr,
                                             static_cast<uint8_t>(recipe.address_vgpr_count)}) ||
        (recipe.address_sgpr &&
         !liveness.is_live_before(*post_sequence,
                                  RegisterRef{RegClass::SGPR, *recipe.address_sgpr,
                                              static_cast<uint8_t>(recipe.address_sgpr_count)}))) {
      recipe.support = ConSanCommunicationAddressSupport::AddressNotLiveAfterSequence;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    if (sequence.execution_owners.empty()) {
      recipe.support = ConSanCommunicationAddressSupport::MissingOwners;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    for (const ConSanExecutionOwner &owner : sequence.execution_owners)
      recipe.owner_descriptor_file_offsets.push_back(owner.descriptor_file_offset);
    std::ranges::sort(recipe.owner_descriptor_file_offsets);
    recipe.owner_descriptor_file_offsets.erase(
        std::ranges::unique(recipe.owner_descriptor_file_offsets).begin(),
        recipe.owner_descriptor_file_offsets.end());
    if (recipe.owner_descriptor_file_offsets.size() != sequence.execution_owners.size()) {
      recipe.support = ConSanCommunicationAddressSupport::MissingOwners;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    std::optional<uint16_t> owner_vgpr_limit;
    for (uint64_t descriptor : recipe.owner_descriptor_file_offsets) {
      const auto count = descriptor_vgpr_allocation_count(original_bytes, descriptor, arch);
      if (!count) {
        recipe.support = ConSanCommunicationAddressSupport::InvalidOwnerDescriptor;
        break;
      }
      owner_vgpr_limit = std::min(owner_vgpr_limit.value_or(*count), *count);
    }
    if (recipe.support == ConSanCommunicationAddressSupport::InvalidOwnerDescriptor) {
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    recipe.scratch_vgpr_count = 2;
    recipe.scratch_vgpr = liveness.find_free_run(post_sequence, recipe.scratch_vgpr_count);
    if (!recipe.scratch_vgpr) {
      recipe.support = ConSanCommunicationAddressSupport::NoPostSequenceScratchPair;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    if (static_cast<uint32_t>(*recipe.scratch_vgpr) + recipe.scratch_vgpr_count >
        *owner_vgpr_limit) {
      recipe.scratch_vgpr.reset();
      recipe.support = ConSanCommunicationAddressSupport::ScratchOutsideOwnerAllocation;
      result.communication_address_recipes.push_back(std::move(recipe));
      continue;
    }
    recipe.support = ConSanCommunicationAddressSupport::Supported;
    result.communication_address_recipes.push_back(std::move(recipe));
  }
}

[[nodiscard]] bool sync_role_has_edge(ConSanSyncMemoryRole role, ConSanPerturbationEdge edge) {
  if (edge == ConSanPerturbationEdge::Release) {
    return role == ConSanSyncMemoryRole::Release || role == ConSanSyncMemoryRole::AcquireRelease ||
           role == ConSanSyncMemoryRole::SequentiallyConsistent;
  }
  return role == ConSanSyncMemoryRole::Acquire || role == ConSanSyncMemoryRole::AcquireRelease ||
         role == ConSanSyncMemoryRole::SequentiallyConsistent;
}

[[nodiscard]] const ConSanSyncEvent *find_sync_event(const ConSanResult &result,
                                                     std::string_view identity) {
  const auto event = std::ranges::find(result.sync_events, identity, &ConSanSyncEvent::identity);
  return event == result.sync_events.end() ? nullptr : &*event;
}

[[nodiscard]] bool sequence_has_exact_members(const ConSanResult &result,
                                              const ConSanSyncSequence &sequence) {
  uint64_t prior_end = sequence.begin_text_offset;
  for (const std::string &identity : sequence.member_event_identities) {
    const ConSanSyncEvent *event = find_sync_event(result, identity);
    if (event == nullptr || event->container_name != sequence.container_name ||
        event->in_kernel != sequence.in_kernel || event->text_offset < prior_end ||
        event->text_offset < sequence.begin_text_offset ||
        event->text_offset + event->size > sequence.end_text_offset) {
      return false;
    }
    prior_end = event->text_offset + event->size;
  }
  return !sequence.member_event_identities.empty();
}

[[nodiscard]] bool
sequence_has_exact_members(const SyncEventIdentityIndex &events,
                           const ConSanSyncSequence &sequence) {
  uint64_t prior_end = sequence.begin_text_offset;
  for (const std::string &identity : sequence.member_event_identities) {
    const ConSanSyncEvent *event = find_sync_event(events, identity);
    if (event == nullptr || event->container_name != sequence.container_name ||
        event->in_kernel != sequence.in_kernel || event->text_offset < prior_end ||
        event->text_offset < sequence.begin_text_offset ||
        event->text_offset + event->size > sequence.end_text_offset) {
      return false;
    }
    prior_end = event->text_offset + event->size;
  }
  return !sequence.member_event_identities.empty();
}

[[nodiscard]] bool
sequence_atomic_has_device_or_system_scope(const SyncEventIdentityIndex &events,
                                           const ConSanSyncSequence &sequence) {
  const ConSanSyncEvent *atomic = nullptr;
  for (const std::string &identity : sequence.member_event_identities) {
    const ConSanSyncEvent *event = find_sync_event(events, identity);
    if (event == nullptr || event->kind != ConSanSyncEventKind::Atomic)
      continue;
    if (atomic != nullptr)
      return false;
    atomic = event;
  }
  // RDNA4 encodes device and system scope as 2 and 3 respectively. Read this
  // from the exact atomic member rather than trusting aggregate sequence
  // metadata inherited while cache/fence members are associated.
  return atomic != nullptr && atomic->raw_scope &&
         (*atomic->raw_scope == 2u || *atomic->raw_scope == 3u);
}

[[nodiscard]] std::string
perturbation_rejection_reason(const SyncEventIdentityIndex &events,
                              const ConSanSyncSequence &sequence, ConSanPerturbationKind kind,
                              ConSanPerturbationEdge edge) {
  if (!sequence.basic_block_index)
    return "missing-exact-basic-block";
  if (sequence.in_cyclic_cfg_component)
    return "cyclic-cfg-component";
  if (sequence.inside_scalar_clause)
    return "inside-s-clause";
  if (is_rocclr_runtime_kernel_name(sequence.container_name))
    return "runtime-helper";
  if (!sequence_has_exact_members(events, sequence))
    return "non-exact-sequence-members";
  if (!consan_sync_confidence_meets(sequence.confidence, ConSanSemanticConfidence::Conservative))
    return "ambiguous-or-unsupported-sequence";

  if (kind == ConSanPerturbationKind::Barrier) {
    if (sequence.kind != ConSanSyncSequenceKind::Barrier ||
        sequence.operation != ConSanSyncOperation::BarrierFull ||
        sequence.member_event_identities.size() != 2u)
      return "not-qualified-full-barrier";
    if (sequence.barrier_operand_source != ConSanBarrierSite::OperandSource::Immediate ||
        !sequence.barrier_id || sequence.barrier_scope == ConSanBarrierSite::Scope::Unknown)
      return "dynamic-or-unknown-barrier-participants";
    return {};
  }

  if (kind == ConSanPerturbationKind::Atomic) {
    if (sequence.kind != ConSanSyncSequenceKind::Atomic)
      return "not-atomic-sequence";
    if (!consan_sync_confidence_meets(sequence.memory_role_confidence,
                                      ConSanSemanticConfidence::Conservative) ||
        !sync_role_has_edge(sequence.memory_role, edge))
      return "unknown-or-inapplicable-memory-role";
    if (!sequence_atomic_has_device_or_system_scope(events, sequence))
      return "unsupported-atomic-scope";
    if (sequence.address_source == ConSanSyncAddressSource::Unknown)
      return "ambiguous-address-provenance";
    return {};
  }
  return "no-perturbation-kind";
}

void build_perturbation_candidate_inventory(ConSanResult &result) {
  result.perturbation_candidates.clear();
  const SyncEventIdentityIndex events = build_sync_event_identity_index(result);
  for (const ConSanSyncSequence &sequence : result.sync_sequences) {
    const ConSanPerturbationKind kind =
        sequence.kind == ConSanSyncSequenceKind::Barrier  ? ConSanPerturbationKind::Barrier
        : sequence.kind == ConSanSyncSequenceKind::Atomic ? ConSanPerturbationKind::Atomic
                                                          : ConSanPerturbationKind::None;
    if (kind == ConSanPerturbationKind::None)
      continue;
    if (sequence.member_event_identities.empty())
      continue;
    for (const ConSanPerturbationEdge edge :
         {ConSanPerturbationEdge::Release, ConSanPerturbationEdge::Acquire}) {
      const std::string rejection = perturbation_rejection_reason(events, sequence, kind, edge);
      const std::string &anchor_identity = edge == ConSanPerturbationEdge::Release
                                               ? sequence.member_event_identities.front()
                                               : sequence.member_event_identities.back();
      const ConSanSyncEvent *anchor = find_sync_event(events, anchor_identity);
      ConSanPerturbationCandidate candidate;
      candidate.kind = kind;
      candidate.edge = edge;
      candidate.identity =
          sequence.identity + (edge == ConSanPerturbationEdge::Release ? "|perturb-edge=release"
                                                                       : "|perturb-edge=acquire");
      candidate.sequence_identity = sequence.identity;
      candidate.container_name = sequence.container_name;
      candidate.in_kernel = sequence.in_kernel;
      candidate.basic_block_index = sequence.basic_block_index.value_or(0);
      candidate.anchor_event_identity = anchor_identity;
      candidate.anchor_text_offset = anchor == nullptr ? 0 : anchor->text_offset;
      candidate.anchor_size = anchor == nullptr ? 0 : anchor->size;
      candidate.ordered_member_identities = sequence.member_event_identities;
      candidate.eligible = rejection.empty() && anchor != nullptr;
      candidate.rejection_reason = anchor == nullptr ? "missing-anchor-event" : rejection;
      result.perturbation_candidates.push_back(std::move(candidate));
    }
  }
}

struct CarriedPerturbationPlan {
  std::string source_candidate_identity;
  std::string source_sequence_identity;
  std::string container_name;
  bool in_kernel = true;
  uint32_t basic_block_index = 0;
  std::string anchor_event_identity;
  std::vector<std::string> ordered_member_identities;
  uint64_t source_anchor_text_offset = 0;
  uint64_t translated_anchor_text_offset = 0;
  uint32_t anchor_size = 0;
  std::optional<uint64_t> source_owner_descriptor_file_offset;
  ConSanPerturbationKind kind = ConSanPerturbationKind::None;
  ConSanPerturbationEdge edge = ConSanPerturbationEdge::Release;
  uint32_t sleep_imm = 0;
  bool overlaps_atomic_mutation = false;
  bool removed_cache_boundary = false;
  bool sequence_semantics_weakened = false;
};

void build_perturbation_plan(const ConSanOptions &options, ConSanResult &result,
                             std::span<const CarriedPerturbationPlan> carried_plans = {}) {
  result.requested_perturbations =
      options.sc_perturb_kind == ConSanPerturbationKind::None ? 0u : options.sc_perturb_max;
  if (options.sc_perturb_kind == ConSanPerturbationKind::None)
    return;
  if (options.flavor != ConSanFlavor::SuperCollider) {
    result.errors.emplace_back("ConSan SC perturbation requires the SuperCollider flavor");
    return;
  }
  if (options.sc_perturb_max == 0u || options.sc_perturb_max > 2u) {
    result.errors.emplace_back("ConSan SC perturb maximum must be from 1 through 2");
    return;
  }
  if (options.sc_perturb_sleep == 0u || options.sc_perturb_sleep > 15u) {
    result.errors.emplace_back("ConSan SC perturb sleep immediate must be from 1 through 15");
    return;
  }
  if (options.sc_perturb_required_count > options.sc_perturb_max) {
    result.errors.emplace_back("ConSan SC perturb required count exceeds the selected maximum");
    return;
  }

  if (!carried_plans.empty()) {
    if (carried_plans.size() > options.sc_perturb_max) {
      result.errors.emplace_back(
          "ConSan carried SC perturbation plans exceed the selected maximum");
      return;
    }
    for (const CarriedPerturbationPlan &carried : carried_plans) {
      if (carried.source_candidate_identity.empty() || carried.source_sequence_identity.empty() ||
          carried.container_name.empty() || carried.anchor_event_identity.empty() ||
          carried.ordered_member_identities.empty() ||
          carried.source_anchor_text_offset >
              std::numeric_limits<uint64_t>::max() - carried.anchor_size ||
          carried.anchor_size == 0u ||
          (carried.removed_cache_boundary && !carried.overlaps_atomic_mutation)) {
        result.errors.emplace_back("ConSan carried perturbation identity is incomplete");
        continue;
      }
      const auto matches_carried = [&](const ConSanPerturbationCandidate &candidate) {
        return candidate.eligible && candidate.kind == carried.kind &&
               candidate.edge == carried.edge &&
               candidate.container_name == carried.container_name &&
               candidate.in_kernel == carried.in_kernel &&
               candidate.anchor_text_offset == carried.translated_anchor_text_offset &&
               candidate.anchor_size == carried.anchor_size;
      };
      const size_t match_count = static_cast<size_t>(
          std::ranges::count_if(result.perturbation_candidates, matches_carried));
      auto candidate = std::ranges::find_if(result.perturbation_candidates, matches_carried);
      if ((carried.removed_cache_boundary || carried.sequence_semantics_weakened) &&
          match_count == 0u) {
        result.perturbation_candidates.push_back(
            {.kind = carried.kind,
             .edge = carried.edge,
             .identity = carried.source_candidate_identity,
             .sequence_identity = carried.source_sequence_identity,
             .container_name = carried.container_name,
             .in_kernel = carried.in_kernel,
             .basic_block_index = carried.basic_block_index,
             .anchor_event_identity = carried.anchor_event_identity,
             .anchor_text_offset = carried.translated_anchor_text_offset,
             .anchor_size = carried.anchor_size,
             .ordered_member_identities = carried.ordered_member_identities,
             .eligible = true,
             .rejection_reason = {}});
        candidate = std::prev(result.perturbation_candidates.end());
      } else if (match_count != 1u || candidate == result.perturbation_candidates.end()) {
        result.errors.emplace_back(
            "ConSan could not recover exactly one carried perturbation edge with its owner");
        continue;
      }
      result.perturbation_plans.push_back(
          {.candidate_identity = candidate->identity,
           .sequence_identity = candidate->sequence_identity,
           .kind = candidate->kind,
           .edge = candidate->edge,
           .anchor_event_identity = candidate->anchor_event_identity,
           .anchor_text_offset = candidate->anchor_text_offset,
           .anchor_size = candidate->anchor_size,
           .sleep_imm = carried.sleep_imm,
           .source_candidate_identity = carried.source_candidate_identity,
           .source_sequence_identity = carried.source_sequence_identity,
           .source_container_name = carried.container_name,
           .source_anchor_event_identity = carried.anchor_event_identity,
           .source_in_kernel = carried.in_kernel,
           .source_anchor_text_offset = carried.source_anchor_text_offset,
           .source_anchor_size = carried.anchor_size,
           .source_owner_descriptor_file_offset = carried.source_owner_descriptor_file_offset,
           .overlaps_atomic_mutation = carried.overlaps_atomic_mutation,
           .removed_cache_boundary = carried.removed_cache_boundary});
    }
    result.planned_perturbations = result.perturbation_plans.size();
    if (options.sc_perturb_required_count != 0u &&
        result.planned_perturbations != options.sc_perturb_required_count) {
      result.errors.emplace_back("ConSan SC perturb required " +
                                 std::to_string(options.sc_perturb_required_count) +
                                 " plans, got " + std::to_string(result.planned_perturbations));
    }
    return;
  }

  std::vector<const ConSanPerturbationCandidate *> matching;
  for (const ConSanPerturbationCandidate &candidate : result.perturbation_candidates) {
    if (!candidate.eligible || candidate.kind != options.sc_perturb_kind ||
        candidate.edge != options.sc_perturb_edge ||
        (!options.test_kernel_name_filter.empty() &&
         candidate.container_name.find(options.test_kernel_name_filter) == std::string::npos))
      continue;
    if (!options.sc_perturb_identity.empty() && candidate.identity != options.sc_perturb_identity)
      continue;
    matching.push_back(&candidate);
  }

  const size_t begin = options.sc_perturb_identity.empty() ? options.sc_perturb_index : 0u;
  for (size_t i = begin;
       i < matching.size() && result.perturbation_plans.size() < options.sc_perturb_max; ++i) {
    const ConSanPerturbationCandidate &candidate = *matching[i];
    result.perturbation_plans.push_back({.candidate_identity = candidate.identity,
                                         .sequence_identity = candidate.sequence_identity,
                                         .kind = candidate.kind,
                                         .edge = candidate.edge,
                                         .anchor_event_identity = candidate.anchor_event_identity,
                                         .anchor_text_offset = candidate.anchor_text_offset,
                                         .anchor_size = candidate.anchor_size,
                                         .sleep_imm = options.sc_perturb_sleep,
                                         .source_candidate_identity = {},
                                         .source_sequence_identity = {},
                                         .source_container_name = {},
                                         .source_anchor_event_identity = {},
                                         .source_in_kernel = true,
                                         .source_anchor_text_offset = 0,
                                         .source_anchor_size = 0,
                                         .source_owner_descriptor_file_offset = std::nullopt,
                                         .overlaps_atomic_mutation = false,
                                         .removed_cache_boundary = false});
  }
  result.planned_perturbations = result.perturbation_plans.size();
  if (options.sc_perturb_required_count != 0u &&
      result.planned_perturbations != options.sc_perturb_required_count) {
    result.errors.emplace_back("ConSan SC perturb required " +
                               std::to_string(options.sc_perturb_required_count) + " plans, got " +
                               std::to_string(result.planned_perturbations));
  }
}

void try_apply_perturbation_patches(const AmdGpuCodeObject &code_object, rj_code_arch_t arch,
                                    ConSanResult &result) {
  if (result.perturbation_plans.empty())
    return;
  if (code_object.text_sections().size() != 1u) {
    result.outcome = ConSanTransformOutcome::Unsupported;
    result.warnings.emplace_back(
        "ConSan SC perturbation requires exactly one executable text section");
    return;
  }

  std::unique_ptr<AmdGpuCodeObject> composed_code_object;
  const AmdGpuCodeObject *emission_code_object = &code_object;
  if (!result.elf_bytes.empty()) {
    composed_code_object =
        std::make_unique<AmdGpuCodeObject>(result.elf_bytes.data(), result.elf_bytes.size());
    if (!composed_code_object->is_valid() || composed_code_object->text_sections().size() != 1u) {
      result.errors.emplace_back(
          "ConSan SC perturbation could not parse the composed redundant-access image");
      return;
    }
    emission_code_object = composed_code_object.get();
  }
  CodeObjectPatcher patcher(*emission_code_object);
  const std::span<const uint8_t> text = patcher.text_bytes();
  const std::vector<LocalNopCave> caves = find_uncovered_nop_caves(code_object, result, arch);
  DbiPatchPlacementPlanner placement_planner(arch, text.size());
  for (const ConSanPatchInfo &existing : result.patches) {
    std::string reservation_error;
    if (!placement_planner.reserve_existing_range(existing.anchor_offset, existing.original_size,
                                                  &reservation_error) ||
        (existing.trampoline_size != 0u &&
         !placement_planner.reserve_existing_range(existing.trampoline_offset,
                                                   existing.trampoline_size, &reservation_error))) {
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          "ConSan SC perturbation could not reserve the composed patch image: " +
          reservation_error);
      return;
    }
  }
  struct PlannedPatch {
    const ConSanPerturbationPlan *plan = nullptr;
    const ConSanPerturbationCandidate *candidate = nullptr;
    DbiPatchPlacement placement;
    std::vector<uint32_t> body_words;
  };
  std::vector<PlannedPatch> planned;
  planned.reserve(result.perturbation_plans.size());
  const uint32_t nop = build_s_nop(0, arch);

  for (const ConSanPerturbationPlan &plan : result.perturbation_plans) {
    const auto candidate =
        std::ranges::find(result.perturbation_candidates, plan.candidate_identity,
                          &ConSanPerturbationCandidate::identity);
    if (candidate == result.perturbation_candidates.end() || !candidate->eligible ||
        candidate->sequence_identity != plan.sequence_identity ||
        candidate->anchor_event_identity != plan.anchor_event_identity ||
        candidate->anchor_text_offset != plan.anchor_text_offset ||
        candidate->anchor_size != plan.anchor_size) {
      result.errors.emplace_back(
          "ConSan SC perturbation plan no longer matches its admitted candidate");
      return;
    }
    if (plan.anchor_size != sizeof(uint32_t) && plan.anchor_size != 2u * sizeof(uint32_t) &&
        plan.anchor_size != 3u * sizeof(uint32_t)) {
      result.outcome = ConSanTransformOutcome::Unsupported;
      result.warnings.emplace_back(
          "ConSan SC perturbation supports only 4-, 8-, and 12-byte boundary instructions");
      return;
    }
    if (plan.anchor_text_offset > text.size() ||
        plan.anchor_size > text.size() - plan.anchor_text_offset) {
      result.errors.emplace_back("ConSan SC perturbation anchor exceeds executable text");
      return;
    }
    std::array<uint32_t, 3> original_words{};
    std::memcpy(original_words.data(), text.data() + plan.anchor_text_offset, plan.anchor_size);
    if (plan.removed_cache_boundary) {
      const bool all_nops = std::ranges::all_of(
          std::span<const uint32_t>(original_words.data(), plan.anchor_size / sizeof(uint32_t)),
          [&](uint32_t word) { return word == nop; });
      if (!plan.overlaps_atomic_mutation || !all_nops) {
        result.errors.emplace_back(
            "ConSan carried removed-cache perturbation boundary is not the staged NOP range");
        return;
      }
    } else {
      std::unique_ptr<Instruction> original;
      try {
        std::unique_ptr<Decoder> decoder = Decoder::create(arch);
        if (decoder)
          original.reset(decoder->decode(original_words.data(), plan.anchor_text_offset));
      } catch (...) {
        original.reset();
      }
      std::string relocatable_error;
      if (!original || static_cast<uint32_t>(original->size()) != plan.anchor_size ||
          !is_relocatable_barrier_move_destination(*original, plan.anchor_text_offset, text, arch,
                                                   &relocatable_error)) {
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back("ConSan SC perturbation boundary is not relocatable: " +
                                     relocatable_error);
        return;
      }
    }

    DbiPatchPlacementRequest request;
    request.anchor_offset = plan.anchor_text_offset;
    request.original_size = plan.anchor_size;
    request.body_size = sizeof(uint32_t) + plan.anchor_size;
    request.allow_appended_cave = false;
    std::optional<DbiPatchPlacement> placement;
    for (const LocalNopCave &cave : caves) {
      const uint64_t cave_end =
          cave.text_offset + static_cast<uint64_t>(cave.word_count) * sizeof(uint32_t);
      const uint64_t reservation_size = request.body_size + sizeof(uint32_t);
      for (uint64_t offset = cave.text_offset;
           offset <= cave_end && reservation_size <= cave_end - offset;
           offset += sizeof(uint32_t)) {
        request.local_cave = DbiPatchLocalCave{offset, cave_end - offset};
        placement = placement_planner.plan(request);
        if (placement)
          break;
      }
      if (placement)
        break;
    }
    if (!placement) {
      request.local_cave.reset();
      request.allow_appended_cave = true;
      std::string placement_error;
      placement = placement_planner.plan(request, &placement_error);
      if (!placement) {
        result.outcome = ConSanTransformOutcome::Unsupported;
        result.warnings.emplace_back(
            "ConSan SC perturbation has no reachable local or appended cave: " + placement_error);
        return;
      }
    }

    const auto ret =
        compute_sopp_branch_simm16(placement->return_branch_offset, placement->return_target);
    if (!ret) {
      result.errors.emplace_back("ConSan SC perturbation return branch exceeds s_branch simm16");
      return;
    }
    std::vector<uint32_t> body_words;
    body_words.reserve(plan.anchor_size / sizeof(uint32_t) + 2u);
    if (plan.edge == ConSanPerturbationEdge::Release)
      body_words.push_back(build_s_sleep(static_cast<uint16_t>(plan.sleep_imm), arch));
    body_words.insert(body_words.end(), original_words.begin(),
                      original_words.begin() + plan.anchor_size / sizeof(uint32_t));
    if (plan.edge == ConSanPerturbationEdge::Acquire)
      body_words.push_back(build_s_sleep(static_cast<uint16_t>(plan.sleep_imm), arch));
    body_words.push_back(build_s_branch(*ret, arch));
    planned.push_back({.plan = &plan,
                       .candidate = &*candidate,
                       .placement = *placement,
                       .body_words = std::move(body_words)});
  }

  std::vector<uint8_t> new_text(text.begin(), text.end());
  for (const PlannedPatch &patch : planned) {
    const auto forward =
        compute_sopp_branch_simm16(patch.placement.anchor_offset, patch.placement.body_offset);
    if (!forward) {
      result.errors.emplace_back("ConSan SC perturbation forward branch exceeds s_branch simm16");
      return;
    }
    const uint32_t branch = build_s_branch(*forward, arch);
    std::memcpy(new_text.data() + patch.placement.anchor_offset, &branch, sizeof(branch));
    for (uint32_t offset = sizeof(uint32_t); offset < patch.plan->anchor_size;
         offset += sizeof(uint32_t)) {
      std::memcpy(new_text.data() + patch.placement.anchor_offset + offset, &nop, sizeof(nop));
    }
    const uint32_t trampoline_size =
        static_cast<uint32_t>(patch.body_words.size() * sizeof(uint32_t));
    if (patch.placement.kind == DbiPatchPlacementKind::AppendedCave) {
      if (new_text.size() != patch.placement.body_offset) {
        result.errors.emplace_back("ConSan SC perturbation emitted a stale appended-cave mapping");
        return;
      }
      append_words(new_text, patch.body_words);
    } else {
      std::memcpy(new_text.data() + patch.placement.body_offset, patch.body_words.data(),
                  trampoline_size);
    }
  }
  if (!patcher.replace_text(new_text, kConSanMaxPatchedFileGrowth)) {
    result.errors.emplace_back("ConSan SC perturbation could not replace executable text");
    return;
  }
  result.elf_bytes = patcher.emit();
  for (const PlannedPatch &patch : planned) {
    ConSanPatchInfo info;
    info.kind = ConSanPatchKind::TrampolineScPerturbation;
    info.anchor_offset = patch.placement.anchor_offset;
    info.trampoline_offset = patch.placement.body_offset;
    info.original_size = patch.plan->anchor_size;
    info.trampoline_size = static_cast<uint32_t>(patch.body_words.size() * sizeof(uint32_t));
    info.perturbation_edge = patch.plan->edge;
    info.perturbation_sequence_identity = patch.plan->sequence_identity;
    info.perturbation_source_candidate_identity = patch.plan->source_candidate_identity;
    info.perturbation_source_sequence_identity = patch.plan->source_sequence_identity;
    info.perturbation_source_container_name = patch.plan->source_container_name;
    info.perturbation_source_anchor_identity = patch.plan->source_anchor_event_identity.empty()
                                                   ? patch.plan->anchor_event_identity
                                                   : patch.plan->source_anchor_event_identity;
    info.perturbation_source_in_kernel = patch.plan->source_in_kernel;
    info.perturbation_source_anchor_offset = patch.plan->source_anchor_text_offset;
    info.perturbation_source_anchor_size = patch.plan->source_anchor_size;
    info.perturbation_source_owner_descriptor_file_offset =
        patch.plan->source_owner_descriptor_file_offset;
    info.perturbation_composite_atomic_overlap = patch.plan->overlaps_atomic_mutation;
    info.perturbation_composite_removed_boundary = patch.plan->removed_cache_boundary;
    if (patch.candidate->in_kernel) {
      const auto owner = std::ranges::find(result.kernels, patch.candidate->container_name,
                                           &ConSanKernelInfo::name);
      if (owner == result.kernels.end()) {
        result.errors.emplace_back("ConSan carried perturbation lost its exact kernel owner");
        result.elf_bytes.clear();
        return;
      }
      info.owner_descriptor_file_offsets.push_back(owner->descriptor_file_offset);
    }
    result.patches.push_back(std::move(info));
  }
  result.applied_perturbations = planned.size();
  result.modified = !result.patches.empty();
}

void annotate_fault_sites_with_sync_sequences(ConSanResult &result) {
  struct EventKey {
    ConSanSyncEventKind kind;
    bool in_kernel;
    std::string_view container_name;
    uint64_t text_offset;

    bool operator==(const EventKey &) const = default;
  };
  struct EventKeyHash {
    size_t operator()(const EventKey &key) const {
      size_t value = std::hash<std::string_view>{}(key.container_name);
      const auto combine = [&](size_t part) {
        value ^= part + 0x9e3779b9u + (value << 6u) + (value >> 2u);
      };
      combine(std::hash<uint64_t>{}(key.text_offset));
      combine(std::hash<uint8_t>{}(static_cast<uint8_t>(key.kind)));
      combine(std::hash<bool>{}(key.in_kernel));
      return value;
    }
  };

  // Aggregate code objects can contain tens of thousands of fault sites and
  // synchronization events with long mangled owner names. A linear event
  // search per site makes this annotation quadratic and spends most of its
  // time comparing those names.
  std::unordered_map<EventKey, const ConSanSyncEvent *, EventKeyHash> event_by_site;
  event_by_site.reserve(result.sync_events.size());
  struct PhysicalEventKey {
    ConSanSyncEventKind kind;
    uint64_t text_offset;

    bool operator==(const PhysicalEventKey &) const = default;
  };
  struct PhysicalEventKeyHash {
    size_t operator()(const PhysicalEventKey &key) const {
      size_t value = std::hash<uint64_t>{}(key.text_offset);
      value ^= std::hash<uint8_t>{}(static_cast<uint8_t>(key.kind)) + 0x9e3779b9u +
               (value << 6u) + (value >> 2u);
      return value;
    }
  };
  std::unordered_map<PhysicalEventKey, const ConSanSyncEvent *, PhysicalEventKeyHash>
      event_by_physical_site;
  event_by_physical_site.reserve(result.sync_events.size());
  for (const ConSanSyncEvent &event : result.sync_events) {
    event_by_site.emplace(
        EventKey{event.kind, event.in_kernel, event.container_name, event.text_offset}, &event);
    const auto [entry, inserted] = event_by_physical_site.emplace(
        PhysicalEventKey{event.kind, event.text_offset}, &event);
    if (!inserted)
      entry->second = nullptr;
  }

  size_t sequence_member_count = 0;
  for (const ConSanSyncSequence &sequence : result.sync_sequences)
    sequence_member_count += sequence.member_event_identities.size();
  std::unordered_map<std::string_view, const ConSanSyncSequence *> sequence_by_event;
  sequence_by_event.reserve(sequence_member_count);
  for (const ConSanSyncSequence &sequence : result.sync_sequences) {
    for (const std::string &identity : sequence.member_event_identities) {
      const auto [entry, inserted] = sequence_by_event.emplace(identity, &sequence);
      if (!inserted)
        entry->second = nullptr;
    }
  }

  for (ConSanFaultSite &site : result.fault_sites) {
    const ConSanSyncEventKind expected_kind =
        site.kind == ConSanFaultSiteKind::Barrier  ? ConSanSyncEventKind::Barrier
        : site.kind == ConSanFaultSiteKind::Atomic ? ConSanSyncEventKind::Atomic
                                                   : ConSanSyncEventKind::OrdinaryMemory;
    const auto exact_event = event_by_site.find(
        EventKey{expected_kind, site.in_kernel, site.container_name, site.text_offset});
    const ConSanSyncEvent *event =
        exact_event == event_by_site.end() ? nullptr : exact_event->second;
    if (event == nullptr) {
      const auto physical_event =
          event_by_physical_site.find(PhysicalEventKey{expected_kind, site.text_offset});
      if (physical_event != event_by_physical_site.end())
        event = physical_event->second;
    }
    if (event == nullptr)
      continue;
    site.sync_event_identity = event->identity;
    const auto sequence_entry = sequence_by_event.find(event->identity);
    const ConSanSyncSequence *sequence =
        sequence_entry == sequence_by_event.end() ? nullptr : sequence_entry->second;
    if (sequence == nullptr)
      continue;
    site.sync_sequence_identity = sequence->identity;
    site.sync_confidence = sequence->confidence;
    site.sync_memory_role = sequence->memory_role;
    switch (sequence->operation) {
    case ConSanSyncOperation::BarrierSignal:
      site.semantic_role = "barrier-signal";
      break;
    case ConSanSyncOperation::BarrierWait:
      site.semantic_role = "barrier-wait";
      break;
    case ConSanSyncOperation::BarrierFull:
      site.semantic_role = "workgroup-barrier";
      break;
    case ConSanSyncOperation::BarrierInit:
      site.semantic_role = "barrier-init-unsupported";
      break;
    case ConSanSyncOperation::BarrierJoin:
      site.semantic_role = "barrier-join-unsupported";
      break;
    case ConSanSyncOperation::BarrierLeave:
      site.semantic_role = "barrier-leave-unsupported";
      break;
    case ConSanSyncOperation::BarrierWakeup:
      site.semantic_role = "barrier-wakeup-unsupported";
      break;
    case ConSanSyncOperation::BarrierStateQuery:
      site.semantic_role = "barrier-state-query-unsupported";
      break;
    case ConSanSyncOperation::AtomicRmw:
    case ConSanSyncOperation::AtomicCompareExchange:
      switch (sequence->memory_role) {
      case ConSanSyncMemoryRole::Acquire:
        site.semantic_role = "atomic-acquire";
        break;
      case ConSanSyncMemoryRole::Release:
        site.semantic_role = "atomic-release";
        break;
      case ConSanSyncMemoryRole::AcquireRelease:
        site.semantic_role = "atomic-acquire-release";
        break;
      case ConSanSyncMemoryRole::SequentiallyConsistent:
        site.semantic_role = "atomic-sequentially-consistent";
        break;
      case ConSanSyncMemoryRole::Unknown:
      case ConSanSyncMemoryRole::None:
        site.semantic_role = "atomic-order-unknown";
        break;
      }
      break;
    case ConSanSyncOperation::OrdinaryLoad:
      site.semantic_role = sequence->memory_role == ConSanSyncMemoryRole::Acquire
                               ? "ordinary-acquire-load"
                               : "ordinary-load";
      break;
    case ConSanSyncOperation::OrdinaryStore:
      site.semantic_role = sequence->memory_role == ConSanSyncMemoryRole::Release
                               ? "ordinary-release-store"
                               : "ordinary-store";
      break;
    case ConSanSyncOperation::Unknown:
    case ConSanSyncOperation::Fence:
      break;
    }
  }
}

[[nodiscard]] ConSanOwnerProofKind owner_proof_kind(KernelCfgOwnerProofKind kind) {
  switch (kind) {
  case KernelCfgOwnerProofKind::KernelLocal:
    return ConSanOwnerProofKind::KernelLocal;
  case KernelCfgOwnerProofKind::DirectCall:
    return ConSanOwnerProofKind::DirectCall;
  case KernelCfgOwnerProofKind::RecoveredIndirectCall:
    return ConSanOwnerProofKind::RecoveredIndirectCall;
  }
  return ConSanOwnerProofKind::RecoveredIndirectCall;
}

[[nodiscard]] std::vector<ConSanExecutionOwner>
execution_owners_for_offset(
    const std::unordered_map<const BasicBlock *, std::vector<ConSanExecutionOwner>> &owners_by_block,
    const BlockOffsetIndex &block_index, uint64_t offset) {
  const BasicBlock *block = block_for_offset(block_index, offset);
  if (block == nullptr)
    return {};
  const auto owners = owners_by_block.find(block);
  return owners == owners_by_block.end() ? std::vector<ConSanExecutionOwner>{} : owners->second;
}

[[nodiscard]] std::vector<ConSanExecutionOwner>
intersect_execution_owners(std::span<const ConSanExecutionOwner> lhs,
                           std::span<const ConSanExecutionOwner> rhs) {
  std::vector<ConSanExecutionOwner> intersection;
  for (const ConSanExecutionOwner &left : lhs) {
    const auto right = std::ranges::find(rhs, left.descriptor_file_offset,
                                         &ConSanExecutionOwner::descriptor_file_offset);
    if (right == rhs.end())
      continue;
    intersection.push_back(
        {.descriptor_file_offset = left.descriptor_file_offset,
         .proof = static_cast<uint8_t>(left.proof) >= static_cast<uint8_t>(right->proof)
                      ? left.proof
                      : right->proof});
  }
  return intersection;
}

void annotate_execution_owners(const AmdGpuCodeObject &code_object, Decoder &decoder,
                               rj_code_arch_t arch, const ConSanOptions &options,
                               const std::vector<std::unique_ptr<BasicBlock>> *reusable_blocks,
                               ConSanResult &result) {
  std::vector<uint64_t> leaders;
  std::vector<uint64_t> kernel_entries;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;
    leaders.push_back(kernel.entry_text_offset);
    kernel_entries.push_back(kernel.entry_text_offset);
  }
  for (const ConSanFunctionInfo &function : result.functions)
    leaders.push_back(function.entry_text_offset);
  for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
    leaders.push_back(range.text_offset);
    if (range.continuation_text_offset)
      leaders.push_back(*range.continuation_text_offset);
  }
  std::ranges::sort(leaders);
  leaders.erase(std::ranges::unique(leaders).begin(), leaders.end());
  std::ranges::sort(kernel_entries);
  kernel_entries.erase(std::ranges::unique(kernel_entries).begin(), kernel_entries.end());

  std::vector<BasicBlock::CodeRange> code_ranges;
  for (const AmdGpuFunctionInfo &function : code_object.functions()) {
    if (function.code_size != 0)
      code_ranges.push_back(
          {.start_offset = function.entry_text_offset, .size = function.code_size});
  }
  for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
    if (range.size != 0)
      code_ranges.push_back({.start_offset = range.text_offset, .size = range.size});
  }
  std::vector<std::unique_ptr<BasicBlock>> rebuilt_blocks;
  if (reusable_blocks == nullptr || !options.preapplied_code_ranges.empty()) {
    rebuilt_blocks = BasicBlock::build(code_object, decoder, arch, leaders, code_ranges);
    reusable_blocks = &rebuilt_blocks;
  }
  const auto &blocks = *reusable_blocks;
  const BlockOffsetIndex block_index = build_block_offset_index(blocks);
  CodeObjectPatcher patcher(code_object);
  const std::span<const uint8_t> text = patcher.text_bytes();
  std::unordered_map<const BasicBlock *, std::vector<ConSanExecutionOwner>> owners_by_block;
  owners_by_block.reserve(blocks.size());
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (!kernel.has_text_range)
      continue;
    std::vector<uint64_t> additional_entry_offsets;
    for (const ConSanPreappliedCodeRange &range : options.preapplied_code_ranges) {
      if (range.kernel_name != kernel.name)
        continue;
      if (block_for_offset(block_index, range.text_offset) != nullptr)
        additional_entry_offsets.push_back(range.text_offset);
      if (range.continuation_text_offset &&
          block_for_offset(block_index, *range.continuation_text_offset) != nullptr)
        additional_entry_offsets.push_back(*range.continuation_text_offset);
    }
    auto scope = build_kernel_cfg_scope(
        blocks, block_index,
        KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                           .additional_entry_offsets = additional_entry_offsets},
        kernel_entries, text);
    if (scope) {
      for (const auto &[block, proof] : scope->owner_proofs) {
        owners_by_block[block].push_back(
            {.descriptor_file_offset = kernel.descriptor_file_offset,
             .proof = owner_proof_kind(proof)});
      }
    }
  }

  for (ConSanFaultSite &site : result.fault_sites)
    site.execution_owners =
        execution_owners_for_offset(owners_by_block, block_index, site.text_offset);
  for (ConSanSyncEvent &event : result.sync_events)
    event.execution_owners =
        execution_owners_for_offset(owners_by_block, block_index, event.text_offset);
  for (ConSanBarrierMoveDestination &destination : result.barrier_move_destinations) {
    destination.execution_owners =
        execution_owners_for_offset(owners_by_block, block_index, destination.text_offset);
  }
  const auto annotate_memory_sites = [&](auto &containers) {
    for (auto &container : containers) {
      for (ConSanLdsSite &site : container.lds_sites) {
        const auto owners =
            execution_owners_for_offset(owners_by_block, block_index, site.text_offset);
        site.owner_descriptor_file_offsets.clear();
        site.owner_descriptor_file_offsets.reserve(owners.size());
        for (const ConSanExecutionOwner &owner : owners)
          site.owner_descriptor_file_offsets.push_back(owner.descriptor_file_offset);
      }
      for (ConSanFlatSite &site : container.flat_sites) {
        const auto owners =
            execution_owners_for_offset(owners_by_block, block_index, site.text_offset);
        site.owner_descriptor_file_offsets.clear();
        site.owner_descriptor_file_offsets.reserve(owners.size());
        for (const ConSanExecutionOwner &owner : owners)
          site.owner_descriptor_file_offsets.push_back(owner.descriptor_file_offset);
      }
    }
  };
  annotate_memory_sites(result.kernels);
  annotate_memory_sites(result.functions);

  // A large aggregate code object can contain hundreds of thousands of sync
  // events and sequences. Looking up every sequence member with a linear scan
  // makes owner annotation quadratic in the sync inventory size.
  std::unordered_map<std::string_view, const ConSanSyncEvent *> sync_event_by_identity;
  sync_event_by_identity.reserve(result.sync_events.size());
  for (const ConSanSyncEvent &event : result.sync_events)
    sync_event_by_identity.emplace(event.identity, &event);
  for (ConSanSyncSequence &sequence : result.sync_sequences) {
    bool first = true;
    for (const std::string &identity : sequence.member_event_identities) {
      const auto event = sync_event_by_identity.find(identity);
      if (event == sync_event_by_identity.end()) {
        sequence.execution_owners.clear();
        first = false;
        break;
      }
      if (first) {
        sequence.execution_owners = event->second->execution_owners;
        first = false;
      } else {
        sequence.execution_owners =
            intersect_execution_owners(sequence.execution_owners, event->second->execution_owners);
      }
    }
  }
}

[[nodiscard]] const ConSanFaultSite *find_fault_site_by_identity(const ConSanResult &result,
                                                                 std::string_view identity,
                                                                 ConSanFaultSiteKind kind) {
  const auto site = std::ranges::find_if(result.fault_sites, [&](const ConSanFaultSite &candidate) {
    return candidate.kind == kind && candidate.identity == identity;
  });
  return site == result.fault_sites.end() ? nullptr : &*site;
}

[[nodiscard]] BarrierSite barrier_site_from_inventory(const ConSanFaultSite &site) {
  return {.text_offset = site.text_offset,
          .file_offset = site.file_offset,
          .size = site.size,
          .range_name = std::string(site.in_kernel ? "kernel:" : "function:") + site.container_name,
          .mnemonic = site.mnemonic};
}

[[nodiscard]] const ConSanKernelInfo *requested_kernel_owner(const ConSanResult &result,
                                                             std::string_view filter) {
  if (filter.empty())
    return nullptr;
  const auto exact = std::ranges::find(result.kernels, filter, &ConSanKernelInfo::name);
  if (exact != result.kernels.end())
    return &*exact;
  const ConSanKernelInfo *match = nullptr;
  for (const ConSanKernelInfo &kernel : result.kernels) {
    if (kernel.name.find(filter) == std::string::npos)
      continue;
    if (match != nullptr)
      return nullptr;
    match = &kernel;
  }
  return match;
}

[[nodiscard]] bool owned_by_requested_kernel(std::span<const ConSanExecutionOwner> owners,
                                             const ConSanResult &result,
                                             const ConSanOptions &options) {
  if (options.test_kernel_name_filter.empty())
    return true;
  const ConSanKernelInfo *kernel = requested_kernel_owner(result, options.test_kernel_name_filter);
  return kernel != nullptr &&
         std::ranges::find(owners, kernel->descriptor_file_offset,
                           &ConSanExecutionOwner::descriptor_file_offset) != owners.end();
}

[[nodiscard]] const ConSanFaultSite *select_fault_site_for_plan(const ConSanResult &result,
                                                                const ConSanOptions &options,
                                                                ConSanFaultSiteKind kind,
                                                                uint32_t legacy_index) {
  if (!options.fault_site_identity.empty()) {
    const ConSanFaultSite *site =
        find_fault_site_by_identity(result, options.fault_site_identity, kind);
    const bool patchable = site != nullptr;
    return patchable && owned_by_requested_kernel(site->execution_owners, result, options)
               ? site
               : nullptr;
  }
  uint32_t index = 0;
  for (const ConSanFaultSite &site : result.fault_sites) {
    if (site.kind != kind)
      continue;
    if (!owned_by_requested_kernel(site.execution_owners, result, options))
      continue;
    if (index++ == legacy_index)
      return &site;
  }
  return nullptr;
}
