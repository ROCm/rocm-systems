// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Kind of original program container through which an instruction was
/// decoded.
///
/// Multiple symbols can alias the same physical instruction. The container is
/// therefore attribution and ownership evidence, not part of
/// `PhysicalSiteId` and not permission to instrument the instruction twice.
enum class ConSanProgramContainerKind : uint8_t {
  /// A dispatchable kernel descriptor and its executable text range.
  Kernel,
  /// A non-kernel function symbol, commonly a helper reached by one or more
  /// kernels.
  Function,
  /// One past the last kind. This is an iteration bound and never a valid
  /// container reference.
  Count,
};

/// Complete iterable set of original-program container kinds.
inline constexpr std::array<ConSanProgramContainerKind, 2> kConSanProgramContainerKinds = {
    ConSanProgramContainerKind::Kernel,
    ConSanProgramContainerKind::Function,
};

/// Return the stable diagnostic spelling of an original-program container
/// kind. Invalid values never default to kernel or function attribution.
[[nodiscard]] constexpr std::string_view
consan_program_container_kind_name(ConSanProgramContainerKind kind) {
  switch (kind) {
  case ConSanProgramContainerKind::Kernel:
    return "kernel";
  case ConSanProgramContainerKind::Function:
    return "function";
  case ConSanProgramContainerKind::Count:
    break;
  }
  return "invalid-program-container-kind";
}

static_assert(kConSanProgramContainerKinds.size() ==
              static_cast<size_t>(ConSanProgramContainerKind::Count));

/// Attribution reference from a semantic site to the symbol range through
/// which it was decoded.
///
/// Kernel descriptor identity is optional because function aliases do not own
/// a descriptor. Execution owners are kept separately: a shared function can
/// be decoded through one symbol yet execute for several kernel descriptors.
struct ConSanProgramContainerRef {
  /// Whether the symbol is a dispatchable kernel or a helper function.
  ConSanProgramContainerKind kind = ConSanProgramContainerKind::Count;
  /// Symbol name retained for diagnostics and temporary legacy lookup.
  std::string name;
  /// Original executable-text entry offset of the symbol range.
  uint64_t entry_text_offset = 0;
  /// Descriptor file offset for a kernel; absent for a function.
  std::optional<uint64_t> kernel_descriptor_file_offset;
  /// Whether this container's code observes the gfx1250 cluster workgroup ID.
  bool uses_gfx1250_cluster_workgroup_id = false;

  bool operator==(const ConSanProgramContainerRef &) const = default;
};

/// Instruction family from which one normalized access fact originated.
///
/// This is decoder provenance, not engine applicability. `DirectToLds` names
/// a global-memory instruction whose architectural destination is LDS; it is
/// kept distinct because its address and payload operands differ from an
/// ordinary DS instruction even though both access group memory.
enum class ConSanAccessOrigin : uint8_t {
  /// Native DS/LDS instruction.
  NativeLds,
  /// FLAT instruction with separately classified address-space provenance.
  Flat,
  /// Global-memory operation carrying a direct-to-LDS modifier.
  DirectToLds,
  /// One past the last origin. This is an iteration bound and never a decoded
  /// access origin.
  Count,
};

/// Complete iterable set of decoded access origins.
inline constexpr std::array<ConSanAccessOrigin, 3> kConSanAccessOrigins = {
    ConSanAccessOrigin::NativeLds,
    ConSanAccessOrigin::Flat,
    ConSanAccessOrigin::DirectToLds,
};

/// Return the stable diagnostic spelling of an access-origin fact.
[[nodiscard]] constexpr std::string_view consan_access_origin_name(ConSanAccessOrigin origin) {
  switch (origin) {
  case ConSanAccessOrigin::NativeLds:
    return "native-lds";
  case ConSanAccessOrigin::Flat:
    return "flat";
  case ConSanAccessOrigin::DirectToLds:
    return "direct-to-lds";
  case ConSanAccessOrigin::Count:
    break;
  }
  return "invalid-access-origin";
}

static_assert(kConSanAccessOrigins.size() == static_cast<size_t>(ConSanAccessOrigin::Count));

/// Normalized address-space fact relevant to ConSan access policy.
///
/// The inventory distinguishes group memory, definitely non-group memory, and
/// unresolved provenance without prematurely applying an engine's strict or
/// likely admission policy.
enum class ConSanAccessAddressSpace : uint8_t {
  /// The access targets workgroup/cluster-visible LDS.
  Group,
  /// The access is known to target private, global, or another non-group
  /// address space.
  NonGroup,
  /// Analysis could not prove whether a FLAT access reaches group memory.
  Unresolved,
  /// One past the last address-space fact. This is an iteration bound and
  /// never a decoded value.
  Count,
};

/// Complete iterable set of normalized access address-space facts.
inline constexpr std::array<ConSanAccessAddressSpace, 3> kConSanAccessAddressSpaces = {
    ConSanAccessAddressSpace::Group,
    ConSanAccessAddressSpace::NonGroup,
    ConSanAccessAddressSpace::Unresolved,
};

/// Return the stable diagnostic spelling of a normalized address-space fact.
[[nodiscard]] constexpr std::string_view
consan_access_address_space_name(ConSanAccessAddressSpace address_space) {
  switch (address_space) {
  case ConSanAccessAddressSpace::Group:
    return "group";
  case ConSanAccessAddressSpace::NonGroup:
    return "non-group";
  case ConSanAccessAddressSpace::Unresolved:
    return "unresolved";
  case ConSanAccessAddressSpace::Count:
    break;
  }
  return "invalid-access-address-space";
}

static_assert(kConSanAccessAddressSpaces.size() ==
              static_cast<size_t>(ConSanAccessAddressSpace::Count));

/// Source of the normalized access address-space claim.
///
/// Confidence is stored separately because the same provenance mechanism can
/// yield an exact or conservative result. Consumers may select a confidence
/// threshold, but they may not rewrite this provenance to make a mode succeed.
enum class ConSanAccessProvenance : uint8_t {
  /// The instruction family itself fixes the access to LDS.
  NativeLdsOpcode,
  /// An encoded FLAT segment field fixes the address space.
  EncodedFlatSegment,
  /// Bounded pointer-flow analysis inferred the likely FLAT address space.
  PropagatedFlatPointer,
  /// Neither encoding nor pointer-flow analysis resolved the address space.
  UnresolvedFlatPointer,
  /// One past the last provenance source. This is an iteration bound and never
  /// a decoded source.
  Count,
};

/// Complete iterable set of normalized access-provenance sources.
inline constexpr std::array<ConSanAccessProvenance, 4> kConSanAccessProvenances = {
    ConSanAccessProvenance::NativeLdsOpcode,
    ConSanAccessProvenance::EncodedFlatSegment,
    ConSanAccessProvenance::PropagatedFlatPointer,
    ConSanAccessProvenance::UnresolvedFlatPointer,
};

/// Return the stable diagnostic spelling of an access-provenance source.
[[nodiscard]] constexpr std::string_view
consan_access_provenance_name(ConSanAccessProvenance provenance) {
  switch (provenance) {
  case ConSanAccessProvenance::NativeLdsOpcode:
    return "native-lds-opcode";
  case ConSanAccessProvenance::EncodedFlatSegment:
    return "encoded-flat-segment";
  case ConSanAccessProvenance::PropagatedFlatPointer:
    return "propagated-flat-pointer";
  case ConSanAccessProvenance::UnresolvedFlatPointer:
    return "unresolved-flat-pointer";
  case ConSanAccessProvenance::Count:
    break;
  }
  return "invalid-access-provenance";
}

static_assert(kConSanAccessProvenances.size() ==
              static_cast<size_t>(ConSanAccessProvenance::Count));

/// Typed reason why part of a decoded access cannot participate in a complete
/// normalized semantic inventory.
///
/// These are analysis facts, not engine rejections. An engine-specific
/// unsupported mnemonic, sampling decision, or resource failure belongs to a
/// later policy or lowering result rather than this enum.
enum class ConSanInventoryExclusionReason : uint8_t {
  /// The decoded instruction is not an access kind modeled by ConSan.
  NonAccessInstruction,
  /// The instruction size is zero, unaligned, or otherwise unusable.
  InvalidInstructionSize,
  /// The decoded access width cannot describe a nonempty byte range.
  InvalidAccessWidth,
  /// An address-bearing form lacks its decoded address operand.
  MissingAddressOperand,
  /// A multi-range instruction's encoded offsets were outside the input image
  /// or could not be normalized.
  RangeEncodingUnavailable,
  /// One past the last reason. This is an iteration bound and never an
  /// exclusion record.
  Count,
};

/// Complete iterable set of inventory-level access limitations.
inline constexpr std::array<ConSanInventoryExclusionReason, 5> kConSanInventoryExclusionReasons = {
    ConSanInventoryExclusionReason::NonAccessInstruction,
    ConSanInventoryExclusionReason::InvalidInstructionSize,
    ConSanInventoryExclusionReason::InvalidAccessWidth,
    ConSanInventoryExclusionReason::MissingAddressOperand,
    ConSanInventoryExclusionReason::RangeEncodingUnavailable,
};

/// Return the stable diagnostic spelling of an inventory-level limitation.
[[nodiscard]] constexpr std::string_view
consan_inventory_exclusion_reason_name(ConSanInventoryExclusionReason reason) {
  switch (reason) {
  case ConSanInventoryExclusionReason::NonAccessInstruction:
    return "non-access-instruction";
  case ConSanInventoryExclusionReason::InvalidInstructionSize:
    return "invalid-instruction-size";
  case ConSanInventoryExclusionReason::InvalidAccessWidth:
    return "invalid-access-width";
  case ConSanInventoryExclusionReason::MissingAddressOperand:
    return "missing-address-operand";
  case ConSanInventoryExclusionReason::RangeEncodingUnavailable:
    return "range-encoding-unavailable";
  case ConSanInventoryExclusionReason::Count:
    break;
  }
  return "invalid-inventory-exclusion-reason";
}

static_assert(kConSanInventoryExclusionReasons.size() ==
              static_cast<size_t>(ConSanInventoryExclusionReason::Count));

/// One machine-readable limitation attached to a normalized access site.
///
/// `detail` may explain the offending encoding for diagnostics, but control
/// flow must use `reason`; prose is deliberately not a hidden policy channel.
struct ConSanInventoryExclusion {
  /// Stable typed limitation.
  ConSanInventoryExclusionReason reason = ConSanInventoryExclusionReason::Count;
  /// Optional human-readable detail with no semantic control-flow meaning.
  std::string detail;

  bool operator==(const ConSanInventoryExclusion &) const = default;
};

/// One normalized byte range accessed by an original instruction.
///
/// The offset is relative to the instruction's dynamic base address when the
/// encoding makes that relationship available. Its absence means “not decoded
/// into this legacy inventory,” not an offset of zero. Width is expressed in
/// bytes so subword and wide accesses share one unit.
struct ConSanAccessRange {
  /// Stable semantic identity of this exact logical range.
  SemanticSiteId id;
  /// Static byte displacement from the dynamic base address, when known.
  std::optional<int64_t> static_byte_offset;
  /// Nonzero logical width of this range in bytes.
  uint32_t byte_width = 0;

  bool operator==(const ConSanAccessRange &) const = default;
};

/// Decoder operands retained by normalized access inventory for later policy
/// and compatibility lowering.
///
/// Optional fields preserve “not encoded/not decoded” rather than inventing a
/// register or immediate. Raw FLAT fields remain target-neutral facts here;
/// interpreting whether a particular combination is instrumentable belongs to
/// engine policy and target lowering.
struct ConSanAccessOperandFacts {
  std::optional<uint16_t> destination_vgpr;
  std::optional<uint16_t> destination_accvgpr;
  std::optional<uint16_t> address_vgpr;
  std::optional<uint16_t> data_vgpr;
  std::optional<uint16_t> second_data_vgpr;
  std::optional<uint32_t> raw_op;
  std::optional<uint32_t> raw_saddr;
  std::optional<bool> raw_scale_offset;
  std::optional<uint32_t> raw_vaddr;
  std::optional<uint32_t> raw_vsrc;
  std::optional<uint32_t> raw_vdst;
  std::optional<int32_t> raw_ioffset;
  std::optional<uint32_t> raw_segment;
  std::optional<uint32_t> raw_scope;
  std::optional<uint32_t> raw_th;

  bool operator==(const ConSanAccessOperandFacts &) const = default;
};

/// Immutable normalized access fact for one container interpretation of a
/// physical instruction.
///
/// Aliasing symbol ranges can produce several records with the same
/// `physical_id`; later policy canonicalizes them before placement. The record
/// preserves original decoder facts, normalized logical ranges, execution
/// owners, confidence, and typed limitations. It contains no selected engine,
/// register, report address, patch kind, or lowering outcome.
struct ConSanAccessInventorySite {
  PhysicalSiteId physical_id;
  ConSanProgramContainerRef container;
  ConSanAccessOrigin origin = ConSanAccessOrigin::Count;
  ConSanLdsAccessKind kind = ConSanLdsAccessKind::Other;
  ConSanAccessAddressSpace address_space = ConSanAccessAddressSpace::Unresolved;
  ConSanAccessProvenance provenance = ConSanAccessProvenance::Count;
  ConSanSemanticConfidence confidence = ConSanSemanticConfidence::Unsupported;
  bool supported_mvp = false;
  uint64_t file_offset = 0;
  uint32_t instruction_size = 0;
  uint32_t decoded_width_bits = 0;
  std::string mnemonic;
  ConSanFlatAddressSpaceHint flat_address_space_hint = ConSanFlatAddressSpaceHint::Unknown;
  ConSanAccessOperandFacts operands;
  std::vector<uint64_t> execution_owner_descriptor_file_offsets;
  std::vector<ConSanAccessRange> ranges;
  std::vector<ConSanInventoryExclusion> exclusions;

  /// Return whether construction found no inventory-level limitation. Engine
  /// policy can still reject a complete semantic access for mode-specific
  /// reasons.
  [[nodiscard]] bool complete() const { return exclusions.empty() && !ranges.empty(); }

  bool operator==(const ConSanAccessInventorySite &) const = default;
};

/// Immutable synchronization and ownership projection produced by semantic
/// analysis.
///
/// Events retain physical and typed semantic identities; sequences and
/// lifecycle groups retain ordered member identities; communication recipes
/// and fence candidates retain the bounded facts derived from those sequences.
/// The view owns no storage and remains valid while its `ProgramInventory`
/// owner is alive. Fault planning and engine policy may query this projection
/// but cannot modify or rebuild sanitizer semantics through it.
struct SynchronizationInventoryView {
  std::span<const ConSanSyncEvent> sync_events;
  std::span<const ConSanSyncSequence> sync_sequences;
  std::span<const ConSanBarrierLifecycleGroup> barrier_lifecycle_groups;
  std::span<const ConSanCommunicationAddressRecipe> communication_address_recipes;
  std::span<const ConSanMoiFenceCandidate> moi_fence_candidates;

  /// Return whether semantic analysis published no synchronization-derived
  /// records of any kind.
  [[nodiscard]] bool empty() const {
    return sync_events.empty() && sync_sequences.empty() && barrier_lifecycle_groups.empty() &&
           communication_address_recipes.empty() && moi_fence_candidates.empty();
  }
};

/// Mutable construction capability for synchronization inventory during the
/// single semantic-analysis stage.
///
/// References point into one `ProgramInventoryBuilder`. Producer passes receive
/// this value explicitly, which makes mutation visible in their signatures and
/// prevents fault planners or lowerers from acquiring write access. The build
/// view must not escape analysis; published consumers receive only
/// `SynchronizationInventoryView`.
struct SynchronizationInventoryBuildView {
  std::vector<ConSanSyncEvent> &sync_events;
  std::vector<ConSanSyncSequence> &sync_sequences;
  std::vector<ConSanBarrierLifecycleGroup> &barrier_lifecycle_groups;
  std::vector<ConSanCommunicationAddressRecipe> &communication_address_recipes;
  std::vector<ConSanMoiFenceCandidate> &moi_fence_candidates;
};

/// Internal shared storage owned by `ProgramInventory` and populated only by
/// `ProgramInventoryBuilder` during the analysis stage.
///
/// This type is named so ownership is explicit in profiles and tests. It is
/// not a semantic API: consumers use the immutable accessors on
/// `ProgramInventory`.
struct ConSanProgramInventoryStorage {
  ConSanCodeObjectId code_object_id;
  bool kernel_metadata_trustworthy = false;
  size_t malformed_kernel_metadata_note_count = 0;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  bool semantic_arch_required = false;
  std::vector<ConSanTextSection> text_sections;
  std::vector<ConSanKernelInfo> kernels;
  std::vector<ConSanFunctionInfo> functions;
  std::vector<ConSanAccessInventorySite> access_sites;
  std::vector<ConSanSyncEvent> sync_events;
  std::vector<ConSanSyncSequence> sync_sequences;
  std::vector<ConSanBarrierLifecycleGroup> barrier_lifecycle_groups;
  std::vector<ConSanCommunicationAddressRecipe> communication_address_recipes;
  std::vector<ConSanMoiFenceCandidate> moi_fence_candidates;
};

/// Immutable owner and query interface for original code-object facts used by
/// ConSan semantic policy.
///
/// Copies share const storage, so site IDs and references remain stable while
/// results move through retries and lowering. Only `ProgramInventoryBuilder`
/// can populate the storage during a transform. The public interface exposes
/// original-program facts plus normalized access and synchronization
/// semantics; it deliberately excludes engine admission, report layout,
/// registers, patches, diagnostics, and mutable transformation state.
class ProgramInventory {
public:
  ProgramInventory() = default;

  [[nodiscard]] bool empty() const {
    return !storage_ || storage_->code_object_id.fingerprint.empty();
  }
  [[nodiscard]] const ConSanCodeObjectId &code_object_id() const {
    return storage_ ? storage_->code_object_id : empty_storage().code_object_id;
  }
  [[nodiscard]] bool kernel_metadata_trustworthy() const {
    return storage_ && storage_->kernel_metadata_trustworthy;
  }
  [[nodiscard]] size_t malformed_kernel_metadata_note_count() const {
    return storage_ ? storage_->malformed_kernel_metadata_note_count : 0;
  }
  [[nodiscard]] rj_code_arch_t arch() const {
    return storage_ ? storage_->arch : ROCJITSU_CODE_ARCH_INVALID;
  }
  [[nodiscard]] rj_code_target_id_t target() const {
    return storage_ ? storage_->target : ROCJITSU_CODE_TARGET_INVALID;
  }
  [[nodiscard]] bool semantic_arch_required() const {
    return storage_ && storage_->semantic_arch_required;
  }
  [[nodiscard]] std::span<const ConSanTextSection> text_sections() const {
    return storage_ ? std::span<const ConSanTextSection>(storage_->text_sections)
                    : std::span<const ConSanTextSection>{};
  }
  [[nodiscard]] std::span<const ConSanKernelInfo> kernels() const {
    return storage_ ? std::span<const ConSanKernelInfo>(storage_->kernels)
                    : std::span<const ConSanKernelInfo>{};
  }
  [[nodiscard]] std::span<const ConSanFunctionInfo> functions() const {
    return storage_ ? std::span<const ConSanFunctionInfo>(storage_->functions)
                    : std::span<const ConSanFunctionInfo>{};
  }
  [[nodiscard]] std::span<const ConSanAccessInventorySite> access_sites() const {
    return storage_ ? std::span<const ConSanAccessInventorySite>(storage_->access_sites)
                    : std::span<const ConSanAccessInventorySite>{};
  }
  [[nodiscard]] SynchronizationInventoryView synchronization_view() const {
    if (!storage_)
      return {};
    return {
        .sync_events = storage_->sync_events,
        .sync_sequences = storage_->sync_sequences,
        .barrier_lifecycle_groups = storage_->barrier_lifecycle_groups,
        .communication_address_recipes = storage_->communication_address_recipes,
        .moi_fence_candidates = storage_->moi_fence_candidates,
    };
  }

private:
  friend class ProgramInventoryBuilder;

  explicit ProgramInventory(std::shared_ptr<const ConSanProgramInventoryStorage> storage)
      : storage_(std::move(storage)) {}

  [[nodiscard]] static const ConSanProgramInventoryStorage &empty_storage() {
    static const ConSanProgramInventoryStorage storage;
    return storage;
  }

  std::shared_ptr<const ConSanProgramInventoryStorage> storage_;
};

/// Construction capability for one `ProgramInventory` analysis stage.
///
/// The builder is the only mutable owner during decode/provenance/ownership
/// analysis. `view()` may publish a const facade while construction continues
/// so legacy consumers can read containers, but the builder must not escape
/// the transform. Once it is destroyed, every surviving inventory reference
/// is immutable. `rebuild_access_inventory` is called after any analysis that
/// enriches raw access or ownership facts and deterministically replaces only
/// the derived normalized-access projection.
class ProgramInventoryBuilder {
public:
  /// Start a fresh inventory for one pristine code-object image.
  explicit ProgramInventoryBuilder(std::span<const uint8_t> code_object_bytes = {})
      : storage_(std::make_shared<ConSanProgramInventoryStorage>()) {
    storage_->code_object_id = make_consan_code_object_id(code_object_bytes);
  }

  /// Start a distinct mutable analysis revision from an immutable inventory.
  ///
  /// Storage is deep-copied: publishing or modifying the revision cannot
  /// mutate any existing `ProgramInventory` view. This operation supports
  /// component-sized reanalysis and adversarial tests; ordinary initial
  /// analysis should use the pristine-byte constructor above.
  explicit ProgramInventoryBuilder(const ProgramInventory &source)
      : storage_(std::make_shared<ConSanProgramInventoryStorage>(
            source.storage_ ? *source.storage_ : ConSanProgramInventoryStorage{})) {}

  /// Publish an immutable shared view of the builder's current revision.
  [[nodiscard]] ProgramInventory view() const { return ProgramInventory(storage_); }

  [[nodiscard]] std::vector<ConSanTextSection> &text_sections() { return storage_->text_sections; }
  [[nodiscard]] std::vector<ConSanKernelInfo> &kernels() { return storage_->kernels; }
  [[nodiscard]] std::vector<ConSanFunctionInfo> &functions() { return storage_->functions; }
  /// Return the analysis-only capability for synchronization-derived facts.
  [[nodiscard]] SynchronizationInventoryBuildView synchronization() {
    return {
        .sync_events = storage_->sync_events,
        .sync_sequences = storage_->sync_sequences,
        .barrier_lifecycle_groups = storage_->barrier_lifecycle_groups,
        .communication_address_recipes = storage_->communication_address_recipes,
        .moi_fence_candidates = storage_->moi_fence_candidates,
    };
  }

  void set_code_object_facts(bool kernel_metadata_trustworthy,
                             size_t malformed_kernel_metadata_note_count, rj_code_arch_t arch,
                             rj_code_target_id_t target) {
    storage_->kernel_metadata_trustworthy = kernel_metadata_trustworthy;
    storage_->malformed_kernel_metadata_note_count = malformed_kernel_metadata_note_count;
    storage_->arch = arch;
    storage_->target = target;
  }

  void set_semantic_arch_required(bool required) { storage_->semantic_arch_required = required; }

  /// Rebuild normalized access facts from the builder's current decoded
  /// containers. Existing `ProgramInventory` views observe the replacement,
  /// but no returned access reference may be retained across this builder-only
  /// operation.
  void rebuild_access_inventory(std::span<const uint8_t> code_object_bytes) {
    storage_->access_sites.clear();
    for (const ConSanKernelInfo &kernel : storage_->kernels)
      append_container_accesses(kernel, code_object_bytes);
    for (const ConSanFunctionInfo &function : storage_->functions)
      append_container_accesses(function, code_object_bytes);
  }

private:
  [[nodiscard]] static bool is_two_address_lds(std::string_view mnemonic) {
    return mnemonic == "ds_store_2addr_b32" || mnemonic == "ds_store_2addr_b64" ||
           mnemonic == "ds_store_2addr_stride64_b32" || mnemonic == "ds_store_2addr_stride64_b64" ||
           mnemonic == "ds_write2_b32" || mnemonic == "ds_write2st64_b32" ||
           mnemonic == "ds_write2_b64" || mnemonic == "ds_write2st64_b64" ||
           mnemonic == "ds_load_2addr_b32" || mnemonic == "ds_load_2addr_b64" ||
           mnemonic == "ds_load_2addr_stride64_b32" || mnemonic == "ds_load_2addr_stride64_b64" ||
           mnemonic == "ds_read2_b32" || mnemonic == "ds_read2st64_b32" ||
           mnemonic == "ds_read2_b64" || mnemonic == "ds_read2st64_b64";
  }

  [[nodiscard]] static std::optional<uint32_t>
  two_address_element_width_bits(std::string_view mnemonic) {
    if (mnemonic.ends_with("b32"))
      return 32u;
    if (mnemonic.ends_with("b64"))
      return 64u;
    return std::nullopt;
  }

  [[nodiscard]] static std::optional<uint32_t>
  two_address_offset_scale_bytes(std::string_view mnemonic) {
    if (mnemonic == "ds_load_2addr_b32" || mnemonic == "ds_store_2addr_b32" ||
        mnemonic == "ds_read2_b32" || mnemonic == "ds_write2_b32")
      return 4u;
    if (mnemonic == "ds_load_2addr_b64" || mnemonic == "ds_store_2addr_b64" ||
        mnemonic == "ds_read2_b64" || mnemonic == "ds_write2_b64")
      return 8u;
    if (mnemonic == "ds_load_2addr_stride64_b32" || mnemonic == "ds_store_2addr_stride64_b32" ||
        mnemonic == "ds_read2st64_b32" || mnemonic == "ds_write2st64_b32")
      return 256u;
    if (mnemonic == "ds_load_2addr_stride64_b64" || mnemonic == "ds_store_2addr_stride64_b64" ||
        mnemonic == "ds_read2st64_b64" || mnemonic == "ds_write2st64_b64")
      return 512u;
    return std::nullopt;
  }

  [[nodiscard]] static std::optional<uint32_t>
  read_little_endian_u32(std::span<const uint8_t> bytes, uint64_t offset) {
    if (offset > bytes.size() || sizeof(uint32_t) > bytes.size() - offset)
      return std::nullopt;
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8u) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16u) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24u);
  }

  /// Decode the static byte displacement of a one-range native LDS access.
  /// CDNA3/4 DS atomics use the low eight-bit offset field; other admitted
  /// one-range DS/VDS forms use the low sixteen bits. Direct-to-LDS transfers
  /// address their destination through M0 and therefore contribute no encoded
  /// displacement. Returning absence preserves unreadable pristine bytes as a
  /// typed inventory gap instead of silently treating them as offset zero.
  [[nodiscard]] std::optional<int64_t>
  single_range_native_static_byte_offset(const ConSanLdsSite &site,
                                         std::span<const uint8_t> bytes) const {
    if (site.direct_to_lds)
      return 0;
    const std::optional<uint32_t> word0 = read_little_endian_u32(bytes, site.file_offset);
    if (!word0)
      return std::nullopt;
    const bool cdna3_or_cdna4 =
        storage_->arch == ROCJITSU_CODE_ARCH_CDNA3 || storage_->arch == ROCJITSU_CODE_ARCH_CDNA4;
    const uint32_t mask =
        cdna3_or_cdna4 && site.kind == ConSanLdsAccessKind::Atomic ? 0xffu : 0xffffu;
    return static_cast<int64_t>(*word0 & mask);
  }

  [[nodiscard]] SemanticSiteId access_range_id(uint64_t text_offset, uint32_t range_ordinal) const {
    return {
        .physical = {.code_object = storage_->code_object_id, .original_text_offset = text_offset},
        .domain = ConSanSemanticSiteDomain::Access,
        .member_ordinal = 0,
        .range_ordinal = range_ordinal,
    };
  }

  template <typename Container>
  [[nodiscard]] static ConSanProgramContainerRef container_ref(const Container &container) {
    ConSanProgramContainerRef result;
    result.name = container.name;
    result.entry_text_offset = container.entry_text_offset;
    result.uses_gfx1250_cluster_workgroup_id = container.uses_gfx1250_cluster_workgroup_id;
    if constexpr (std::is_same_v<Container, ConSanKernelInfo>) {
      result.kind = ConSanProgramContainerKind::Kernel;
      result.kernel_descriptor_file_offset = container.descriptor_file_offset;
    } else {
      result.kind = ConSanProgramContainerKind::Function;
    }
    return result;
  }

  [[nodiscard]] ConSanAccessInventorySite
  make_access_site(const ConSanProgramContainerRef &container, const ConSanLdsSite &site,
                   std::span<const uint8_t> bytes) const {
    ConSanAccessInventorySite result;
    result.physical_id = {
        .code_object = storage_->code_object_id,
        .original_text_offset = site.text_offset,
    };
    result.container = container;
    result.origin =
        site.direct_to_lds ? ConSanAccessOrigin::DirectToLds : ConSanAccessOrigin::NativeLds;
    result.kind = site.kind;
    result.address_space = ConSanAccessAddressSpace::Group;
    result.provenance = ConSanAccessProvenance::NativeLdsOpcode;
    result.confidence = ConSanSemanticConfidence::Exact;
    result.supported_mvp = site.supported_mvp;
    result.file_offset = site.file_offset;
    result.instruction_size = site.size;
    result.decoded_width_bits = site.width_bits;
    result.mnemonic = site.mnemonic;
    result.operands.destination_vgpr = site.dst_vgpr;
    result.operands.destination_accvgpr = site.dst_accvgpr;
    result.operands.address_vgpr = site.addr_vgpr;
    result.operands.data_vgpr = site.data_vgpr;
    result.operands.second_data_vgpr = site.second_data_vgpr;
    result.execution_owner_descriptor_file_offsets = site.owner_descriptor_file_offsets;
    append_common_exclusions(result, site.direct_to_lds);
    if (is_two_address_lds(site.mnemonic)) {
      const auto width = two_address_element_width_bits(site.mnemonic);
      const auto scale = two_address_offset_scale_bytes(site.mnemonic);
      const auto word0 = read_little_endian_u32(bytes, site.file_offset);
      if (width && scale && word0) {
        result.ranges.push_back({.id = access_range_id(site.text_offset, 0),
                                 .static_byte_offset = (*word0 & 0xffu) * *scale,
                                 .byte_width = *width / 8u});
        result.ranges.push_back({.id = access_range_id(site.text_offset, 1),
                                 .static_byte_offset = ((*word0 >> 8u) & 0xffu) * *scale,
                                 .byte_width = *width / 8u});
      } else {
        result.exclusions.push_back({
            .reason = ConSanInventoryExclusionReason::RangeEncodingUnavailable,
            .detail = "two-address LDS offsets could not be decoded",
        });
      }
    } else if (site.width_bits != 0 && site.width_bits % 8u == 0) {
      result.ranges.push_back(
          {.id = access_range_id(site.text_offset, 0),
           .static_byte_offset = single_range_native_static_byte_offset(site, bytes),
           .byte_width = site.width_bits / 8u});
    }
    return result;
  }

  [[nodiscard]] ConSanAccessInventorySite
  make_access_site(const ConSanProgramContainerRef &container, const ConSanFlatSite &site) const {
    const auto [address_space, provenance, confidence] = flat_provenance(site.address_space_hint);
    ConSanAccessInventorySite result;
    result.physical_id = {
        .code_object = storage_->code_object_id,
        .original_text_offset = site.text_offset,
    };
    result.container = container;
    result.origin = ConSanAccessOrigin::Flat;
    result.kind = site.kind;
    result.address_space = address_space;
    result.provenance = provenance;
    result.confidence = confidence;
    result.file_offset = site.file_offset;
    result.instruction_size = site.size;
    result.decoded_width_bits = site.width_bits;
    result.mnemonic = site.mnemonic;
    result.flat_address_space_hint = site.address_space_hint;
    result.operands.destination_vgpr = site.dst_vgpr;
    result.operands.address_vgpr = site.addr_vgpr;
    result.operands.data_vgpr = site.data_vgpr;
    result.operands.raw_op = site.raw_op;
    result.operands.raw_saddr = site.raw_saddr;
    result.operands.raw_scale_offset = site.raw_scale_offset;
    result.operands.raw_vaddr = site.raw_vaddr;
    result.operands.raw_vsrc = site.raw_vsrc;
    result.operands.raw_vdst = site.raw_vdst;
    result.operands.raw_ioffset = site.raw_ioffset;
    result.operands.raw_segment = site.raw_segment;
    result.operands.raw_scope = site.raw_scope;
    result.operands.raw_th = site.raw_th;
    result.execution_owner_descriptor_file_offsets = site.owner_descriptor_file_offsets;
    append_common_exclusions(result, false);
    if (site.width_bits != 0 && site.width_bits % 8u == 0) {
      result.ranges.push_back({.id = access_range_id(site.text_offset, 0),
                               .static_byte_offset = site.raw_ioffset,
                               .byte_width = site.width_bits / 8u});
    }
    return result;
  }

  static void append_common_exclusions(ConSanAccessInventorySite &site,
                                       bool address_operand_not_required) {
    if (site.kind == ConSanLdsAccessKind::Other) {
      site.exclusions.push_back({
          .reason = ConSanInventoryExclusionReason::NonAccessInstruction,
          .detail = "decoded instruction is outside ConSan's access kinds",
      });
    }
    if (site.instruction_size == 0 || site.instruction_size % sizeof(uint32_t) != 0) {
      site.exclusions.push_back({
          .reason = ConSanInventoryExclusionReason::InvalidInstructionSize,
          .detail = "decoded instruction size is not a nonzero dword multiple",
      });
    }
    if (site.decoded_width_bits == 0 || site.decoded_width_bits % 8u != 0) {
      site.exclusions.push_back({
          .reason = ConSanInventoryExclusionReason::InvalidAccessWidth,
          .detail = "decoded access width is not a nonzero byte multiple",
      });
    }
    if (!address_operand_not_required && !site.operands.address_vgpr) {
      site.exclusions.push_back({
          .reason = ConSanInventoryExclusionReason::MissingAddressOperand,
          .detail = "decoded access has no vector address operand",
      });
    }
  }

  /// Internal tuple returned while normalizing a legacy FLAT provenance hint.
  /// It keeps address-space, source, and confidence assignments adjacent so a
  /// new hint cannot update one semantic dimension while silently leaving the
  /// others at an unrelated default.
  struct FlatProvenanceFacts {
    ConSanAccessAddressSpace address_space;
    ConSanAccessProvenance provenance;
    ConSanSemanticConfidence confidence;
  };

  [[nodiscard]] static FlatProvenanceFacts flat_provenance(ConSanFlatAddressSpaceHint hint) {
    switch (hint) {
    case ConSanFlatAddressSpaceHint::Group:
      return {ConSanAccessAddressSpace::Group, ConSanAccessProvenance::EncodedFlatSegment,
              ConSanSemanticConfidence::Exact};
    case ConSanFlatAddressSpaceHint::Private:
    case ConSanFlatAddressSpaceHint::Global:
      return {ConSanAccessAddressSpace::NonGroup, ConSanAccessProvenance::EncodedFlatSegment,
              ConSanSemanticConfidence::Exact};
    case ConSanFlatAddressSpaceHint::MaybeGroup:
      return {ConSanAccessAddressSpace::Group, ConSanAccessProvenance::PropagatedFlatPointer,
              ConSanSemanticConfidence::Conservative};
    case ConSanFlatAddressSpaceHint::MaybePrivate:
      return {ConSanAccessAddressSpace::NonGroup, ConSanAccessProvenance::PropagatedFlatPointer,
              ConSanSemanticConfidence::Conservative};
    case ConSanFlatAddressSpaceHint::Unknown:
      return {ConSanAccessAddressSpace::Unresolved, ConSanAccessProvenance::UnresolvedFlatPointer,
              ConSanSemanticConfidence::Ambiguous};
    }
    return {ConSanAccessAddressSpace::Unresolved, ConSanAccessProvenance::UnresolvedFlatPointer,
            ConSanSemanticConfidence::Unsupported};
  }

  template <typename Container>
  void append_container_accesses(const Container &container,
                                 std::span<const uint8_t> code_object_bytes) {
    const ConSanProgramContainerRef reference = container_ref(container);
    storage_->access_sites.reserve(storage_->access_sites.size() + container.lds_sites.size() +
                                   container.flat_sites.size());
    for (const ConSanLdsSite &site : container.lds_sites)
      storage_->access_sites.push_back(make_access_site(reference, site, code_object_bytes));
    for (const ConSanFlatSite &site : container.flat_sites)
      storage_->access_sites.push_back(make_access_site(reference, site));
  }

  std::shared_ptr<ConSanProgramInventoryStorage> storage_;
};
